# mcts/mcts_joint.py

import functools

import jax
import jax.numpy as jnp
import chex
import mctx

from model.model import FlaxMAMuZeroNet
import utils.utils as utils
from config import ExperimentConfig
from mcts.base import MCTSPlanner, MCTSPlanOutput


class MCTSJointPlanner(MCTSPlanner):
    """
    Joint MCTS: searches over the full combinatorial joint action space A^N.

    Per-agent policy logits are combined under an independence assumption
    (log P(a1,...,aN) = Σ log P(ai)) to produce joint action logits for mctx.
    Gumbel sampling keeps the number of evaluated actions tractable even when
    A^N is large.
    """

    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        super().__init__(model, config)
        # Pre-compute the shape tuple used for indexing into the joint action space.
        # e.g. 3 agents with 5 actions each → (5, 5, 5)
        self.joint_action_shape: tuple[int, ...] = (self.action_space_size,) * self.num_agents

    def _recurrent_fn(
        self,
        params,
        rng_key: chex.Array,
        action: chex.Array,
        embedding: chex.Array,
    ) -> tuple[mctx.RecurrentFnOutput, chex.Array]:
        """
        One simulation step for the joint search.

        Args:
            action:    Flat joint action index into A^N. Shape: (B,)
            embedding: Current latent state. Shape: (B, N, D)
        """
        latent = embedding

        # Decode the flat joint action index into per-agent actions.
        joint_action_tuple = jnp.unravel_index(action, self.joint_action_shape)
        joint_action = jnp.array(joint_action_tuple).T.reshape(latent.shape[0], self.num_agents)

        model_output = self.model.apply(
            {"params": params},
            latent,
            joint_action,
            method=self.model.recurrent_inference,
            rngs={"dropout": rng_key},
        )

        value = utils.support_to_scalar(model_output.value_logits, self.value_support)
        reward = utils.support_to_scalar(model_output.reward_logits, self.reward_support)
        joint_logits = self._logits_to_joint_logits(model_output.policy_logits)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, self.discount_gamma),
                prior_logits=joint_logits,
                value=value,
            ),
            model_output.hidden_state,
        )

    def _plan_loop(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """Runs a single joint MCTS search and decodes the result into per-agent actions."""
        init_key, gumbel_key = jax.random.split(rng_key)

        model_output = self.model.apply(
            {"params": params}, observation, rngs={"dropout": init_key}
        )
        root_value = utils.support_to_scalar(model_output.value_logits, self.value_support)
        root_joint_logits = self._logits_to_joint_logits(model_output.policy_logits)

        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=gumbel_key,
            root=mctx.RootFnOutput(
                prior_logits=root_joint_logits,
                value=root_value,
                embedding=model_output.hidden_state,
            ),
            recurrent_fn=self._recurrent_fn_jit,
            num_simulations=self.num_simulations,
            max_depth=self.max_depth_gumbel_search,
            max_num_considered_actions=self.num_gumbel_samples,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value, use_mixed_value=True
            ),
        )

        # Decode the chosen flat joint action index back to per-agent actions.
        joint_action_tuple = jnp.unravel_index(
            policy_output.action, self.joint_action_shape
        )
        final_joint_action = jnp.array(joint_action_tuple).squeeze(axis=-1)

        # Convert the joint policy target distribution to per-agent marginals for training.
        marginal_policy_targets = self._joint_policy_to_marginal(
            policy_output.action_weights[None, :]
        ).squeeze(0)

        return MCTSPlanOutput(
            joint_action=final_joint_action,
            policy_targets=marginal_policy_targets,
            root_value=root_value.squeeze().astype(float),
            agent_order=jnp.arange(self.num_agents),
        )

    def _logits_to_joint_logits(self, logits: chex.Array) -> chex.Array:
        """
        Converts per-agent logits (B, N, A) to joint action logits (B, A^N).

        Assumes agent policies are independent:
            log P(a1, ..., aN) = Σ_i log P(ai)

        The outer product is computed iteratively, expanding one agent at a time.
        """
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, N, A)
        joint = log_probs[:, 0, :]                        # (B, A)
        for i in range(1, self.num_agents):
            # Expand to (B, A^i, A) and sum, then flatten to (B, A^(i+1))
            joint = (joint[:, :, None] + log_probs[:, i, None, :]).reshape(logits.shape[0], -1)
        return joint

    def _joint_policy_to_marginal(self, joint_policy: chex.Array) -> chex.Array:
        """
        Converts a joint policy distribution (B, A^N) to per-agent marginals (B, N, A).

        Agent i's marginal is obtained by summing over all other agents' action axes.
        """
        batch_size = joint_policy.shape[0]
        reshaped = joint_policy.reshape(batch_size, *self.joint_action_shape)  # (B, A, A, ..., A)

        marginals = []
        for i in range(self.num_agents):
            # Axes for all agents except i (offset by 1 for the batch dim).
            other_axes = tuple(j + 1 for j in range(self.num_agents) if j != i)
            marginals.append(jnp.sum(reshaped, axis=other_axes))  # (B, A)

        return jnp.stack(marginals, axis=1)  # (B, N, A)
