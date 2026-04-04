# mcts/mcts_independent.py

import functools

import jax
import jax.numpy as jnp
import chex
import mctx

from model import FlaxMAMuZeroNet
import utils.transforms as utils
from config import ExperimentConfig
from mcts.base import MCTSPlanner, MCTSPlanOutput


class MCTSIndependentPlanner(MCTSPlanner):
    """
    Independent MCTS: runs a separate Gumbel MuZero search for each agent.

    During each agent's search, all other agents' actions are fixed to either
    the argmax or a sample from their current policy network. This avoids the
    exponential joint action space at the cost of ignoring inter-agent coordination.
    """

    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        super().__init__(model, config)
        self.independent_argmax = config.mcts.independent_argmax

    def _recurrent_fn(
        self,
        params,
        rng_key: chex.Array,
        action: chex.Array,
        embedding: tuple[chex.Array, chex.Array],
    ) -> tuple[mctx.RecurrentFnOutput, tuple]:
        """
        One simulation step for the searching agent.

        Other agents' actions are determined by their policy network (argmax or sample),
        then the searching agent's action is inserted at its index before running dynamics.

        Args:
            action:    The action being explored for the current agent. Shape: (B,)
            embedding: (latent, agent_idx) where latent is (B, N, D) and agent_idx is (B,)
        """
        latent, agent_idx = embedding
        batch_size = latent.shape[0]
        batch_indices = jnp.arange(batch_size)

        # Get current policy for all agents to fill in the non-searching agents' actions.
        prior_logits, _ = self.model.apply(
            {"params": params}, latent, method=self.model.predict
        )
        if self.independent_argmax:
            actions = jnp.argmax(prior_logits, axis=-1)          # (B, N)
        else:
            rng_key, sample_key = jax.random.split(rng_key)
            actions = jax.random.categorical(sample_key, prior_logits)  # (B, N)

        # Override the searching agent's slot with the action being evaluated.
        joint_action = actions.at[batch_indices, agent_idx].set(action)  # (B, N)

        model_output = self.model.apply(
            {"params": params},
            latent,
            joint_action,
            method=self.model.recurrent_inference,
            rngs={"dropout": rng_key},
        )

        value = utils.support_to_scalar(model_output.value_logits, self.value_support)
        reward = utils.support_to_scalar(model_output.reward_logits, self.reward_support)
        # Prior for the next node is the searching agent's policy at the next state.
        prior = model_output.policy_logits[batch_indices, agent_idx]  # (B, A)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, self.discount_gamma),
                prior_logits=prior,
                value=value,
            ),
            (model_output.hidden_state, agent_idx),
        )

    def _plan_loop(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """
        Runs one independent MCTS search per agent via jax.lax.scan.

        Each agent searches over its own action space with other agents held fixed.
        The scan carries no state between agents — each search starts from the same root.
        """
        init_key, rng_key = jax.random.split(rng_key)
        model_output = self.model.apply(
            {"params": params}, observation, rngs={"dropout": init_key}
        )
        root_latent = model_output.hidden_state          # (B, N, D)
        root_logits = model_output.policy_logits         # (B, N, A)
        root_value = utils.support_to_scalar(model_output.value_logits, self.value_support)

        # One rng key and index per agent, scanned in order.
        keys = jax.random.split(rng_key, self.num_agents)       # (N, 2)
        idxs = jnp.arange(self.num_agents, dtype=jnp.int32)    # (N,)

        def agent_step(carry, inputs):
            key, agent = inputs
            # Add exploration noise to this agent's root prior.
            mcts_key, noisy_logits = self.add_dirichlet_noise(key, root_logits[:, agent, :])
            embedding = (root_latent, jnp.array([agent], jnp.int32))

            out = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=mcts_key,
                root=mctx.RootFnOutput(
                    prior_logits=noisy_logits,
                    value=root_value,
                    embedding=embedding,
                ),
                recurrent_fn=self._recurrent_fn_jit,
                num_simulations=self.num_simulations,
                max_depth=self.max_depth_gumbel_search,
                max_num_considered_actions=self.num_gumbel_samples,
                qtransform=functools.partial(
                    mctx.qtransform_completed_by_mix_value, use_mixed_value=True
                ),
            )
            search_value = out.search_tree.summary().value  # (B,)
            return carry, (out.action, out.action_weights, search_value)

        # carry=None because each agent's search is independent — no state flows between them.
        _, results = jax.lax.scan(agent_step, None, (keys, idxs))
        actions, weights, search_values = results
        # actions: (N, B) — scan stacks per-agent outputs along axis 0
        # weights: (N, B, A)
        # search_values: (N, B)

        return MCTSPlanOutput(
            joint_action=jnp.moveaxis(actions, 0, 1),       # (B, N)
            policy_targets=jnp.moveaxis(weights, 0, 1),     # (B, N, A)
            root_value=jnp.mean(search_values, axis=0),     # (B,)
            agent_order=jnp.arange(self.num_agents),        # (N,)
        )
