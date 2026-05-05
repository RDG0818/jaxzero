import os
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxzero.config import MAZeroConfig
from jaxzero.model.transforms import phi_inv as _phi_inv

_ctree_dir = os.path.join(os.path.dirname(__file__), "ctree")
if _ctree_dir not in sys.path:
    sys.path.insert(0, _ctree_dir)
from cytree import Tree_batch


class SearchOutput(NamedTuple):
    root_value: np.ndarray        # (B,)
    sampled_actions: list         # list[B] of (K, N)
    sampled_visit_counts: list    # list[B] of (K,)
    sampled_qvalues: list         # list[B] of (K,)
    sampled_imp_ratio: list       # list[B] of (K,)


class SampledMCTS:
    """Sampled MCTS backed by the C++/Cython ctree from MAZero.

    Tree topology, OS(λ), and UCB selection run in C++. JAX model calls
    remain batched across all B environments per simulation step.
    """

    def __init__(self, config: MAZeroConfig, model):
        self.config = config
        self.model = model

        self._jit_initial = jax.jit(model.apply)
        
        def _recurrent_step(params, hidden_pool, ix, batch_actions):
            # hidden_pool: (N_pool, B, N_agents, D)
            # ix: (B,) indices into N_pool
            # Gather hidden states: (B, N_agents, D)
            # Use jnp.take to index into the first dimension (pool)
            h = jax.vmap(lambda p, i: p[i])(hidden_pool.transpose(1, 0, 2, 3), ix)
            out = model.apply(params, h, batch_actions, method=model.recurrent_inference)
            return out

        self._jit_recurrent = jax.jit(_recurrent_step)

        value_support = config.value_support_size
        reward_support = config.reward_support_size
        self._jit_val = jax.jit(lambda logits: _phi_inv(logits, value_support))
        self._jit_rew = jax.jit(lambda logits: _phi_inv(logits, reward_support))

    # ------------------------------------------------------------------

    @staticmethod
    def _softmax_legal(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """Softmax over logits with legal-action masking. (B, N, A) → (B, N, A)"""
        log_probs = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(log_probs)
        probs = probs * legal.astype(np.float32)
        probs += legal.astype(np.float32) * 1e-4
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs.astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Softmax without masking. (B, N, A) → (B, N, A)"""
        log_probs = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs.astype(np.float32)

    # ------------------------------------------------------------------

    def search(
        self,
        params,
        obs: np.ndarray,    # (B, N, obs_dim)
        legal: np.ndarray,  # (B, N, A)
        rng: np.random.Generator,
    ) -> SearchOutput:
        """Run Sampled MCTS for a batch of B environments."""
        cfg = self.config
        B = obs.shape[0]
        K = cfg.sampled_action_times

        # --- Initial inference ---
        out0 = self._jit_initial(params, jnp.array(obs))
        values0 = np.array(self._jit_val(out0.value_logits), dtype=np.float32)   # (B,)
        policy0 = np.array(out0.policy_logits)                                    # (B, N, A)
        hidden0 = out0.hidden_state                                              # (B, N, D) - KEEP ON GPU

        # Legal-masked softmax for root policy
        policy_probs = self._softmax_legal(policy0, legal)  # (B, N, A)

        # Dirichlet noise for root exploration
        noise_alpha = cfg.root_dirichlet_alpha
        noise_eps = cfg.root_exploration_fraction
        noises = rng.dirichlet(
            np.ones(cfg.action_space_size) * noise_alpha,
            size=(B, cfg.num_agents),
        ).astype(np.float32)  # (B, N, A)
        # Apply legal mask to noise too
        noises *= legal.astype(np.float32)
        noises += legal.astype(np.float32) * 1e-4
        noises /= noises.sum(axis=-1, keepdims=True)

        # beta = exploration-augmented sampling distribution
        beta = (1.0 - noise_eps) * policy_probs + noise_eps * noises  # (B, N, A)
        beta *= legal.astype(np.float32)
        beta /= beta.sum(axis=-1, keepdims=True)
        beta = beta.astype(np.float32)

        # --- Build ctree ---
        seed = int(rng.integers(0, 2**32 - 1))
        trees = Tree_batch(
            B, cfg.num_agents, cfg.action_space_size, K,
            cfg.num_simulations, cfg.tree_value_stat_delta_lb,
            seed, cfg.mcts_rho, cfg.mcts_lambda,
        )
        rewards0 = np.zeros(B, dtype=np.float32)
        trees.prepare(rewards0, values0, policy_probs, beta, K, noise_eps, noises)

        # --- Simulation loop ---
        # Pre-allocate buffer on GPU for all simulation steps
        # Shape: (num_simulations + 1, B, N, D)
        D = hidden0.shape[-1]
        pool_jnp = jnp.zeros((cfg.num_simulations + 1, B, cfg.num_agents, D), dtype=jnp.float32)
        pool_jnp = pool_jnp.at[0].set(hidden0)

        for sim in range(cfg.num_simulations):
            ix_lst, _, batch_actions = trees.batch_selection(
                cfg.pb_c_base, cfg.pb_c_init, cfg.discount
            )
            # ix_lst: list of length B, values 0..sim
            ix_jnp = jnp.array(ix_lst)

            rec_out = self._jit_recurrent(
                params, pool_jnp, ix_jnp, jnp.array(batch_actions)
            )
            
            # These transfers are still needed for ctree (C++), but small (scalars/logits)
            rec_rewards = np.array(self._jit_rew(rec_out.reward_logits), dtype=np.float32)  # (B,)
            rec_values = np.array(self._jit_val(rec_out.value_logits), dtype=np.float32)    # (B,)
            rec_policy = np.array(rec_out.policy_logits)                                    # (B, N, A)
            
            # Keep hidden state on GPU
            rec_hidden = rec_out.hidden_state                                               # (B, N, D)

            rec_probs = self._softmax(rec_policy)  # no legal masking at non-root
            
            # Update pool on GPU
            pool_jnp = pool_jnp.at[sim + 1].set(rec_hidden)

            trees.batch_expansion_and_backup(
                sim + 1, cfg.discount, K,
                rec_rewards, rec_values, rec_probs, rec_probs,
            )

        # --- Collect results ---
        return SearchOutput(
            root_value=trees.get_roots_values(),
            sampled_actions=trees.get_roots_sampled_actions(),
            sampled_visit_counts=trees.get_roots_sampled_visit_count(),
            sampled_qvalues=trees.get_roots_sampled_qvalues(cfg.discount),
            sampled_imp_ratio=trees.get_roots_sampled_imp_ratio(),
        )
