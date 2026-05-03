import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxzero.config import MAZeroConfig
from jaxzero.model.transforms import phi_inv as _phi_inv


class SearchOutput(NamedTuple):
    root_value: np.ndarray        # (B,)
    sampled_actions: list         # list[B] of (K, N)
    sampled_visit_counts: list    # list[B] of (K,)
    sampled_qvalues: list         # list[B] of (K,)
    sampled_imp_ratio: list       # list[B] of (K,)


class Node:
    __slots__ = [
        'visit_count', 'q_value', 'reward', 'value_sum',
        'prior', 'beta', 'hidden', 'children',
        'sampled_actions', 'expanded',
    ]

    def __init__(self):
        self.visit_count: int = 0
        self.q_value: np.ndarray | None = None
        self.reward: float = 0.0
        self.value_sum: float = 0.0
        self.prior: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.hidden: np.ndarray | None = None
        self.children: dict[int, 'Node'] = {}
        self.sampled_actions: np.ndarray | None = None
        self.expanded: bool = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class SampledMCTS:
    """Sampled MCTS with batched inference across environments.

    Each simulation step issues one model call of shape (B, ...) covering all
    B environments simultaneously, instead of B sequential calls of shape (1, ...).
    This keeps the GPU batch dimension large and minimizes Python round-trips.
    """

    def __init__(self, config: MAZeroConfig, model):
        self.config = config
        self.model = model

        self._jit_initial = jax.jit(model.apply)
        self._jit_recurrent = jax.jit(
            lambda p, h, a: model.apply(p, h, a, method=model.recurrent_inference)
        )

        value_support = config.value_support_size
        reward_support = config.reward_support_size
        self._jit_val = jax.jit(lambda logits: _phi_inv(logits, value_support))
        self._jit_rew = jax.jit(lambda logits: _phi_inv(logits, reward_support))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_actions(
        self,
        policy_logits: np.ndarray,
        legal_mask: np.ndarray,
        K: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample K joint actions from the policy, respecting legal_mask.

        Returns:
            sampled_actions: (K, N)
            beta:            (K,) joint prob under policy
            prior:           (K,) copy of beta before Dirichlet
        """
        N, A = policy_logits.shape
        log_probs = policy_logits - policy_logits.max(axis=-1, keepdims=True)
        probs = np.exp(log_probs)
        probs = probs * legal_mask.astype(np.float32)
        probs += legal_mask.astype(np.float32) * 1e-8
        probs /= probs.sum(axis=-1, keepdims=True)

        actions = np.zeros((K, N), dtype=np.int32)
        for n in range(N):
            actions[:, n] = rng.choice(A, size=K, p=probs[n])

        joint_prob = np.ones(K, dtype=np.float64)
        for n in range(N):
            joint_prob *= probs[n, actions[:, n]]
        joint_prob = joint_prob.astype(np.float32)

        return actions, joint_prob.copy(), joint_prob.copy()

    def _add_dirichlet(self, beta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.dirichlet(np.ones(len(beta)) * self.config.root_dirichlet_alpha)
        eps = self.config.root_exploration_fraction
        return (1 - eps) * beta + eps * noise

    def _ucb_score(self, parent: Node, child_idx: int, parent_visits: int) -> float:
        cfg = self.config
        c = cfg.pb_c_init + math.log(
            (parent_visits + cfg.pb_c_base + 1) / cfg.pb_c_base
        )
        child = parent.children[child_idx]
        n_child = child.visit_count
        prior = float(parent.prior[child_idx])
        beta = float(parent.beta[child_idx])
        q = float(parent.q_value[child_idx]) if parent.q_value is not None else 0.0
        explore = (
            prior / (beta + 1e-8)
            * math.sqrt(parent_visits + 1)
            / (1 + n_child)
            * c
        )
        return q + explore

    def _select(self, root: Node) -> list[tuple[Node, int]]:
        path: list[tuple[Node, int]] = []
        node = root
        while node.expanded:
            K = len(node.sampled_actions)
            parent_visits = sum(c.visit_count for c in node.children.values())
            unvisited = [i for i in range(K) if node.children[i].visit_count == 0]
            if unvisited:
                best_idx = unvisited[0]
            else:
                best_idx = max(
                    range(K),
                    key=lambda i: self._ucb_score(node, i, parent_visits),
                )
            path.append((node, best_idx))
            child = node.children[best_idx]
            if not child.expanded:
                break
            node = child
        return path

    def _backup(self, path: list[tuple[Node, int]], leaf_value: float):
        value = leaf_value
        for node, action_idx in reversed(path):
            child = node.children[action_idx]
            value = child.reward + self.config.discount * value
            child.visit_count += 1
            n = child.visit_count
            if node.q_value is None:
                node.q_value = np.zeros(len(node.sampled_actions), dtype=np.float32)
            old_q = node.q_value[action_idx]
            node.q_value[action_idx] = old_q + (value - old_q) / n

    # ------------------------------------------------------------------
    # Public API — batched search
    # ------------------------------------------------------------------

    def search(
        self,
        params,
        obs: np.ndarray,    # (B, N, obs_dim)
        legal: np.ndarray,  # (B, N, A)
        rng: np.random.Generator,
    ) -> SearchOutput:
        """Run Sampled MCTS for a batch of B environments.

        All model calls have shape (B, ...) — one call per simulation step
        across all B environments simultaneously.
        """
        B = obs.shape[0]
        K = self.config.sampled_action_times
        cfg = self.config

        # --- Initial inference: one B-batch call ---
        out0 = self._jit_initial(params, jnp.array(obs))
        all_values0 = np.array(self._jit_val(out0.value_logits))  # (B,)
        all_policy0 = np.array(out0.policy_logits)                # (B, N, A)
        all_hidden0 = np.array(out0.hidden_state)                 # (B, N, D)

        # --- Build B root nodes ---
        roots = []
        for b in range(B):
            root = Node()
            root.hidden = all_hidden0[b]
            root.value_sum = float(all_values0[b])
            root.visit_count = 1
            actions, beta, prior = self._sample_actions(all_policy0[b], legal[b], K, rng)
            beta = self._add_dirichlet(beta, rng)
            root.sampled_actions = actions
            root.beta = beta
            root.prior = prior
            root.q_value = np.zeros(K, dtype=np.float32)
            root.expanded = True
            root.children = {k: Node() for k in range(K)}
            roots.append(root)

        # Preallocate inference buffers — reused every simulation step
        N_agents = obs.shape[1]
        D = all_hidden0.shape[-1]
        hidden_buf = np.empty((B, N_agents, D), dtype=np.float32)
        action_buf = np.empty((B, N_agents), dtype=np.int32)

        # --- Simulations: B-batched model call per step ---
        for _ in range(cfg.num_simulations):
            paths = [self._select(roots[b]) for b in range(B)]

            active = []
            for b in range(B):
                if paths[b]:
                    parent, action_idx = paths[b][-1]
                    hidden_buf[b] = parent.hidden
                    action_buf[b] = parent.sampled_actions[action_idx]
                    active.append(b)

            if not active:
                break

            # One batched recurrent call covering all B envs
            rec_out = self._jit_recurrent(
                params, jnp.array(hidden_buf), jnp.array(action_buf)
            )
            rec_values = np.array(self._jit_val(rec_out.value_logits))   # (B,)
            rec_rewards = np.array(self._jit_rew(rec_out.reward_logits)) # (B,)
            rec_policy = np.array(rec_out.policy_logits)                 # (B, N, A)
            rec_hidden = np.array(rec_out.hidden_state)                  # (B, N, D)

            for b in active:
                parent, action_idx = paths[b][-1]
                child = parent.children[action_idx]
                child.hidden = rec_hidden[b]
                child.reward = float(rec_rewards[b])
                child.value_sum = float(rec_values[b])

                child_legal = np.ones((cfg.num_agents, cfg.action_space_size), dtype=bool)
                c_actions, c_beta, c_prior = self._sample_actions(
                    rec_policy[b], child_legal, K, rng
                )
                child.sampled_actions = c_actions
                child.beta = c_beta
                child.prior = c_prior
                child.q_value = np.zeros(K, dtype=np.float32)
                child.expanded = True
                child.children = {kk: Node() for kk in range(K)}

                self._backup(paths[b], child.value_sum)

        # --- Collect results ---
        all_actions, all_visits, all_qvalues, all_ratios = [], [], [], []
        for b in range(B):
            root = roots[b]
            visits = np.array(
                [root.children[k].visit_count for k in range(K)], dtype=np.int32
            )
            all_actions.append(root.sampled_actions)
            all_visits.append(visits)
            all_qvalues.append(root.q_value.copy())
            all_ratios.append(root.prior / (root.beta + 1e-8))

        return SearchOutput(
            root_value=all_values0,
            sampled_actions=all_actions,
            sampled_visit_counts=all_visits,
            sampled_qvalues=all_qvalues,
            sampled_imp_ratio=all_ratios,
        )
