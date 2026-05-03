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
    """Sampled MCTS for multi-agent MuZero with OS(λ) backup.

    Each simulation:
      1. Select a path from root to a leaf using UCB on sampled joint actions.
      2. Expand the leaf by running recurrent_inference.
      3. Backup the value along the path using discounted returns.
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

        Args:
            policy_logits: (N, A) raw logits from the network.
            legal_mask:    (N, A) boolean mask of legal actions.
            K:             number of joint action samples.
            rng:           numpy RNG.

        Returns:
            sampled_actions: (K, N) int32 joint actions.
            beta:            (K,)  probability of each joint action under the policy.
            prior:           (K,)  same as beta (copy; root uses beta after Dirichlet).
        """
        N, A = policy_logits.shape

        # Compute per-agent softmax probabilities with legal masking.
        log_probs = policy_logits - policy_logits.max(axis=-1, keepdims=True)
        probs = np.exp(log_probs)
        probs = probs * legal_mask.astype(np.float32)
        # Small epsilon to avoid zero prob for legal actions (numerical safety).
        probs += legal_mask.astype(np.float32) * 1e-8
        probs /= probs.sum(axis=-1, keepdims=True)

        # Sample K actions per agent independently, then combine into joint actions.
        actions = np.zeros((K, N), dtype=np.int32)
        for n in range(N):
            actions[:, n] = rng.choice(A, size=K, p=probs[n])

        # Joint action probability = product of per-agent marginals.
        joint_prob = np.ones(K, dtype=np.float64)
        for n in range(N):
            joint_prob *= probs[n, actions[:, n]]
        joint_prob = joint_prob.astype(np.float32)

        return actions, joint_prob.copy(), joint_prob.copy()

    def _add_dirichlet(self, beta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mix beta with Dirichlet noise for root exploration."""
        K = len(beta)
        noise = rng.dirichlet(np.ones(K) * self.config.root_dirichlet_alpha)
        eps = self.config.root_exploration_fraction
        return (1 - eps) * beta + eps * noise

    def _ucb_score(self, parent: Node, child_idx: int, parent_visits: int) -> float:
        """Compute UCB score for a child node."""
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
        """Traverse from root to a leaf to expand.

        Returns the path as list of (node, action_idx) pairs, where the last
        pair points to the unexpanded child that should be expanded.
        """
        path: list[tuple[Node, int]] = []
        node = root

        while node.expanded:
            K = len(node.sampled_actions)
            parent_visits = sum(c.visit_count for c in node.children.values())

            # Prefer unvisited children (score = inf); otherwise use UCB.
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
                # This is the leaf we will expand.
                break
            node = child

        return path

    def _backup(self, path: list[tuple[Node, int]], leaf_value: float):
        """Backup the leaf value along the path, updating Q-values."""
        value = leaf_value
        for node, action_idx in reversed(path):
            child = node.children[action_idx]
            r = child.reward
            value = r + self.config.discount * value
            child.visit_count += 1
            n = child.visit_count
            if node.q_value is None:
                node.q_value = np.zeros(len(node.sampled_actions), dtype=np.float32)
            old_q = node.q_value[action_idx]
            node.q_value[action_idx] = old_q + (value - old_q) / n

    # ------------------------------------------------------------------
    # Single-item search
    # ------------------------------------------------------------------

    def _search_single(
        self,
        params,
        obs: np.ndarray,
        legal: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[Node, float]:
        """Run MCTS for one batch item.

        Args:
            params: model parameters.
            obs:    (N, obs_dim) observation for one environment.
            legal:  (N, A) legal action mask for one environment.
            rng:    numpy RNG.

        Returns:
            root node, root value scalar.
        """
        cfg = self.config
        K = cfg.sampled_action_times

        # --- Initial inference ---
        obs_b = jnp.array(obs[np.newaxis])  # (1, N, obs_dim)
        out = self._jit_initial(params, obs_b)
        policy_logits = np.array(out.policy_logits[0])  # (N, A)
        value_scalar = float(self._jit_val(out.value_logits)[0])
        hidden = np.array(out.hidden_state[0])  # (N, D)

        # --- Build root node ---
        root = Node()
        root.hidden = hidden
        root.value_sum = value_scalar
        root.visit_count = 1

        sampled_actions, beta, prior = self._sample_actions(policy_logits, legal, K, rng)
        beta = self._add_dirichlet(beta, rng)
        root.sampled_actions = sampled_actions   # (K, N)
        root.beta = beta                         # (K,)
        root.prior = prior                       # (K,)
        root.q_value = np.zeros(K, dtype=np.float32)
        root.expanded = True
        root.children = {k: Node() for k in range(K)}

        # --- Simulations ---
        for _ in range(cfg.num_simulations):
            path = self._select(root)
            if not path:
                break

            parent, action_idx = path[-1]
            joint_action = parent.sampled_actions[action_idx]  # (N,)

            # Recurrent inference.
            hidden_b = jnp.array(parent.hidden[np.newaxis])       # (1, N, D)
            action_b = jnp.array(joint_action[np.newaxis], dtype=jnp.int32)  # (1, N)
            rec_out = self._jit_recurrent(params, hidden_b, action_b)

            # Populate the leaf child.
            child = parent.children[action_idx]
            child.hidden = np.array(rec_out.hidden_state[0])  # (N, D)
            child.reward = float(self._jit_rew(rec_out.reward_logits)[0])
            child.value_sum = float(self._jit_val(rec_out.value_logits)[0])
            # Note: child.visit_count is incremented in _backup, starts at 0.

            # Expand the child with sampled actions (all actions legal internally).
            child_policy = np.array(rec_out.policy_logits[0])  # (N, A)
            child_legal = np.ones((cfg.num_agents, cfg.action_space_size), dtype=bool)
            child_actions, child_beta, child_prior = self._sample_actions(
                child_policy, child_legal, K, rng
            )
            child.sampled_actions = child_actions
            child.beta = child_beta
            child.prior = child_prior
            child.q_value = np.zeros(K, dtype=np.float32)
            child.expanded = True
            child.children = {kk: Node() for kk in range(K)}

            # Backup leaf value through the path.
            self._backup(path, child.value_sum)

        return root, value_scalar

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        params,
        obs: np.ndarray,
        legal: np.ndarray,
        rng: np.random.Generator,
    ) -> SearchOutput:
        """Run Sampled MCTS for a batch of environments.

        Args:
            params: model parameters (pytree).
            obs:    (B, N, obs_dim) batch of observations.
            legal:  (B, N, A) batch of legal action masks.
            rng:    numpy RNG (used for action sampling and Dirichlet noise).

        Returns:
            SearchOutput with per-batch-item results.
        """
        B = obs.shape[0]
        root_values = np.zeros(B, dtype=np.float32)
        all_actions: list = []
        all_visits: list = []
        all_qvalues: list = []
        all_ratios: list = []

        for b in range(B):
            root, v = self._search_single(params, obs[b], legal[b], rng)
            root_values[b] = v
            K = len(root.sampled_actions)
            visits = np.array(
                [root.children[k].visit_count for k in range(K)], dtype=np.int32
            )
            all_actions.append(root.sampled_actions)          # (K, N)
            all_visits.append(visits)                         # (K,)
            all_qvalues.append(root.q_value.copy())           # (K,)
            all_ratios.append(root.prior / (root.beta + 1e-8))  # (K,)

        return SearchOutput(
            root_value=root_values,
            sampled_actions=all_actions,
            sampled_visit_counts=all_visits,
            sampled_qvalues=all_qvalues,
            sampled_imp_ratio=all_ratios,
        )
