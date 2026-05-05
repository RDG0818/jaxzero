import numpy as np


class GameHistory:
    """Stores one complete episode trajectory."""

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_space_size: int,
        stacked_observations: int = 4,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_space_size = action_space_size
        self.stacked_observations = stacked_observations

        self._obs_history: list[np.ndarray] = []  # each (N, obs_dim)
        self.actions: list[np.ndarray] = []        # each (N,)
        self.rewards: list[float] = []
        self.legal_actions: list[np.ndarray] = []  # each (N, A)
        self.root_values: list[float] = []
        self.pred_values: list[float] = []
        self.sampled_actions: list[np.ndarray] = []  # each (K, N)
        self.sampled_policies: list[np.ndarray] = [] # each (K,)
        self.sampled_qvalues: list[np.ndarray] = []  # each (K,)
        self.sampled_masks: list[np.ndarray] = []    # each (K,)

    def store_observation(self, obs: np.ndarray):
        self._obs_history.append(obs.copy())

    def store_action(self, action: np.ndarray):
        self.actions.append(action.copy())

    def store_reward(self, reward: float):
        self.rewards.append(float(reward))

    def store_legal_actions(self, legal: np.ndarray):
        self.legal_actions.append(legal.copy())

    def store_root_value(self, v: float):
        self.root_values.append(float(v))

    def store_pred_value(self, v: float):
        self.pred_values.append(float(v))

    def store_search_stats(
        self,
        sampled_actions: np.ndarray,
        visit_counts: np.ndarray,
        qvalues: np.ndarray,
        mask: np.ndarray,
    ):
        self.sampled_actions.append(sampled_actions.copy())
        self.sampled_policies.append(visit_counts.copy())
        self.sampled_qvalues.append(qvalues.copy())
        self.sampled_masks.append(mask.copy())

    def __len__(self) -> int:
        return len(self.rewards)

    def obs(self, t: int, stacked_obs: int) -> np.ndarray:
        """Return stacked obs window ending at step t.

        Pads with the earliest available obs when t < stacked_obs - 1.
        Returns shape (N, obs_dim * stacked_obs).
        """
        frames = []
        for i in range(stacked_obs - 1, -1, -1):
            idx = t - i
            if idx < 0:
                idx = 0
            frames.append(self._obs_history[idx])
        return np.concatenate(frames, axis=-1)

    def make_target(
        self,
        pos: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> tuple:
        """Build training targets for one position.

        Returns:
            obs_batch:     (unroll_steps+1, N, obs_dim*stacked_obs)
            actions_batch: (unroll_steps, N)
            rewards_batch: (unroll_steps,)
            values_batch:  (unroll_steps+1,)
            policies_batch:(unroll_steps+1, K)  — sampled visit counts
            qvals_batch:   (unroll_steps+1, K)
            masks_batch:   (unroll_steps+1, K)
        """
        T = len(self)
        S = self.stacked_observations
        K = len(self.sampled_actions[0])

        obs_batch = np.stack([self.obs(min(pos + k, T - 1), S) for k in range(unroll_steps + 1)])
        actions_batch = np.stack([
            self.actions[min(pos + k, T - 1)] for k in range(unroll_steps)
        ])
        rewards_batch = np.array([
            self.rewards[min(pos + k, T - 1)] for k in range(unroll_steps)
        ])

        # n-step value targets
        values_batch = []
        for k in range(unroll_steps + 1):
            t = pos + k
            value = 0.0
            for n in range(td_steps):
                if t + n < T:
                    value += (discount ** n) * self.rewards[t + n]
            bootstrap_t = t + td_steps
            if bootstrap_t < T:
                value += (discount ** td_steps) * self.root_values[bootstrap_t]
            values_batch.append(value)
        values_batch = np.array(values_batch)

        policies_batch = np.stack([self.sampled_policies[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
        qvals_batch = np.stack([self.sampled_qvalues[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
        masks_batch = np.stack([
            self.sampled_masks[pos + k] if pos + k < T else np.zeros(K, dtype=bool)
            for k in range(unroll_steps + 1)
        ])

        return obs_batch, actions_batch, rewards_batch, values_batch, policies_batch, qvals_batch, masks_batch
