import numpy as np
from typing import List

import ray

from config import ExperimentConfig


@ray.remote
class ReplayBufferActor:
    """Manages prioritized experience replay in a dedicated process."""

    def __init__(self, obs_size: int, action_size: int, config: ExperimentConfig):
        from utils.replay_buffer import ReplayBuffer

        self.buffer = ReplayBuffer(
            capacity=config.train.replay_buffer_size,
            observation_shape=(obs_size,),
            action_space_size=action_size,
            num_agents=config.train.num_agents,
            unroll_steps=config.train.unroll_steps,
            alpha=config.train.replay_buffer_alpha,
            beta_start=config.train.replay_buffer_beta_start,
            beta_frames=config.train.replay_buffer_beta_frames,
        )

        # Sidecar arrays for per-root-child Q-data (not stored in C++ buffer).
        # Indexed by ring-buffer position (add_counter % capacity).
        # Used by the learner for action-level AWPO (Q_k - V_root advantage).
        cap = config.train.replay_buffer_size
        K   = config.mcts.num_gumbel_samples
        N   = config.train.num_agents
        self._q_actions = np.zeros((cap, K, N), dtype=np.int32)
        self._q_values  = np.zeros((cap, K),    dtype=np.float32)
        self._q_visits  = np.zeros((cap, K),    dtype=np.float32)
        self._q_valid   = np.zeros(cap,          dtype=bool)
        self._capacity  = cap
        self._add_counter = 0

    def add(self, items: list, priorities: List[float]):
        for item, priority in zip(items, priorities):
            slot = self._add_counter % self._capacity
            self.buffer.add(item, priority)
            if item.root_child_q is not None:
                self._q_actions[slot] = item.root_child_actions
                self._q_values[slot]  = item.root_child_q
                self._q_visits[slot]  = item.root_child_visits
                self._q_valid[slot]   = True
            else:
                self._q_valid[slot] = False
            self._add_counter += 1

    def sample(self, batch_size: int):
        result = self.buffer.sample(batch_size)
        if result[0] is None:
            return None, None, None, None
        batch, weights, indices = result
        # Attach sidecar Q-data for the sampled indices.
        # Items that were added before the sidecar was active have q_valid=False.
        q_data = {
            "root_child_actions": self._q_actions[indices],  # (B, K, N)
            "root_child_q":       self._q_values[indices],   # (B, K)
            "root_child_visits":  self._q_visits[indices],   # (B, K)
            "q_valid":            self._q_valid[indices],    # (B,) bool
        }
        return batch, weights, indices, q_data

    def sample_for_reanalysis(self, batch_size: int):
        return self.buffer.sample_for_reanalysis(batch_size)

    def update_targets(self, indices, policy_targets, root_values):
        self.buffer.update_targets(indices, policy_targets, root_values)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.buffer.update_priorities(indices, priorities)

    def get_size(self) -> int:
        return len(self.buffer)

    def get_stats(self) -> dict:
        return self.buffer.get_stats()
