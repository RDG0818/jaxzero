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

    def add(self, items: list, priorities: List[float]):
        for item, priority in zip(items, priorities):
            self.buffer.add(item, priority)

    def sample(self, batch_size: int):
        return self.buffer.sample(batch_size)

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
