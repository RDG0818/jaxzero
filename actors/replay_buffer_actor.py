import numpy as np
from typing import List

import ray

from config import CONFIG


@ray.remote
class ReplayBufferActor:
    """Manages prioritized experience replay in a dedicated process."""

    def __init__(self, obs_size: int, action_size: int):
        from utils.replay_buffer import ReplayBuffer

        self.buffer = ReplayBuffer(
            capacity=CONFIG.train.replay_buffer_size,
            observation_shape=(obs_size,),
            action_space_size=action_size,
            num_agents=CONFIG.train.num_agents,
            unroll_steps=CONFIG.train.unroll_steps,
            alpha=CONFIG.train.replay_buffer_alpha,
            beta_start=CONFIG.train.replay_buffer_beta_start,
            beta_frames=CONFIG.train.replay_buffer_beta_frames,
        )

    def add(self, items: list, priorities: List[float]):
        for item, priority in zip(items, priorities):
            self.buffer.add(item, priority)

    def sample(self, batch_size: int):
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.buffer.update_priorities(indices, priorities)

    def get_size(self) -> int:
        return len(self.buffer)
