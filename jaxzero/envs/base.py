from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class EnvWrapper(ABC):
    obs_size: int
    action_space_size: int
    num_agents: int
    stacked_observations: int

    @abstractmethod
    def reset(self, rng_key) -> tuple[np.ndarray, Any]:
        """Returns (obs: (N, obs_size), state)."""

    @abstractmethod
    def step(self, rng_key, state, actions: np.ndarray) -> tuple[np.ndarray, Any, float, bool, bool]:
        """Returns (obs, state, reward, done, won)."""

    @abstractmethod
    def get_legal_actions(self, state) -> np.ndarray:
        """Returns (N, A) bool mask."""
