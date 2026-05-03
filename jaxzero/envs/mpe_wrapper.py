import numpy as np
import jax
from typing import Any
from jaxmarl import make
from jaxzero.envs.base import EnvWrapper


class MPEWrapper(EnvWrapper):
    """MPE SimpleSpreads wrapper for sanity checks. Uniform legal actions."""

    def __init__(self, num_agents: int = 3, stacked_observations: int = 1):
        self.stacked_observations = stacked_observations
        self._env = make("MPE_simple_spread_v3", num_agents=num_agents)
        self.num_agents = num_agents
        self._agents = self._env.agents
        sample_obs = self._env.observation_space(self._agents[0]).shape[0]
        self._raw_obs_size = sample_obs
        self.obs_size = sample_obs * stacked_observations
        self.action_space_size = self._env.action_space(self._agents[0]).n
        self._obs_stack = None

    def reset(self, rng_key) -> tuple[np.ndarray, Any]:
        """Reset the environment and return (obs, state).

        obs shape: (num_agents, obs_size).
        """
        obs_dict, state = self._env.reset(rng_key)
        raw_obs = np.stack([np.array(obs_dict[a]) for a in self._agents])
        self._obs_stack = np.tile(raw_obs[:, np.newaxis, :], (1, self.stacked_observations, 1))
        return self._obs_stack.reshape(self.num_agents, -1), state

    def step(self, rng_key, state, actions: np.ndarray) -> tuple[np.ndarray, Any, float, bool, bool]:
        """Step the environment. Returns (obs, state, reward, done, won=False)."""
        actions_dict = {a: int(actions[i]) for i, a in enumerate(self._agents)}
        obs_dict, state, reward_dict, done_dict, info = self._env.step(rng_key, state, actions_dict)
        raw_obs = np.stack([np.array(obs_dict[a]) for a in self._agents])
        self._obs_stack = np.roll(self._obs_stack, shift=-1, axis=1)
        self._obs_stack[:, -1, :] = raw_obs
        stacked = self._obs_stack.reshape(self.num_agents, -1)
        reward = float(np.mean([reward_dict[a] for a in self._agents]))
        done = bool(done_dict["__all__"])
        return stacked, state, reward, done, False

    def get_legal_actions(self, state) -> np.ndarray:
        """All actions are legal in MPE. Returns all-True mask (num_agents, action_space_size)."""
        return np.ones((self.num_agents, self.action_space_size), dtype=bool)
