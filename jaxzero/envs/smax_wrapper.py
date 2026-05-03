import numpy as np
import jax
import jax.numpy as jnp
from typing import Any
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxzero.envs.base import EnvWrapper


class SMAXWrapper(EnvWrapper):
    """SMAX (StarCraft Multi-Agent Challenge) wrapper with observation stacking.

    Uses HeuristicEnemySMAX so the enemy team is controlled by a built-in
    heuristic, making it safe to use from a single-process training loop.
    """

    def __init__(self, map_name: str = "3m", stacked_observations: int = 4):
        self.stacked_observations = stacked_observations
        scenario = map_name_to_scenario(map_name)
        self._env = make("HeuristicEnemySMAX", scenario=scenario)
        self.num_agents = scenario.num_allies
        self._agents = [f"ally_{i}" for i in range(self.num_agents)]
        sample_obs = self._env.observation_space(self._agents[0]).shape[0]
        self._raw_obs_size = sample_obs
        self.obs_size = sample_obs * stacked_observations
        self.action_space_size = self._env.action_space(self._agents[0]).n
        self._obs_stack = None

    def reset(self, rng_key) -> tuple[np.ndarray, Any]:
        """Reset the environment and return (obs, state).

        obs shape: (num_agents, obs_size)  — first raw obs tiled across all frames.
        """
        obs_dict, state = self._env.reset(rng_key)
        raw_obs = np.stack([np.array(obs_dict[a]) for a in self._agents])  # (N, raw_obs)
        self._obs_stack = np.tile(raw_obs[:, np.newaxis, :], (1, self.stacked_observations, 1))
        stacked = self._obs_stack.reshape(self.num_agents, -1)
        return stacked, state

    def step(self, rng_key, state, actions: np.ndarray) -> tuple[np.ndarray, Any, float, bool, bool]:
        """Step the environment.

        Returns (obs, state, reward, done, won) where reward and done are
        shared across all allies (cooperative task).
        """
        actions_dict = {a: int(actions[i]) for i, a in enumerate(self._agents)}
        obs_dict, state, reward_dict, done_dict, info = self._env.step(rng_key, state, actions_dict)

        raw_obs = np.stack([np.array(obs_dict[a]) for a in self._agents])
        self._obs_stack = np.roll(self._obs_stack, shift=-1, axis=1)
        self._obs_stack[:, -1, :] = raw_obs
        stacked = self._obs_stack.reshape(self.num_agents, -1)

        reward = float(reward_dict[self._agents[0]])
        done = bool(done_dict["__all__"])
        won = done and reward > 0.5
        return stacked, state, reward, done, won

    def get_legal_actions(self, state) -> np.ndarray:
        """Return boolean legal-action mask of shape (num_agents, action_space_size)."""
        avail = self._env.get_avail_actions(state)
        return np.stack([np.array(avail[a], dtype=bool) for a in self._agents])
