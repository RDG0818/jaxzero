# utils/mpe_env_wrapper.py
import jax
import jaxmarl
import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, List, Tuple


class MPEEnvWrapper:
    """
    A stateless wrapper for JaxMARL MPE environments.

    Converts JaxMARL's dict-based outputs to stacked NumPy arrays.
    RNG keys are passed in by the caller — this class holds no random state.
    """

    def __init__(self, env_name: str, num_agents: int, max_steps: int):
        self.env = jaxmarl.make(env_name, num_agents=num_agents)
        self.num_agents: int = num_agents
        self.agents: List[str] = self.env.agents
        self.observation_size: int = self.env.observation_space(self.agents[0]).shape[0]
        self.observation_shape: Tuple[int, ...] = (self.observation_size,)
        self.action_space_size: int = self.env.action_space(self.agents[0]).n

    def reset(self, rng_key: jax.Array) -> Tuple[np.ndarray, Any]:
        """
        Resets the environment.

        Args:
            rng_key: JAX PRNG key for the reset.

        Returns:
            observations: Shape (1, num_agents, observation_size)
            state: Initial environment state.
        """
        obs_dict, state = self.env.reset(rng_key)
        return self._stack_dict(obs_dict), state

    def step(
        self, rng_key: jax.Array, state: Any, actions: np.ndarray
    ) -> Tuple[np.ndarray, Any, float, bool]:
        """
        Steps the environment.

        Args:
            rng_key: JAX PRNG key for the step.
            state: Current environment state.
            actions: Joint actions for all agents. Shape: (num_agents,)

        Returns:
            next_observations: Shape (1, num_agents, observation_size)
            next_state: Subsequent environment state.
            team_reward: Summed reward across all agents.
            episode_done: True if all agents are done.
        """
        action_dict = {agent: int(action) for agent, action in zip(self.agents, actions)}
        next_obs, next_state, reward, done, _ = self.env.step(rng_key, state, action_dict)

        next_obs = {k: np.array(v) for k, v in next_obs.items()}
        reward = {k: np.array(v) for k, v in reward.items()}
        done = {k: np.array(v) for k, v in done.items()}
        return self._stack_dict(next_obs), next_state, float(sum(reward.values())), bool(all(done.values()))

    def _stack_dict(self, data_dict: Dict[str, jnp.ndarray]) -> np.ndarray:
        """Stacks per-agent data into shape (1, num_agents, *data_shape)."""
        data_list = [np.asarray(data_dict[agent], dtype=np.float32) for agent in self.agents]
        return np.stack(data_list, axis=0)[np.newaxis, ...]
