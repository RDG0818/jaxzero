import numpy as np
import jax
import pytest
from jaxzero.envs.smax_wrapper import SMAXWrapper
from jaxzero.envs.mpe_wrapper import MPEWrapper


def test_smax_reset_shapes():
    env = SMAXWrapper(map_name="3m", stacked_observations=4)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    assert obs.shape == (env.num_agents, env.obs_size)


def test_smax_step_shapes():
    env = SMAXWrapper(map_name="3m", stacked_observations=4)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    actions = np.zeros(env.num_agents, dtype=np.int32)
    rng2 = jax.random.PRNGKey(1)
    obs2, state2, reward, done, won = env.step(rng2, state, actions)
    assert obs2.shape == (env.num_agents, env.obs_size)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(won, bool)


def test_smax_legal_actions_shape():
    env = SMAXWrapper(map_name="3m", stacked_observations=4)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    legal = env.get_legal_actions(state)
    assert legal.shape == (env.num_agents, env.action_space_size)
    assert legal.dtype == bool


def test_smax_obs_stacking():
    """After reset, all 4 stacked frames should be identical (first obs repeated)."""
    env = SMAXWrapper(map_name="3m", stacked_observations=4)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    raw_size = env.obs_size // env.stacked_observations
    # First frame and last frame should be identical
    assert np.allclose(obs[:, :raw_size], obs[:, -raw_size:])


def test_mpe_reset_shapes():
    env = MPEWrapper()
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    assert obs.shape[0] == env.num_agents
    assert obs.shape[1] == env.obs_size


def test_mpe_legal_actions_all_valid():
    """MPE has uniform legal actions."""
    env = MPEWrapper()
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    legal = env.get_legal_actions(state)
    assert legal.all()
