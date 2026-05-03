import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet
from jaxzero.mcts.sampled_mcts import SampledMCTS, SearchOutput


B, N, A = 2, 3, 9
OBS_DIM = 80 * 4


def make_config():
    return MAZeroConfig(
        num_agents=N,
        obs_size=OBS_DIM,
        action_space_size=A,
        num_simulations=10,
        sampled_action_times=5,
    )


def make_net_and_params(config):
    net = MAMuZeroNet(config=config)
    obs = jnp.ones((B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    return net, params


def test_search_output_shapes():
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.ones((B, N, OBS_DIM), dtype=np.float32)
    legal = np.ones((B, N, A), dtype=bool)
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    assert isinstance(result, SearchOutput)
    assert result.root_value.shape == (B,)
    assert len(result.sampled_actions) == B
    assert len(result.sampled_visit_counts) == B


def test_legal_masking_respected():
    """If only action 0 is legal for all agents, all sampled actions should be 0."""
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.ones((B, N, OBS_DIM), dtype=np.float32)
    legal = np.zeros((B, N, A), dtype=bool)
    legal[:, :, 0] = True  # only action 0 legal
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    for b in range(B):
        assert (result.sampled_actions[b] == 0).all()


def test_visit_counts_sum_to_simulations():
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.ones((B, N, OBS_DIM), dtype=np.float32)
    legal = np.ones((B, N, A), dtype=bool)
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    for b in range(B):
        assert result.sampled_visit_counts[b].sum() == config.num_simulations


def test_batch_independence():
    """Both batch items should get independent root values (not identical)."""
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.random.randn(B, N, OBS_DIM).astype(np.float32)
    legal = np.ones((B, N, A), dtype=bool)
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    assert not np.allclose(result.root_value[0], result.root_value[1])
