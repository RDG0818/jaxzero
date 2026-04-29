import jax
import jax.numpy as jnp
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet


B, N, D, A = 2, 3, 128, 9
OBS_DIM = 80 * 4  # stacked obs


def make_net(config=None):
    if config is None:
        config = MAZeroConfig(
            num_agents=N,
            obs_size=OBS_DIM,
            action_space_size=A,
        )
    return MAMuZeroNet(config=config), config


def test_initial_inference_shapes():
    net, config = make_net()
    obs = jax.random.normal(jax.random.PRNGKey(42), (B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    out = net.apply(params, obs)
    assert out.hidden_state.shape == (B, N, D)
    assert out.value_logits.shape == (B, 11)
    assert out.policy_logits.shape == (B, N, A)
    assert out.reward_logits.shape == (B, 11)


def test_recurrent_inference_shapes():
    net, config = make_net()
    obs = jax.random.normal(jax.random.PRNGKey(42), (B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    hidden = net.apply(params, obs).hidden_state
    actions = jnp.zeros((B, N), dtype=jnp.int32)
    out = net.apply(params, hidden, actions, method=net.recurrent_inference)
    assert out.hidden_state.shape == (B, N, D)
    assert out.reward_logits.shape == (B, 11)
    assert out.value_logits.shape == (B, 11)
    assert out.policy_logits.shape == (B, N, A)


def test_dynamics_residual():
    """next_hidden != input hidden (residual adds, not replaces)."""
    net, config = make_net()
    obs = jax.random.normal(jax.random.PRNGKey(42), (B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    hidden = net.apply(params, obs).hidden_state
    actions = jnp.zeros((B, N), dtype=jnp.int32)
    out = net.apply(params, hidden, actions, method=net.recurrent_inference)
    assert not jnp.allclose(out.hidden_state, hidden)


def test_project_online_shape():
    net, config = make_net()
    obs = jax.random.normal(jax.random.PRNGKey(42), (B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    hidden = net.apply(params, obs).hidden_state
    proj = net.apply(params, hidden, method=net.project_online)
    assert proj.shape == (B, 128)


def test_project_target_shape():
    net, config = make_net()
    obs = jax.random.normal(jax.random.PRNGKey(42), (B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    hidden = net.apply(params, obs).hidden_state
    proj = net.apply(params, hidden, method=net.project_target)
    assert proj.shape == (B, 128)


def test_all_params_initialized():
    """init() via __call__ must include dynamics and projection params."""
    net, config = make_net()
    obs = jax.random.normal(jax.random.PRNGKey(42), (B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    param_keys = set(params['params'].keys())
    assert 'representation_net' in param_keys
    assert 'dynamics_net' in param_keys
    assert 'prediction_net' in param_keys
    assert 'projection_net' in param_keys
