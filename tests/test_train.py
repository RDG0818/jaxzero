import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet
from jaxzero.train import make_update_fn, awpo_sharp_loss
from jaxzero.reanalyze import BatchData


B, N, A, U, K = 2, 3, 9, 3, 5
OBS_DIM = 80 * 4


def make_config():
    return MAZeroConfig(
        num_agents=N,
        obs_size=OBS_DIM,
        action_space_size=A,
        unroll_steps=U,
        td_steps=3,
        batch_size=B,
        num_simulations=5,
        sampled_action_times=K,
        hidden_state_size=128,
    )


def make_fake_batch(config):
    return BatchData(
        obs=np.random.randn(B, U + 1, N, OBS_DIM).astype(np.float32),
        actions=np.zeros((B, U, N), dtype=np.int32),
        target_rewards=np.random.randn(B, U).astype(np.float32),
        target_values=np.random.randn(B, U + 1).astype(np.float32),
        target_policies=np.ones((B, U + 1, K), dtype=np.float32) / K,
        target_qvalues=np.random.randn(B, U + 1, K).astype(np.float32),
        target_masks=np.ones((B, U + 1, K), dtype=np.float32),
        sampled_actions=np.zeros((B, U + 1, K, N), dtype=np.int32),
        weights=np.ones(B, dtype=np.float32),
        indices=np.arange(B),
    )


def test_update_fn_runs():
    config = make_config()
    net = MAMuZeroNet(config=config)
    obs_init = jnp.ones((1, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs_init)
    update_fn = make_update_fn(net, config)
    batch = make_fake_batch(config)
    loss, grads, aux, priorities = update_fn(params, batch)
    assert jnp.isfinite(loss)
    assert priorities.shape == (config.batch_size,)


def test_loss_decreases_on_repeated_batch():
    """Loss should decrease when repeatedly training on same batch."""
    config = make_config()
    net = MAMuZeroNet(config=config)
    obs_init = jnp.ones((1, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs_init)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    update_fn = make_update_fn(net, config)
    batch = make_fake_batch(config)

    losses = []
    for _ in range(20):
        loss, grads, aux, priorities = update_fn(params, batch)
        losses.append(float(loss))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_update_fn_returns_aux():
    """make_update_fn should return (loss, grads, aux) with per-component losses."""
    config = make_config()
    net = MAMuZeroNet(config=config)
    obs_init = jnp.ones((1, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs_init)
    update_fn = make_update_fn(net, config)
    batch = make_fake_batch(config)

    result = update_fn(params, batch)
    assert len(result) == 4, "update_fn should return (loss, grads, aux, priorities)"
    loss, grads, aux, priorities = result
    assert jnp.isfinite(loss)
    assert priorities.shape == (config.batch_size,)
    assert "reward_loss" in aux
    assert "value_loss" in aux
    assert "policy_loss" in aux
    assert "consistency_loss" in aux
    assert all(jnp.isfinite(v) for v in aux.values())


def test_awpo_loss_shape():
    policy_logits = jnp.zeros((B, N, A))
    sampled_actions = jnp.zeros((B, K, N), dtype=jnp.int32)
    visit_counts = jnp.ones((B, K)) / K
    advantages = jnp.zeros((B, K))
    masks = jnp.ones((B, K))
    loss = awpo_sharp_loss(policy_logits, sampled_actions, visit_counts, advantages, masks, alpha=3.0)
    assert loss.shape == (B,)


def test_awpo_loss_std_normalization():
    """Advantages with larger variance should produce larger adv_weight spread."""
    B, N, A, K = 1, 2, 5, 4
    policy_logits = jnp.zeros((B, N, A))
    sampled_actions = jnp.zeros((B, K, N), dtype=jnp.int32)
    visit_counts = jnp.ones((B, K)) / K
    masks = jnp.ones((B, K))

    # Small variance advantages — same relative structure as large
    small_adv = jnp.array([[0.1, 0.2, 0.1, 0.2]])   # std ≈ 0.05
    # Large variance advantages — same relative structure, 10x scale
    large_adv = jnp.array([[1.0, 2.0, 1.0, 2.0]])   # std ≈ 0.5

    loss_small = awpo_sharp_loss(policy_logits, sampled_actions, visit_counts, small_adv, masks, alpha=3.0)
    loss_large = awpo_sharp_loss(policy_logits, sampled_actions, visit_counts, large_adv, masks, alpha=3.0)

    # With std normalization, both should produce similar loss magnitude
    # (same relative structure → same normalized advantages → same loss)
    assert jnp.abs(loss_small - loss_large).mean() < 0.01, (
        f"Std normalization should make loss scale-invariant: small={loss_small}, large={loss_large}"
    )
