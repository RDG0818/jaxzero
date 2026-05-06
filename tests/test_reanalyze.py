import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from typing import NamedTuple
from unittest.mock import patch, MagicMock
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet
from jaxzero.game import GameHistory
from jaxzero.replay_buffer import PrioritizedReplayBuffer
from jaxzero.reanalyze import ReanalyzeWorker, BatchData, _pad_to_k


class SearchOutput(NamedTuple):
    """Mirrors jaxzero.mcts.sampled_mcts.SearchOutput without importing cytree."""
    root_value: np.ndarray
    sampled_actions: list
    sampled_visit_counts: list
    sampled_qvalues: list
    sampled_imp_ratio: list = []


B, N, A, OBS_DIM = 4, 3, 9, 80 * 4
K = 10


def make_config(use_reanalyze=False):
    return MAZeroConfig(
        num_agents=N,
        obs_size=OBS_DIM,
        action_space_size=A,
        batch_size=B,
        unroll_steps=3,
        td_steps=3,
        use_reanalyze=use_reanalyze,
        num_simulations=5,
        sampled_action_times=K,
        min_replay_size=5,
    )


def make_game(T=20):
    g = GameHistory(num_agents=N, obs_dim=OBS_DIM // 4, action_space_size=A, stacked_observations=4)
    for t in range(T):
        g.store_observation(np.random.randn(N, OBS_DIM // 4).astype(np.float32))
        g.store_action(np.zeros(N, dtype=np.int32))
        g.store_reward(float(np.random.randn()))
        g.store_legal_actions(np.ones((N, A), dtype=bool))
        g.store_root_value(float(np.random.randn()))
        g.store_pred_value(float(np.random.randn()))
        g.store_search_stats(
            sampled_actions=np.zeros((K, N), dtype=np.int32),
            visit_counts=np.ones(K) / K,
            qvalues=np.random.randn(K).astype(np.float32),
            mask=np.ones(K, dtype=bool),
        )
    return g


def make_buffer_ctx(config):
    buf = PrioritizedReplayBuffer(config)
    for _ in range(10):
        buf.add(make_game())
    return buf.prepare_batch_context(B, beta=0.4)


def test_pad_to_k_pads_shorter():
    visits = np.array([10.0, 5.0, 3.0], dtype=np.float32)
    actions = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=np.int32)  # (3, N=3)
    qvals = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    K, N_agents = 5, 3

    pol, sa, qv, mask = _pad_to_k(visits, actions, qvals, K, N_agents)

    assert pol.shape == (K,)
    assert sa.shape == (K, N_agents)
    assert qv.shape == (K,)
    assert mask.shape == (K,)
    assert mask.dtype == bool
    # first 3 valid, last 2 padded
    assert mask[:3].all() and not mask[3:].any()
    # visit counts normalized to sum=1 over valid entries
    np.testing.assert_allclose(pol[:3].sum(), 1.0, atol=1e-5)
    # padded entries are zero
    assert pol[3:].sum() == 0.0
    assert sa[3:].sum() == 0


def test_pad_to_k_exact_k():
    visits = np.array([4.0, 3.0, 2.0, 1.0, 0.5], dtype=np.float32)
    actions = np.zeros((5, 2), dtype=np.int32)
    qvals = np.zeros(5, dtype=np.float32)
    K, N_agents = 5, 2

    pol, sa, qv, mask = _pad_to_k(visits, actions, qvals, K, N_agents)

    assert mask.all()
    np.testing.assert_allclose(pol.sum(), 1.0, atol=1e-5)


def test_pad_to_k_empty():
    """K_actual=0: all-zero pol, all-False mask."""
    K, N_agents = 5, 3
    pol, actions, qvals, mask = _pad_to_k(
        np.array([], dtype=np.float32),
        np.zeros((0, N_agents), dtype=np.int32),
        np.array([], dtype=np.float32),
        K, N_agents,
    )
    assert pol.shape == (K,)
    assert actions.shape == (K, N_agents)
    assert qvals.shape == (K,)
    assert mask.shape == (K,)
    assert not mask.any()
    assert pol.sum() == 0.0


def test_batch_shapes_no_reanalyze():
    config = make_config(use_reanalyze=False)
    net = MAMuZeroNet(config=config)
    obs = np.ones((1, N, OBS_DIM), dtype=np.float32)
    params = net.init(jax.random.PRNGKey(0), obs)
    worker = ReanalyzeWorker(config=config, model=net)
    ctx = make_buffer_ctx(config)
    batch = worker.make_batch(ctx, params)
    U = config.unroll_steps
    assert batch.obs.shape == (B, U + 1, N, OBS_DIM)
    assert batch.actions.shape == (B, U, N)
    assert batch.target_rewards.shape == (B, U)
    assert batch.target_values.shape == (B, U + 1)
    assert batch.target_policies.shape == (B, U + 1, K)
    assert batch.target_qvalues.shape == (B, U + 1, K)
    assert batch.target_masks.shape == (B, U + 1, K)
    assert batch.sampled_actions.shape == (B, U + 1, K, N)
    assert batch.weights.shape == (B,)


def test_policy_loss_no_gradient_through_value_baseline():
    """Policy gradient must not flow through V_net baseline into value params."""
    from jaxzero.train import make_update_fn

    # Zero out all losses except policy so any value-param gradient must come from policy loss
    config = make_config(use_reanalyze=False)
    config = dataclasses.replace(
        config,
        value_loss_coeff=0.0,
        reward_loss_coeff=0.0,
        consistency_coeff=0.0,
    )
    net = MAMuZeroNet(config=config)
    params = net.init(jax.random.PRNGKey(0), jnp.ones((1, N, OBS_DIM), dtype=jnp.float32))
    worker = ReanalyzeWorker(config=config, model=net)
    ctx = make_buffer_ctx(config)
    batch = worker.make_batch(ctx, params)

    update_fn = make_update_fn(net, config)
    _, grads, _, _ = update_fn(params, batch)

    # With stop_gradient, policy loss must not produce gradients in value_mlp output kernel
    value_out_kernel = grads['params']['prediction_net']['value_mlp']['output']['kernel']
    assert np.allclose(value_out_kernel, 0.0, atol=1e-6), (
        f"Policy gradient leaked into value_mlp output kernel. max={np.abs(value_out_kernel).max():.2e}"
    )


def _make_mock_search_result(B_flat, K, N):
    """Returns a SearchOutput where each position has K uniform visits to action 0."""
    return SearchOutput(
        root_value=np.zeros(B_flat, dtype=np.float32),
        sampled_actions=[np.zeros((K, N), dtype=np.int32) for _ in range(B_flat)],
        sampled_visit_counts=[np.ones(K, dtype=np.float32) for _ in range(B_flat)],
        sampled_qvalues=[np.full(K, 99.0, dtype=np.float32) for _ in range(B_flat)],
        sampled_imp_ratio=[np.ones(K, dtype=np.float32) for _ in range(B_flat)],
    )


def test_reanalysis_replaces_stored_policy_targets():
    """When use_reanalyze=True, make_batch must use fresh MCTS targets, not stored ones."""
    import sys

    config = make_config(use_reanalyze=True)
    config = dataclasses.replace(config, revisit_policy_search_rate=1.0)
    net = MAMuZeroNet(config=config)
    params = net.init(jax.random.PRNGKey(0), np.ones((1, N, OBS_DIM), dtype=np.float32))

    U = config.unroll_steps
    B_flat = B * (U + 1)  # all positions in one MCTS call

    mock_result = _make_mock_search_result(B_flat, K, N)

    # cytree is compiled for Python 3.10; mock the whole sampled_mcts module so
    # the local import inside ReanalyzeWorker.__init__ doesn't hit cytree.
    mock_mcts_instance = MagicMock()
    mock_mcts_instance.search.return_value = mock_result

    mock_sampled_mcts_module = MagicMock()
    mock_sampled_mcts_module.SampledMCTS.return_value = mock_mcts_instance

    with patch.dict(sys.modules, {'jaxzero.mcts.sampled_mcts': mock_sampled_mcts_module}):
        worker = ReanalyzeWorker(config=config, model=net)
        assert worker._mcts is not None

        ctx = make_buffer_ctx(config)
        batch = worker.make_batch(ctx, params)

    mock_mcts_instance.search.assert_called_once()
    call_obs = mock_mcts_instance.search.call_args[0][1]   # second positional arg = obs
    assert call_obs.shape == (B_flat, N, OBS_DIM), (
        f"Expected obs shape ({B_flat}, {N}, {OBS_DIM}), got {call_obs.shape}"
    )

    # Q-values must be 99.0 (from mock), not stored random values
    assert np.all(batch.target_qvalues == 99.0), (
        f"Reanalyzed Q-values should be 99.0 but got: {batch.target_qvalues[:2, 0, :3]}"
    )

    # Visit counts must be uniform 1/K (mock returns ones, normalized)
    expected_pol = 1.0 / K
    np.testing.assert_allclose(batch.target_policies, expected_pol, atol=1e-5)


def test_no_reanalysis_uses_stored_targets():
    """When use_reanalyze=False, make_batch must NOT call MCTS."""
    config = make_config(use_reanalyze=False)
    net = MAMuZeroNet(config=config)
    params = net.init(jax.random.PRNGKey(0), np.ones((1, N, OBS_DIM), dtype=np.float32))

    worker = ReanalyzeWorker(config=config, model=net)
    assert worker._mcts is None  # no MCTS when use_reanalyze=False

    ctx = make_buffer_ctx(config)
    games, positions, _, _ = ctx
    stored_qvals_0 = games[0].sampled_qvalues[int(positions[0])]

    batch = worker.make_batch(ctx, params)

    np.testing.assert_array_equal(
        batch.target_qvalues[0, 0],
        stored_qvals_0,
    )
