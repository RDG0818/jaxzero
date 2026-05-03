import numpy as np
import jax
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet
from jaxzero.game import GameHistory
from jaxzero.replay_buffer import PrioritizedReplayBuffer
from jaxzero.reanalyze import ReanalyzeWorker, BatchData


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
