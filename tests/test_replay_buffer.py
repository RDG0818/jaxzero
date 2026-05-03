import numpy as np
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.game import GameHistory
from jaxzero.replay_buffer import PrioritizedReplayBuffer


def make_game(T=20, N=3, obs_dim=80, A=9):
    g = GameHistory(num_agents=N, obs_dim=obs_dim, action_space_size=A, stacked_observations=4)
    K = 10
    for t in range(T):
        g.store_observation(np.random.randn(N, obs_dim).astype(np.float32))
        g.store_action(np.zeros(N, dtype=np.int32))
        g.store_reward(float(np.random.randn()))
        g.store_legal_actions(np.ones((N, A), dtype=bool))
        g.store_root_value(float(np.random.randn()))
        g.store_pred_value(float(np.random.randn()))
        g.store_search_stats(
            sampled_actions=np.zeros((K, N), dtype=np.int32),
            visit_counts=np.ones(K) / K,
            qvalues=np.zeros(K),
            mask=np.ones(K, dtype=bool),
        )
    return g


def make_buffer(size=1000):
    config = MAZeroConfig(
        replay_buffer_size=size,
        min_replay_size=10,
        priority_alpha=0.6,
        priority_beta_start=0.4,
        batch_size=4,
    )
    return PrioritizedReplayBuffer(config), config


def test_can_sample_false_when_empty():
    buf, cfg = make_buffer()
    assert not buf.can_sample(4)


def test_can_sample_true_after_enough_games():
    buf, cfg = make_buffer()
    for _ in range(20):
        buf.add(make_game())
    assert buf.can_sample(4)


def test_prepare_batch_shapes():
    buf, cfg = make_buffer()
    for _ in range(20):
        buf.add(make_game())
    games, positions, indices, weights = buf.prepare_batch_context(4, beta=0.4)
    assert len(games) == 4
    assert len(positions) == 4
    assert len(indices) == 4
    assert weights.shape == (4,)


def test_priority_update():
    buf, cfg = make_buffer()
    for _ in range(20):
        buf.add(make_game())
    games, positions, indices, weights = buf.prepare_batch_context(4, beta=0.4)
    new_priorities = np.ones(4)
    buf.update_priorities(indices, new_priorities)  # should not raise


def test_buffer_capacity_limit():
    buf, cfg = make_buffer(size=5)
    for _ in range(10):
        buf.add(make_game())
    assert buf.size <= 5
