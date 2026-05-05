import numpy as np
import pytest
from jaxzero.game import GameHistory


N, A, D = 3, 9, 80
OBS_DIM = D * 4
K = 10


def make_game(T=20):
    g = GameHistory(num_agents=N, obs_dim=D, action_space_size=A, stacked_observations=4)
    for t in range(T):
        obs = np.random.randn(N, D).astype(np.float32)
        g.store_observation(obs)
        g.store_action(np.zeros(N, dtype=np.int32))
        g.store_reward(1.0)
        g.store_legal_actions(np.ones((N, A), dtype=bool))
        g.store_root_value(1.0)
        g.store_pred_value(0.9)
        g.store_search_stats(
            sampled_actions=np.zeros((K, N), dtype=np.int32),
            visit_counts=np.ones(K) / K,
            qvalues=np.zeros(K),
            mask=np.ones(K, dtype=bool),
        )
    return g


def test_obs_stacked_shape():
    g = make_game(T=10)
    obs = g.obs(t=5, stacked_obs=4)
    assert obs.shape == (N, D * 4)


def test_obs_at_start_pads():
    """t=0 should pad with first observation repeated."""
    g = make_game(T=10)
    obs_t0 = g.obs(t=0, stacked_obs=4)
    obs_t1 = g.obs(t=1, stacked_obs=4)
    assert obs_t0.shape == (N, D * 4)
    assert obs_t1.shape == (N, D * 4)


def test_game_length():
    g = make_game(T=15)
    assert len(g) == 15


def test_make_target_shapes():
    g = make_game(T=20)
    obs_b, actions_b, rewards_b, values_b, policies_b, qvals_b, masks_b = g.make_target(
        pos=5, unroll_steps=5, td_steps=5, discount=0.99
    )
    assert obs_b.shape == (6, N, D * 4)    # pos + unroll_steps+1 obs
    assert actions_b.shape == (5, N)
    assert rewards_b.shape == (5,)
    assert values_b.shape == (6,)


def test_make_target_out_of_episode_masks_are_zero():
    """Steps past episode end must have all-False masks (no policy loss contribution)."""
    K, N, A, S = 5, 2, 3, 1
    game = GameHistory(num_agents=N, obs_dim=4, action_space_size=A, stacked_observations=S)

    # 3-step episode
    for t in range(3):
        game.store_observation(np.zeros((N, 4)))
        game.store_action(np.zeros(N, dtype=np.int32))
        game.store_reward(1.0)
        game.store_root_value(0.5)
        game.store_pred_value(0.0)
        game.store_legal_actions(np.ones((N, A), dtype=bool))
        game.store_search_stats(
            sampled_actions=np.zeros((K, N), dtype=np.int32),
            visit_counts=np.ones(K) / K,
            qvalues=np.zeros(K),
            mask=np.ones(K, dtype=bool),
        )

    # pos=1, unroll_steps=5 → steps k=2,3,4,5 go past T=3
    _, _, _, _, _, _, masks_batch = game.make_target(
        pos=1, unroll_steps=5, td_steps=3, discount=0.99
    )
    # k=0,1 are within episode (pos+k = 1,2 < 3), k=2..5 are past end
    assert masks_batch[0].all(), "step 0 within episode should have valid mask"
    assert masks_batch[1].all(), "step 1 within episode should have valid mask"
    for k in range(2, 6):
        assert not masks_batch[k].any(), f"step {k} past episode end should be all-False"
