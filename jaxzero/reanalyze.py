import numpy as np
from typing import NamedTuple, Any
from jaxzero.config import MAZeroConfig
from jaxzero.game import GameHistory


def _pad_to_k(
    visits: np.ndarray,   # (K_actual,) float32 visit counts, K_actual <= K
    actions: np.ndarray,  # (K_actual, N) int32 joint actions
    qvals: np.ndarray,    # (K_actual,) float32 Q-values
    K: int,               # target width
    N: int,               # number of agents
) -> tuple:
    """Pad ctree outputs to fixed width K. Returns (pol, sa, qv, mask).

    pol:  (K,) float32 — normalized visit counts (sum=1 over valid entries, 0 for padded)
    sa:   (K, N) int32 — sampled joint actions (padded with zeros)
    qv:   (K,) float32 — Q-values (padded with zeros)
    mask: (K,) bool    — True for real entries, False for padding
    """
    K_actual = len(visits)
    pol = np.zeros(K, dtype=np.float32)
    sa = np.zeros((K, N), dtype=np.int32)
    qv = np.zeros(K, dtype=np.float32)
    mask = np.zeros(K, dtype=bool)

    pol[:K_actual] = visits[:K_actual] / (visits[:K_actual].sum() + 1e-8)
    sa[:K_actual] = actions[:K_actual]
    qv[:K_actual] = qvals[:K_actual]
    mask[:K_actual] = True

    return pol, sa, qv, mask


class BatchData(NamedTuple):
    obs: np.ndarray              # (B, U+1, N, obs_dim)
    actions: np.ndarray          # (B, U, N)
    target_rewards: np.ndarray   # (B, U)
    target_values: np.ndarray    # (B, U+1)
    target_policies: np.ndarray  # (B, U+1, K) — visit counts
    target_qvalues: np.ndarray   # (B, U+1, K)
    target_masks: np.ndarray     # (B, U+1, K)
    sampled_actions: np.ndarray  # (B, U+1, K, N) — joint actions for AWPO
    weights: np.ndarray          # (B,)
    indices: np.ndarray          # (B,)


class ReanalyzeWorker:
    def __init__(self, config: MAZeroConfig, model=None):
        self.config = config
        self.model = model
        if config.use_reanalyze and model is not None:
            from jaxzero.mcts.sampled_mcts import SampledMCTS
            self._mcts = SampledMCTS(config=config, model=model)
            self._rng = np.random.default_rng(config.seed)
        else:
            self._mcts = None

    def make_batch(self, buffer_context, params: Any) -> BatchData:
        games, positions, indices, weights = buffer_context
        B = len(games)
        U = self.config.unroll_steps

        obs_list, actions_list, rewards_list, values_list = [], [], [], []
        policies_list, qvals_list, masks_list, sampled_actions_list = [], [], [], []

        for b in range(B):
            game: GameHistory = games[b]
            pos = int(positions[b])
            obs_b, act_b, rew_b, val_b, pol_b, qv_b, mask_b = game.make_target(
                pos=pos,
                unroll_steps=U,
                td_steps=self.config.td_steps,
                discount=self.config.discount,
            )
            # Gather sampled_actions for AWPO: (U+1, K, N)
            T = len(game)
            K = len(game.sampled_actions[0])
            sa_b = np.stack([
                game.sampled_actions[min(pos + k, T - 1)]
                for k in range(U + 1)
            ])  # (U+1, K, N)

            obs_list.append(obs_b)
            actions_list.append(act_b)
            rewards_list.append(rew_b)
            values_list.append(val_b)
            policies_list.append(pol_b)
            qvals_list.append(qv_b)
            masks_list.append(mask_b)
            sampled_actions_list.append(sa_b)

        return BatchData(
            obs=np.stack(obs_list),
            actions=np.stack(actions_list),
            target_rewards=np.stack(rewards_list),
            target_values=np.stack(values_list),
            target_policies=np.stack(policies_list),
            target_qvalues=np.stack(qvals_list),
            target_masks=np.stack(masks_list),
            sampled_actions=np.stack(sampled_actions_list),
            weights=weights,
            indices=indices,
        )
