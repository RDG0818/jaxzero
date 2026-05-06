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
        K = self.config.sampled_action_times
        N = self.config.num_agents

        re_num = int(np.ceil(B * self.config.revisit_policy_search_rate))
        if self._mcts is None:
            re_num = 0

        fresh_policies = None
        fresh_qvalues = None
        fresh_masks = None
        fresh_sa = None

        if re_num > 0:
            obs_list, legal_list = [], []
            for b in range(re_num):
                game = games[b]
                pos = int(positions[b])
                T = len(game)
                for k in range(U + 1):
                    t = min(pos + k, T - 1)
                    obs_list.append(game.obs(t, self.config.stacked_observations))
                    legal_list.append(game.legal_actions[t])

            obs_np = np.stack(obs_list)     # (re_num*(U+1), N, obs_size)
            legal_np = np.stack(legal_list) # (re_num*(U+1), N, A)

            result = self._mcts.search(params, obs_np, legal_np, self._rng)

            fp = np.zeros((re_num, U + 1, K), dtype=np.float32)
            fq = np.zeros((re_num, U + 1, K), dtype=np.float32)
            fm = np.zeros((re_num, U + 1, K), dtype=bool)
            fsa = np.zeros((re_num, U + 1, K, N), dtype=np.int32)

            for flat_idx in range(re_num * (U + 1)):
                b = flat_idx // (U + 1)
                k = flat_idx % (U + 1)
                pol, sa, qv, mask = _pad_to_k(
                    result.sampled_visit_counts[flat_idx],
                    result.sampled_actions[flat_idx],
                    result.sampled_qvalues[flat_idx],
                    K, N,
                )
                fp[b, k] = pol
                fsa[b, k] = sa
                fq[b, k] = qv
                fm[b, k] = mask

            fresh_policies = fp
            fresh_qvalues = fq
            fresh_masks = fm
            fresh_sa = fsa

        obs_list_out, actions_list_out, rewards_list_out = [], [], []
        values_list_out, policies_list_out, qvals_list_out = [], [], []
        masks_list_out, sa_list_out = [], []

        for b in range(B):
            game = games[b]
            pos = int(positions[b])
            T = len(game)

            obs_b, act_b, rew_b, val_b, pol_b, qv_b, mask_b = game.make_target(
                pos=pos,
                unroll_steps=U,
                td_steps=self.config.td_steps,
                discount=self.config.discount,
            )

            if b < re_num and fresh_policies is not None:
                pol_b = fresh_policies[b]
                qv_b = fresh_qvalues[b]
                mask_b = fresh_masks[b]
                sa_b = fresh_sa[b]
            else:
                sa_b = np.stack([
                    game.sampled_actions[min(pos + k, T - 1)]
                    for k in range(U + 1)
                ])

            obs_list_out.append(obs_b)
            actions_list_out.append(act_b)
            rewards_list_out.append(rew_b)
            values_list_out.append(val_b)
            policies_list_out.append(pol_b)
            qvals_list_out.append(qv_b)
            masks_list_out.append(mask_b)
            sa_list_out.append(sa_b)

        return BatchData(
            obs=np.stack(obs_list_out),
            actions=np.stack(actions_list_out),
            target_rewards=np.stack(rewards_list_out),
            target_values=np.stack(values_list_out),
            target_policies=np.stack(policies_list_out),
            target_qvalues=np.stack(qvals_list_out),
            target_masks=np.stack(masks_list_out),
            sampled_actions=np.stack(sa_list_out),
            weights=weights,
            indices=indices,
        )
