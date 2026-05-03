# jaxzero — MAZero JAX Rewrite

## What this is
Faithful JAX/Flax rewrite of MAZero (ICLR 2024). Targets SMAX 3m via JaxMARL.

## Key design decisions
- Python-tree MCTS (not mctx). Model calls JIT'd.
- CommunicationNet inside DynamicsNetwork per spec Eq. 4 (not pre-search).
- Positional encoding in CommunicationNet: learned nn.Embed, added to projected input.
- AWPO sharp loss. Policy loss: log π(a) * visit_count * exp(adv / α).
- Sync serial training loop first; Ray async later.
- Config: frozen MAZeroConfig dataclass; replace obs_size/action_space_size at startup.
- BatchData.sampled_actions (B, U+1, K, N): wired from GameHistory through ReanalyzeWorker.
- PredictionNetwork.value_mlp: zero_init_output=False (non-trivial init needed for MCTS).

## Running
```
python -m jaxzero.main --env 3m
python -m jaxzero.main --env mpe  # sanity check
python -m pytest tests/ -v
```

## Known paper hyperparams (Table 1)
K=10 sampled actions, N=100 simulations, lr=1e-4, stacked_obs=4, unroll=5, td=5

## Prior failures (sequential-muzero)
1. mctx.gumbel_muzero_policy instead of Sampled MCTS
2. K=5/N=50 vs paper K=10/N=100
3. CommunicationNet outside dynamics
4. No positional encoding in transformer
