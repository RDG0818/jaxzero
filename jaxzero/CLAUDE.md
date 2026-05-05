# jaxzero — MAZero JAX Rewrite

## What this is
Faithful JAX/Flax rewrite of MAZero (ICLR 2024). Targets SMAX 3m via JaxMARL.

## Key design decisions
- Python-tree MCTS (not mctx). Model calls JIT'd.
- CommunicationNet inside DynamicsNetwork per spec Eq. 4 (not pre-search).
- Positional encoding in CommunicationNet: learned nn.Embed, added to projected input.
- AWPO sharp loss. Policy loss: log π(a) * visit_count * exp(adv / α).
- Sync serial training loop first; Ray async in `train_async.py`.
- Config: frozen MAZeroConfig dataclass; replace obs_size/action_space_size at startup.
- BatchData.sampled_actions (B, U+1, K, N): wired from GameHistory through ReanalyzeWorker.
- PredictionNetwork.value_mlp: zero_init_output=False (non-trivial init needed for MCTS).
- Target network: LearnerActor.target_params refreshed every target_model_interval steps; reanalyze uses target_params (not live) for stable bootstrapping.
- MCTS GPU opt: hidden states in jnp pool (num_simulations+1, B, N, D); only scalars/logits transferred for ctree.
- ReanalyzeActor: async Ray actor sampling stored games, re-running MCTS, pushing fresh targets back. Enabled via num_reanalyze_actors > 0 in config or --num_reanalyze CLI flag.
- actors/__init__.py: direct imports (no lazy __getattr__); all actors safe to import after Ray init.

## AWPO loss details (train.py)

- `awpo_sharp_loss` takes `advantages = Q - V_net` where V_net = `phi_inv(value_logits, S_v)`.
- Normalization is BATCH-level (over all B*K valid entries), not per-sample. Per-sample (K=10) amplifies noise into random gradients and causes policy plateau.
- Clip applies to the normalized argument BEFORE exp: `exp(clip(adv_norm/alpha, -3, 3))` → weights in [0.05, 20]. Old code clipped the exp value which destroyed signal for high-advantage actions.

## Running
```
python -m jaxzero.main --env 3m
python -m jaxzero.main --env 3m --async_training
python -m jaxzero.main --env 3m --async_training --num_actors 4 --num_reanalyze 2
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
