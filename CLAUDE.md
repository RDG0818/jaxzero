# jaxzero — MAZero JAX Rewrite

Faithful JAX/Flax rewrite of MAZero (ICLR 2024). Targets SMAX 3m via JaxMARL.

## Running

```bash
# Sync training
python -m jaxzero.main --env 3m

# Async training (Ray actors)
python -m jaxzero.main --env 3m --async_training
python -m jaxzero.main --env 3m --async_training --num_actors 4 --num_reanalyze 2

# Tests
python -m pytest tests/ -v
```

## Architecture

### Sync path
`train.py` — serial collect → reanalyze → train loop.

### Async path (`train_async.py`)
Ray actor-learner architecture:
- **LearnerActor** — owns model params + optimizer on GPU; serves params; trains
- **DataActor** — MCTS episode collection (CPU JAX); pushes to buffer
- **ReplayBufferActor** — `PrioritizedReplayBuffer` in isolated Ray process; no JAX
- **ReanalyzeActor** — samples stored games, re-runs MCTS, writes fresh targets back

All actors import JAX lazily inside `__init__` to avoid fork-after-JAX-init deadlock. Ray must init before any JAX import (`main.py` does this).

### Config (`config.py`)
Frozen `MAZeroConfig` dataclass. `obs_size`/`action_space_size` replaced at startup from env probe.

Key async fields:
| Field | Default | Meaning |
|-------|---------|---------|
| `num_actors` | 3 | DataActor count |
| `num_reanalyze_actors` | 0 | ReanalyzeActor count (0 = disabled) |
| `reanalyze_batch_size` | 8 | Positions per reanalyze pass |
| `param_update_interval` | 1 | How often actors pull fresh params |
| `learner_steps_per_call` | 10 | Train steps per learner task |
| `target_model_interval` | 200 | Steps between target param refresh |

## Key design decisions

- Python-tree MCTS (not mctx). Model calls JIT'd.
- CommunicationNet inside DynamicsNetwork per spec Eq. 4 (not pre-search).
- Positional encoding in CommunicationNet: learned `nn.Embed`, added to projected input.
- AWPO loss: `log π(a) * visit_count * exp(adv / α)`.
- Target network: `LearnerActor.target_params` refreshed every `target_model_interval` steps; used for reanalyze bootstrapping (not live params).
- MCTS hidden states kept on GPU (`jnp` array pool, shape `(num_simulations+1, B, N, D)`); only scalars/logits transferred to CPU for ctree.
- `BatchData.sampled_actions` shape: `(B, U+1, K, N)`.
- `PredictionNetwork.value_mlp`: `zero_init_output=False`.

## Paper hyperparams (Table 1)
K=10, N=100 simulations, lr=1e-4, stacked_obs=4, unroll=5, td=5

## Prior failures (sequential-muzero)
1. `mctx.gumbel_muzero_policy` instead of Sampled MCTS
2. K=5/N=50 vs paper K=10/N=100
3. CommunicationNet outside dynamics
4. No positional encoding in transformer
