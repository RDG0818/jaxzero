# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Commands

```bash
# Setup
conda create -n muzero python=3.10.18 && conda activate muzero
pip install -r requirements.txt
wandb login  # set wandb_mode to "online" in config.py to enable

# Run training
python train.py

# Run the only existing unit test
python -m pytest unit_tests/test_model.py
```

To change environment, planner mode, or hyperparameters, edit `config.py` directly — there is no CLI argument parsing.

## Critical JAX + Ray Constraint

JAX eagerly allocates the entire GPU, and Ray spawns isolated processes. To avoid SEGFAULTs and driver crashes:
- **All JAX imports must be inside Ray actor methods/`__init__`**, never at module top-level in actor files.
- **No JAX objects in global scope** of any Ray-managed process.
- The replay buffer converts everything to NumPy before storage to prevent DeviceArrays crossing process boundaries.

Violating these rules causes silent failures or segfaults that are difficult to debug.

## Architecture

**Training system** (`train.py`): Asynchronous actor-learner pattern using Ray.
- `LearnerActor` (GPU): pulls batches from `ReplayBufferActor`, runs JIT-compiled training step, pushes updated params.
- `DataActor` (CPU, N instances): runs episodes using JIT-compiled MCTS planner, sends `ReplayItem`s to the buffer.
- `ReplayBufferActor`: holds a `ReplayBuffer` (prioritized experience replay) backed by pre-allocated NumPy arrays.
- Actors sync params every `param_update_interval` episodes.

**World model** (`model/model.py`, `model/attention.py`):
- `FlaxMAMuZeroNet`: top-level Flax module with four sub-networks.
  - `RepresentationNetwork`: encodes per-agent observation → latent state `(B, N, D)`.
  - `DynamicsNetwork`: takes latent + joint action → next latent + reward logits. Optionally applies `TransformerAttentionEncoder` before the MLP for inter-agent communication.
  - `PredictionNetwork`: latent → per-agent policy logits `(B, N, A)` + centralized value logits.
  - `ProjectionNetwork`: SimSiam-style self-supervised consistency head (online branch uses prediction head; target branch does not).
- Reward and value outputs are **categorical distributions** over a discrete support (tz-transform), not scalars. Use `utils.scalar_to_support` / `utils.support_to_scalar` to convert.
- `model.__call__` = initial inference; `model.recurrent_inference` = dynamics unroll (used inside MCTS).
- Parameter initialization trick: `__call__` calls `dynamics_net` and `project` with dummy inputs when `is_mutable_collection('params')` to force all submodule params to materialize in one pass.

**MCTS planners** (`mcts/`):
- `MCTSPlanner` (base): holds common config, `DiscreteSupport` objects, `add_dirichlet_noise`. Subclasses must set `self.plan_jit` and `self._recurrent_fn_jit` in `__init__`. Public entry point: `planner.plan(params, rng_key, obs)`.
- `MCTSIndependentPlanner`: runs one `gumbel_muzero_policy` search per agent via `jax.lax.scan`. Other agents' actions are fixed to argmax (or sampled) from the prior policy during each agent's search.
- `MCTSJointPlanner`: single search over the joint action space with sampling for scalability.
- All planners use `mctx.gumbel_muzero_policy` from DeepMind's MCTX library.

**Data flow**: `observation (B,N,obs_dim)` → representation → latent `(B,N,D)` → MCTS (calls `recurrent_inference` inside simulations) → `MCTSPlanOutput` (joint_action, policy_targets, root_value, agent_order) → `Transition` → `Episode` → `process_episode` (sliding window n-step returns) → `ReplayItem` → `ReplayBuffer`.

**Config** (`config.py`): Single frozen `ExperimentConfig = ExperimentConfig()` instance (`CONFIG`) imported everywhere. Three nested dataclasses: `ModelConfig`, `MCTSConfig`, `TrainConfig`.

## Key TODOs in the Codebase

- Model saving/loading (orbax-checkpoint is in requirements but unused)
- Reanalyze actors to reduce stale data in the replay buffer
- Environment wrapper abstract base class
- SMAC/jaxMARL environment wrapper (`utils/smax_env_wrapper.py` is a stub)
- Unit tests for `utils/` and `train.py`
- Remove `model.predict()` method and consolidate to `recurrent_inference` only (marked TODO in `model.py`)
- Full rewrite of `train.py` (see notes below)

## Environment Wrappers

`utils/mpe_env_wrapper.py` wraps JaxMARL MPE environments. Any new environment wrapper must expose:
- `reset(rng_key) → (observation, state)` where observation shape is `(1, N, obs_dim)`
- `step(rng_key, state, actions) → (next_obs, next_state, reward, done)`
- `observation_shape: Tuple[int, ...]`, `observation_size: int`, `action_space_size: int`

## Future Improvements

Collected from the refactor session. Items marked **[easy]** are straightforward implementation tasks; **[research]** require design decisions.

### Utils

- **[easy] Flashbax for the replay buffer** — [instadeep/flashbax](https://github.com/instadeep/flashbax) is a JAX-native PER buffer. Replace `ReplayBuffer` with `flashbax.make_prioritised_flat_buffer`. Eliminates the CPU→GPU batch transfer on every training step, which is the main bottleneck in the current data pipeline. Also provides a proper segment tree for O(log n) priority updates vs. the current O(n) recomputation.

- **[easy] Vectorized environment rollouts** — JaxMARL supports `jax.vmap` over `env.reset` and `env.step`. The `MPEEnvWrapper` currently returns batch size 1. Wrapping in `jax.vmap` allows running B parallel environments and returning `(B, N, obs)` directly, multiplying data throughput without changing the architecture.

- **[easy] Remove `use_logits` from `support_to_scalar`** — the parameter is always `True` in practice; remove it and unconditionally apply softmax.

- **[easy] Rename `_h`/`_h_inv`** → `muzero_scale` / `muzero_scale_inv` for readability.

### Model

- **[easy] EMA target encoder** — replace the SimSiam-style consistency head with a BYOL-style exponential moving average (EMA) target network. EMA targets are more stable and remove the need for the stop-gradient trick inside `project_target`. Requires adding an EMA parameter update step in the training loop.

- **[easy] Observation normalization** — add a running mean/variance normalization layer at the top of `RepresentationNetwork`. Stabilizes training when observation scales vary across environments. Can be implemented as a `flax.linen.Module` that updates running stats via `self.variable('stats', ...)`.

- **[medium] Recurrent dynamics (GRU/LSTM)** — replace or augment the MLP in `DynamicsNetwork` with a recurrent cell to handle partial observability. Requires carrying hidden state through MCTS simulations, which means passing it in the `embedding` field of `mctx.RootFnOutput`.

- **[medium] Multi-step consistency loss (SPR)** — instead of computing consistency only between step t and t+1, compute it over k unroll steps. SPR (Self-Predictive Representations, Schwarzer et al. 2021) shows this significantly improves sample efficiency. The unroll already exists; add consistency targets at each step.

- **[medium] Data augmentation for consistency** — apply random observation noise or masking before the representation network for the online branch only. The target branch sees clean observations. Shown to improve consistency loss quality in EfficientZero.

- **[research] Value Equivalence Model (VEM)** — only propagate gradients through model transitions in directions relevant to the policy/value, not all directions. Reduces model capacity wasted on irrelevant state dimensions (Grimm et al. 2020).

- **[research] Stochastic MuZero** — explicitly model transition uncertainty with a latent variable in `DynamicsNetwork`. Useful for stochastic environments where a deterministic model is systematically wrong.

### MCTS

- **[medium] Sequential MCTS (proper implementation)** — agents search in order, each conditioning on the committed actions of prior agents. The key design question is what to put in `coordination_info`: either the prior agents' selected actions (concatenated into the latent) or a communication vector from the prior agents' MCTS trees.

- **[medium] Temperature annealing** — Gumbel MuZero's `max_num_considered_actions` acts like temperature. Anneal it down over training (high early for exploration, low late for exploitation). Currently fixed at `num_gumbel_samples`.

- **[medium] Factored policy targets** — for `MCTSJointPlanner`, the current marginal extraction (summing over other agents' axes) discards coordination information. An alternative: keep the full joint policy as the target and train with a factored policy head that explicitly models correlations.

- **[research] Communication before commitment** — before each MCTS search, agents exchange a communication vector (e.g., their current policy embedding). This vector is concatenated to the latent state, allowing the dynamics model to condition on what other agents intend to do. Bridges the gap between independent and sequential MCTS.

- **[research] Count-based / intrinsic exploration** — add an exploration bonus to the MCTS root value based on visitation counts or a learned density model. Helps in sparse-reward cooperative tasks where the agents need to discover coordinated behaviors.

### Training / Infrastructure

- **[easy] Hydra config** — replace the single `ExperimentConfig` dataclass with Hydra for composable YAML configs and CLI overrides. Minimal code change; large usability improvement for sweeps.

- **[easy] Checkpointing** — orbax-checkpoint is already in requirements. Add periodic `checkpointer.save(step, {'params': params})` in `LearnerActor` and a restore path at startup.

- **[easy] Standalone eval script** — a `eval.py` that loads a checkpoint, runs N episodes with MCTS (no training), and logs mean return. Useful for comparing runs without re-training.

- **[medium] Reanalyze actors** — add a dedicated `ReanalyzeActor` that re-runs MCTS on stored observations with the latest params to generate fresher policy/value targets. Standard in MuZero but not yet implemented.
