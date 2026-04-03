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
- `MCTSSequentialPlanner`: WIP — agents search sequentially, each conditioning on prior agents' results.
- `MCTSJointPlanner`: single search over the joint action space with sampling for scalability.
- All planners use `mctx.gumbel_muzero_policy` from DeepMind's MCTX library.

**Data flow**: `observation (B,N,obs_dim)` → representation → latent `(B,N,D)` → MCTS (calls `recurrent_inference` inside simulations) → `MCTSPlanOutput` (joint_action, policy_targets, root_value, agent_order) → `Transition` → `Episode` → `process_episode` (sliding window n-step returns) → `ReplayItem` → `ReplayBuffer`.

**Config** (`config.py`): Single frozen `ExperimentConfig = ExperimentConfig()` instance (`CONFIG`) imported everywhere. Three nested dataclasses: `ModelConfig`, `MCTSConfig`, `TrainConfig`.

## Key TODOs in the Codebase

- Model saving/loading (orbax-checkpoint is in requirements but unused)
- Reanalyze actors to reduce stale data in the replay buffer
- Environment wrapper abstract base class (exists in `synch` branch)
- SMAC/jaxMARL environment wrapper (`utils/smax_env_wrapper.py` exists but may be incomplete)
- Unit tests for all files — currently `unit_tests/test_model.py` is empty
- Remove `model.predict()` method and consolidate to `recurrent_inference` only (marked TODO in `model.py:278`)
- Fix stale TODO comments in `train.py` (lines 490, 503)

## Environment Wrappers

`utils/mpe_env_wrapper.py` wraps PettingZoo MPE environments. Any new environment wrapper must expose:
- `reset() → (observation, state)` where observation shape is `(N, obs_dim)`
- `step(state, actions) → (next_obs, next_state, reward, done)`
- `observation_size: int`, `observation_space: Tuple`, `action_space_size: int`
