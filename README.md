# Sequential MuZero

Multi-agent MuZero-style world-model in JAX/Flax for research and prototyping. Developed originally for the Tactical Behaviors for Autonomous Maneuver (TBAM) project in collaboration with Mississippi State University, Rutgers University, and the Army Research Lab.

Contact: rdg291@msstate.edu

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Research Directions & Future Ideas](#research-directions--future-ideas)
7. [What Didn't Work](#what-didnt-work)
8. [Relevant Papers](#relevant-papers)
9. [License](#license)

---

## Overview

This project is a model-based multi-agent reinforcement learning framework built on JAX/Flax. It extends MuZero-style planning (MCTS + learned world model) to cooperative multi-agent environments.

The core research question is how to efficiently search over the exponential joint action space in multi-agent environments. The primary direction explored here is **sequential MCTS**, where each agent searches over its own individual action space and conditions on the plans of prior agents. This reduces planning complexity from exponential to linear in the number of agents while preserving coordination signals.

---

## Features

### World Model (`model/`)

- **Representation network**: encodes per-agent observations `(B, N, obs_dim)` into latent states `(B, N, D)`.
- **Dynamics network**: takes a latent state and joint actions, returns the next imagined latent state and reward logits. Optionally prepends a `TransformerAttentionEncoder` for inter-agent communication in latent space.
- **Prediction network**: maps a latent state to per-agent policy logits `(B, N, A)` and a centralized value estimate.
- **Projection network**: BYOL-style temporal consistency head. Online branch applies projection + prediction MLP; target branch uses a slowly-moving EMA copy of the parameters (`ema_decay=0.999`), updated outside the JIT boundary.
- **Categorical reward/value**: scalar targets are encoded as distributions over a discrete support (tz-transform) for improved training stability. See `utils/transforms.py`.
- **Communication before commitment** (`use_root_communication`): an optional cross-agent attention pass applied to root latent states *before* independent MCTS searches begin. Lets agents broadcast their current hidden state to all other agents, enabling each agent to condition its search on a summary of the global state — without requiring inter-search communication or breaking CTDE.

### MCTS Planners (`mcts/`)

All planners use [`mctx.gumbel_muzero_policy`](https://github.com/google-deepmind/mctx) — DeepMind's JAX-native Gumbel MuZero MCTS.

- **`MCTSIndependentPlanner`**: one Gumbel MuZero search per agent via `jax.lax.scan`. Other agents are fixed to their policy argmax during each agent's search. Supports `use_root_communication` to run a cross-agent attention pass before the searches.
- **`MCTSJointPlanner`**: single search over the combinatorial joint action space `A^N` with independence-factored logits. Marginal policy targets extracted by summing over other agents' axes. Scales poorly with N but provides a strong coordination baseline.
- Dirichlet noise is added to the root of each search to encourage exploration.

### Training Infrastructure (`actors/`, `training/`)

Asynchronous actor-learner architecture using [Ray](https://www.ray.io), inspired by EfficientZero.

- **`LearnerActor`** (GPU): self-driving training loop — runs N steps internally per Ray call to amortize scheduling overhead (~100ms/call). Prefetches the next batch while the GPU trains. EMA update is JIT-compiled. All GPU→CPU metrics and priorities are packed into a single `jnp.concatenate` and transferred in one DMA transaction.
- **`DataActor`** (CPU, multiple instances): runs MCTS episodes, ships `ReplayItem`s to the replay buffer. Parameter syncs are asynchronous: `get_params.remote()` is fired at end of episode and resolved at the start of the next, so the ~300ms transfer overlaps with MCTS compute.
- **`ReplayBufferActor`**: wraps `ReplayBuffer` with prioritized experience replay (PER). Uses a hand-written C++ backend when built (see below), falling back to a pure-Python implementation if the `.so` is not present.
- **`ReanalyzeActor`** (CPU, optional): continuously re-runs MCTS on stored observations with the latest parameters to generate fresher policy/value targets. Writes only to position 0 (root) of each stored sequence.

### C++ Replay Buffer (`csrc/replay_buffer/`)

A hand-written, lock-free prioritized replay buffer with pybind11 Python bindings.

- **Lock-free ring buffer + sum tree**: `std::atomic<int64_t>` write pointer; sum tree nodes stored as `std::atomic<uint32_t>` reinterpreted as IEEE-754 floats via `std::bit_cast`. Delta propagation uses a CAS loop for float-safe atomic addition. All atomics use `memory_order_relaxed` — sufficient for the single Ray actor (SPSC) use case.
- **CUDA pinned output buffers**: `sample()` and `sample_for_reanalysis()` gather directly into `cudaMallocHost` memory (falls back to `calloc` without CUDA). This eliminates the pageable→pinned intermediate copy that `jax.device_put()` would otherwise incur — saving ~50–200 µs per training step at 200KB batch size.
- **Stratified PER sampling**: divides the priority sum into B equal segments and draws one sample per segment using an xorshift64 PRNG. Reduces variance compared to naive importance sampling.
- **Vitter's Algorithm R**: `sample_for_reanalysis()` uses reservoir sampling for uniform selection without replacement in O(n) time with no extra allocation.
- **Pure-Python fallback**: if the `.so` is not built, `utils/replay_buffer.py` falls back to `cpprb` + NumPy arrays automatically. The public API is identical.

### Baselines (`baselines/`)

IPPO and MAPPO implemented in pure JAX (no Ray):
- Parameter sharing across agents; agent one-hot ID appended to observations.
- Vectorized rollout via `jax.lax.scan` over T timesteps × B parallel environments.
- **MAPPO centralized critic**: global state = concatenation of all agent observations `(B, N*obs_size)`.

---

## Installation

### Local

```bash
git clone https://github.com/RDG0818/sequential-muzero
cd sequential-muzero

conda create -n mazero python=3.10.18
conda activate mazero

pip install -r requirements.txt
wandb login  # optional; set wandb_mode: "online" in configs/train/default.yaml to enable

# Build the C++ replay buffer (run once after cloning, or after modifying csrc/)
pip install pybind11
python setup.py build_ext --inplace
python -c "import _replay_buffer_cpp; print('C++ backend loaded')"
```

### Docker

Tested on AWS `g4dn.8xlarge` with NVIDIA Container Toolkit.

```bash
docker build -t muzero .
docker run --gpus all --shm-size=32g -it --rm muzero
python train/muzero.py
```

### Running

```bash
# MuZero (default config)
python train/muzero.py

# Switch MCTS planner
python train/muzero.py mcts=joint

# Override hyperparameters
python train/muzero.py train.num_episodes=50000 train.batch_size=256

# Enable root communication
python train/muzero.py mcts.use_root_communication=true

# Baselines
python train/ippo.py
python train/mappo.py

# Evaluation
python eval.py
python eval.py eval_episodes=200 train.checkpoint_dir=checkpoints

# Tests
pytest tests/ -v
```

---

## Project Structure

```
csrc/
  replay_buffer/
    sum_tree.h            # lock-free atomic sum tree (header-only)
    pinned_alloc.h        # cudaMallocHost / calloc fallback (header-only)
    replay_buffer.h/.cpp  # ReplayBuffer C++ class
    bindings.cpp          # pybind11 module (_replay_buffer_cpp)
CMakeLists.txt            # builds _replay_buffer_cpp.*.so; CUDA optional
setup.py                  # python setup.py build_ext --inplace
train/
  muzero.py               # entry point: builds ExperimentConfig, launches Ray actors
  ippo.py                 # IPPO baseline (pure JAX, no Ray)
  mappo.py                # MAPPO baseline (pure JAX, no Ray)
eval.py                   # loads checkpoint, runs N MCTS episodes, logs return
config.py                 # ModelConfig, MCTSConfig, TrainConfig, ExperimentConfig dataclasses
configs/
  config.yaml             # root defaults
  model/default.yaml      # model architecture hyperparameters
  mcts/default.yaml       # MCTS hyperparameters (independent planner)
  mcts/joint.yaml         # joint planner preset
  train/default.yaml      # training hyperparameters
  baseline/               # IPPO / MAPPO hyperparameters
actors/
  learner_actor.py        # LearnerActor (GPU)
  data_actor.py           # DataActor (CPU)
  reanalyze_actor.py      # ReanalyzeActor (CPU)
  replay_buffer_actor.py  # ReplayBufferActor (wraps ReplayBuffer)
training/
  loop.py                 # run_warmup(), run_training_loop()
model/
  model.py                # FlaxMAMuZeroNet and sub-networks
  attention.py            # TransformerAttentionEncoder
  layers.py               # MLP
mcts/
  base.py                 # MCTSPlanner base class, MCTSPlanOutput
  mcts_independent.py     # MCTSIndependentPlanner (+ optional root communication)
  mcts_joint.py           # MCTSJointPlanner
envs/
  mpe_env_wrapper.py      # MPEEnvWrapper + VecMPEEnvWrapper (JaxMARL MPE)
  smax_env_wrapper.py     # SMAC environment wrapper
baselines/
  networks.py             # ActorCritic + CentralizedCritic Flax modules
  ippo.py                 # IPPO: GAE + PPO clip
  mappo.py                # MAPPO: GAE + PPO clip + centralized critic
utils/
  transforms.py           # DiscreteSupport, scalar_to_support, support_to_scalar, muzero_scale/inv
  replay_buffer.py        # ReplayBuffer, ReplayItem, Episode, Transition, process_episode
  logging_utils.py        # logger singleton
tests/
  test_model.py
  test_mcts.py
  test_replay_buffer.py   # 20 tests: shapes, dtypes, PER sampling, update_targets, reanalysis
```

### For the Next Developer

JAX eagerly allocates the entire GPU, and Ray spawns isolated processes. This combination requires discipline:

- **All JAX imports must be inside Ray actor methods / `__init__`**, never at module top-level in actor files.
- **No JAX objects in global scope** of any Ray-managed process.
- The replay buffer converts everything to NumPy before storage to prevent `DeviceArray`s crossing process boundaries.

Violating these rules causes SEGFAULTs or silent failures that are hard to debug. If you hit strange Ray+JAX errors, reach out: rdg291@msstate.edu

---

## Configuration

All hyperparameters live in `configs/`. Pass overrides on the command line — no code changes needed.

Key settings in `configs/mcts/default.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `planner_mode` | `independent` | `independent` or `joint` |
| `num_simulations` | `50` | MCTS simulations per agent per step |
| `num_gumbel_samples` | `8` | Actions considered at root (temperature proxy) |
| `use_root_communication` | `false` | Cross-agent attention before independent searches |
| `independent_argmax` | `true` | Fix off-turn agents to argmax (vs. sample) |

Key settings in `configs/train/default.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `batch_size` | `256` | Samples per training step |
| `unroll_steps` | `5` | Dynamics unroll length |
| `n_step` | `10` | n-step return horizon |
| `num_data_actors` | `4` | Parallel CPU data collection workers |
| `num_reanalyze_actors` | `1` | Reanalysis workers (0 to disable) |
| `replay_buffer_capacity` | `100000` | Replay buffer size |

---

## Research Directions & Future Ideas

### Core Research Problem

Instead of searching over the joint action space $A^N$ (exponential), decompose planning into $N$ individual searches over $A$ each (linear). The challenge is enabling coordination between agents who plan independently.

This is implemented in `mcts/mcts_independent.py`. The key open research questions are:

- What information should be shared between agents before or during their searches?
- How should prior agents' decisions influence later agents' searches without breaking CTDE?
- How do we avoid temporal staleness when a planning signal derived at the root becomes stale deeper in the tree?

### Communication Before Commitment (Implemented)

An optional cross-agent attention pass (`use_root_communication=true`) applied to root latent states before any MCTS search runs. Each agent's hidden state becomes a weighted mixture of all agents' hidden states, giving each search a global view of the team's current state. Zero overhead when disabled.

This is the least invasive coordination mechanism in the codebase — worth trying first before more complex inter-search communication.

### Other Future Directions

**Model:**
- **Observation normalization** — running mean/variance layer at the top of `RepresentationNetwork`. Stabilizes training when observation scales vary across environments.
- **Recurrent dynamics (GRU/LSTM)** — replace or augment the MLP in `DynamicsNetwork` with a recurrent cell for partial observability. Requires carrying hidden state through MCTS simulations.
- **Multi-step consistency loss (SPR)** — compute consistency targets over k unroll steps instead of just step t→t+1. Schwarzer et al. (2021) show significant sample efficiency gains.
- **Data augmentation for consistency** — apply random noise or masking to the online branch's observations only. Target branch sees clean observations.

**MCTS:**
- **Sequential MCTS** — agents plan in order, each conditioning on committed actions of prior agents. The key design question: concatenate prior actions into the latent, or pass a communication vector between trees?
- **Temperature annealing** — anneal `num_gumbel_samples` from high (exploration) to low (exploitation) over training.
- **Factored policy targets** — for `MCTSJointPlanner`, preserve joint coordination information instead of marginalizing it out.

**Training:**
- **Larger batch sizes** — 512 or 1024 amortizes Ray scheduling overhead better. The C++ replay buffer handles high throughput without contention.
- **More DataActors** — GPU learner is the bottleneck after the self-driving loop fix and async param sync. Adding more actors still helps data throughput.

---

## What Didn't Work

### Permutation Invariant Critic

Integrated a [Permutation Invariant Critic](https://arxiv.org/pdf/1911.00025) (PIC) as the reward/value head. Minimal performance gains on SMAC environments, significant architectural complexity, insufficient novelty. May still be useful with larger agent counts.

### Synchronous Training

Replaced the asynchronous architecture with a fully synchronous JAX training loop (see `synch` branch). Performance was worse than random — the policy entered a negative learning cycle with unclear root cause. Reduced data diversity is a likely factor, but doesn't fully explain the degradation. Worth revisiting to eliminate Ray entirely.

### Delta Network + Coordination Context

A GRU coordination cell aggregated hidden states and action statistics from the previous agent's search into a "planning vector" passed to the next agent. A separate delta MLP computed a correction to off-turn agents' actions.

Failed because: no clear training objective (matching MCTS targets pushed delta to zero; value-based gradients inapplicable to strictly off-policy data), and the planning vector was only meaningful at the root but had to be applied steps later, introducing temporal staleness.

Conceptually the most promising architecture seen for combining inter-agent conditioning with CTDE. If someone solves the objective and staleness problems, this may still be viable.

### Policy Network Conditioning

Simplified the delta network by directly feeding the planning vector into the policy network (zero vector when absent). Still suffered from temporal staleness in the search. Also creates a training/execution mismatch: training almost always has the planning vector; execution almost never does. The policy improving with the vector doesn't imply it improves without it.

### Autoregressive Policy Network

Attempted to implement the autoregressive policy from [Multi-Agent Transformer](https://arxiv.org/pdf/2205.14953) — sequential action generation with transformer-based inter-agent dependencies. Blocked by JAX tracer issues from the recursive sequential structure. Even with a working implementation, each policy call becomes quadratic in agents × search steps, incompatible with efficient MCTS in non-toy environments.

This is the most promising idea for coordination quality. Strong empirical evidence in the literature. Extremely complex implementation — expect significant time on architecture, profiling, and attention optimizations (KV caching, quantization) before it's viable.

---

## Relevant Papers

- [MuZero](https://arxiv.org/pdf/1911.08265)
- [Gumbel MuZero](https://openreview.net/pdf?id=bERaNdoegnO)
- [EfficientZero](https://arxiv.org/pdf/2111.00210)
- [MAZero](https://openreview.net/pdf?id=CpnKq3UJwp)
- [Multi-Agent Transformer](https://arxiv.org/pdf/2205.14953)
- [Permutation Invariant Critic](https://arxiv.org/pdf/1911.00025)
- [Heterogeneous Agent Reinforcement Learning](https://arxiv.org/pdf/2304.09870)
- [Self-Predictive Representations (SPR)](https://arxiv.org/pdf/2007.05929)

---

## License

This project is licensed under the [MIT License](LICENSE).
You are free to use, modify, and distribute this software with attribution.
