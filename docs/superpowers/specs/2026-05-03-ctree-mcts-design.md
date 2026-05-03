# ctree-backed Sampled MCTS Design

**Goal:** Replace the Python Node-based MCTS in `jaxzero/mcts/sampled_mcts.py` with the battle-tested C++/Cython ctree from the original MAZero repo, keeping the same external interface and batched JAX model calls.

**Architecture:** Copy MAZero's compiled ctree (C++/Cython) into the repo. Write a thin Python wrapper that manages the hidden-state pool and bridges ctree's index-based API with our JAX model. Tree topology, OS(λ), and UCB selection all run in C++. Model inference stays as batched JAX calls `(B, N, D)`.

**Tech Stack:** Cython, C++ (g++), JAX, numpy, existing `MAMuZeroNet` JAX model.

---

## Why

Current Python Node tree: dict lookups, object GC, Python-loop UCB = slow. ctree is the reference implementation used by the MAZero paper authors. OS(λ) is already correct and tested. The only missing piece is a JAX-compatible glue layer.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `jaxzero/mcts/ctree/` | Copy from `MAZero/core/mcts/ctree/ctree_sampled/` | C++/Cython tree (unchanged) |
| `jaxzero/mcts/ctree/setup.py` | Create | Build script for Cython extension |
| `jaxzero/mcts/sampled_mcts.py` | Rewrite | Python glue: pool management, JAX model calls, SearchOutput |
| `jaxzero/mcts/__init__.py` | No change | — |
| All other files | No change | train.py, reanalyze.py, game.py, config.py unchanged |

---

## ctree API (from MAZero cytree.pyx)

```python
# Construction
trees = cytree.Tree_batch(
    batch_size,           # B
    agent_num,            # N
    action_space_size,    # A
    sampled_times,        # K
    simulation_num,       # num_simulations
    tree_value_stat_delta_lb,
    random_seed,          # uint32
    rho,                  # mcts_rho (0.75)
    lam,                  # mcts_lambda (0.8)
)

# Root initialisation
trees.prepare(
    rewards,       # (B,) float32 — zeros at root
    values,        # (B,) float32 — phi_inv of value_logits
    policy_probs,  # (B, N, A) float32 — softmax of policy_logits, legal-masked
    beta,          # (B, N, A) float32 — policy_probs + Dirichlet noise
    sampled_times, # K
    noise_epsilon, # root_exploration_fraction
    noises,        # (B, N, A) float32 — Dirichlet samples
)

# Per-simulation
ix_lst, iy_lst, batch_actions = trees.batch_selection(pb_c_base, pb_c_init, discount)
# ix_lst: list[int] length B — pool depth index for selected node
# iy_lst: list[int] length B — batch (env) index
# batch_actions: (B, N) int32

trees.batch_expansion_and_backup(
    sim_idx + 1,   # hidden_state_index_x (1-based)
    discount,
    sampled_times,
    rewards,       # (B,) float32
    values,        # (B,) float32
    policy_probs,  # (B, N, A) float32
    beta,          # (B, N, A) float32 — no noise at non-root nodes
)

# Results
roots_values         = trees.get_roots_values()               # (B,) float32
roots_actions        = trees.get_roots_sampled_actions()      # list[B] of (K, N) int32
roots_visit_counts   = trees.get_roots_sampled_visit_count()  # list[B] of (K,) int32
roots_qvalues        = trees.get_roots_sampled_qvalues(discount) # list[B] of (K,) float32
roots_imp_ratio      = trees.get_roots_sampled_imp_ratio()    # list[B] of (K,) float32
```

---

## Hidden State Pool

ctree stores tree topology only. Hidden states live in a Python pool:

```
hidden_states_pool: list[np.ndarray]   # each entry shape (B, N, D)
hidden_states_pool[0] = initial hidden states from representation network
hidden_states_pool[sim_idx+1] = hidden states produced at simulation step sim_idx
```

`batch_selection` returns `(ix, iy)` per env — look up `hidden_states_pool[ix][iy]` to get `(N, D)` for that env, stack to `(B, N, D)` for the batched model call.

---

## Search Loop

```
1. jit_initial(params, obs)                        # (B, N, obs_dim) → MuZeroOutput
2. phi_inv(value_logits) → values (B,)
3. softmax(policy_logits) → policy_probs (B, N, A), apply legal mask
4. Dirichlet noise → beta (B, N, A)
5. hidden_states_pool = [np.array(out0.hidden_state)]
6. trees.prepare(zeros_reward, values, policy_probs, beta, ...)

for sim in range(num_simulations):
    ix_lst, iy_lst, batch_actions = trees.batch_selection(...)
    hidden_batch = np.stack([hidden_states_pool[ix][iy]
                             for ix, iy in zip(ix_lst, iy_lst)])  # (B, N, D)
    rec_out = jit_recurrent(params, jnp.array(hidden_batch), jnp.array(batch_actions))
    rewards = np.array(jit_rew(rec_out.reward_logits))   # (B,)
    values  = np.array(jit_val(rec_out.value_logits))    # (B,)
    policy_probs = softmax(np.array(rec_out.policy_logits))  # (B, N, A)
    hidden_states_pool.append(np.array(rec_out.hidden_state))
    trees.batch_expansion_and_backup(sim+1, discount, K, rewards, values, policy_probs, policy_probs)

return SearchOutput(
    root_value=roots_values,
    sampled_actions=roots_actions,
    sampled_visit_counts=roots_visit_counts,
    sampled_qvalues=roots_qvalues,
    sampled_imp_ratio=roots_imp_ratio,
)
```

---

## Key Differences from MAZero Glue

| MAZero original | jaxzero adapter |
|---|---|
| `model.recurrent_inference(hidden, actions)` PyTorch | `jit_recurrent(params, hidden, actions)` JAX |
| hidden shape `(B, hidden_size)` flat | hidden shape `(B, N, D)` — pool stores `(B, N, D)`, lookup gives `(N, D)`, stacked to `(B, N, D)` |
| value/reward already scalar in NetworkOutput | `phi_inv(logits)` → scalar via `jit_val` / `jit_rew` |
| legal masking before prepare | same — mask `policy_probs` before calling prepare |
| `torch.no_grad()` + `model.eval()` | JAX: no state, JIT handles it |

---

## External Interface (unchanged)

```python
class SampledMCTS:
    def __init__(self, config: MAZeroConfig, model): ...
    def search(
        self,
        params,
        obs: np.ndarray,    # (B, N, obs_dim)
        legal: np.ndarray,  # (B, N, A)
        rng: np.random.Generator,
    ) -> SearchOutput: ...
```

`train.py` and `reanalyze.py` call `mcts.search(...)` — no changes needed there.

---

## Build

```python
# jaxzero/mcts/ctree/setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "cytree",
    sources=["cytree.pyx", "lib/cnode.cpp", ...],
    include_dirs=[np.get_include(), "lib", "../common_lib"],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17"],
)
setup(ext_modules=cythonize([ext]))
```

Build command: `cd jaxzero/mcts/ctree && python setup.py build_ext --inplace`

---

## Config Fields Used

From `MAZeroConfig`: `pb_c_base`, `pb_c_init`, `discount`, `mcts_rho`, `mcts_lambda`, `root_dirichlet_alpha`, `root_exploration_fraction`, `num_simulations`, `sampled_action_times`, `tree_value_stat_delta_lb`, `num_agents`, `action_space_size`.

All already present in config — no new fields needed.

---

## Testing

- Existing `tests/test_mcts.py` tests `SampledMCTS.search()` via the public interface — should pass after rewrite without changes.
- Smoke: `python -m jaxzero.main --env mpe --num_simulations 10 --training_steps 50` runs without error.
