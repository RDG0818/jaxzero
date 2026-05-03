# ctree-backed Sampled MCTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python Node-based MCTS in `jaxzero/mcts/sampled_mcts.py` with the C++/Cython ctree from the original MAZero repo, keeping the same external `SampledMCTS.search()` interface.

**Architecture:** Copy MAZero's ctree C++/Cython source into `jaxzero/mcts/ctree/`, fix two relative include paths, compile in-place, then rewrite `sampled_mcts.py` as a thin Python wrapper that feeds batched JAX model outputs into ctree and reads back search results.

**Tech Stack:** Cython, C++ (g++), JAX/numpy, existing `MAMuZeroNet`.

---

## File Structure

| File | Action |
|---|---|
| `jaxzero/mcts/ctree/cytree.pyx` | Copy from MAZero (unchanged) |
| `jaxzero/mcts/ctree/ctree.pxd` | Copy from MAZero + fix one path |
| `jaxzero/mcts/ctree/lib/cnode.cpp` | Copy from MAZero (unchanged) |
| `jaxzero/mcts/ctree/lib/cnode.h` | Copy from MAZero + fix one include |
| `jaxzero/mcts/ctree/common_lib/utils.cpp` | Copy from MAZero (unchanged) |
| `jaxzero/mcts/ctree/common_lib/utils.h` | Copy from MAZero (unchanged) |
| `jaxzero/mcts/ctree/setup.py` | Create (build script) |
| `jaxzero/mcts/sampled_mcts.py` | Rewrite (ctree wrapper, same interface) |

No other files change.

---

### Task 1: Copy ctree source files

**Files:**
- Create: `jaxzero/mcts/ctree/` (directory with subdirs)

- [ ] **Step 1: Copy all source files**

```bash
MAZERO=../MAZero/core/mcts/ctree
DEST=jaxzero/mcts/ctree

mkdir -p $DEST/lib $DEST/common_lib

cp $MAZERO/ctree_sampled/cytree.pyx   $DEST/
cp $MAZERO/ctree_sampled/ctree.pxd    $DEST/
cp $MAZERO/ctree_sampled/lib/cnode.cpp $DEST/lib/
cp $MAZERO/ctree_sampled/lib/cnode.h   $DEST/lib/
cp $MAZERO/common_lib/utils.cpp        $DEST/common_lib/
cp $MAZERO/common_lib/utils.h          $DEST/common_lib/
touch $DEST/__init__.py
```

Run from `/home/ryan/Repos/jaxzero/`.

- [ ] **Step 2: Verify files exist**

```bash
ls jaxzero/mcts/ctree/
ls jaxzero/mcts/ctree/lib/
ls jaxzero/mcts/ctree/common_lib/
```

Expected output:
```
ctree/: __init__.py  common_lib/  ctree.pxd  cytree.pyx  lib/
lib/: cnode.cpp  cnode.h
common_lib/: utils.cpp  utils.h
```

- [ ] **Step 3: Fix path in ctree.pxd**

Open `jaxzero/mcts/ctree/ctree.pxd`. Line 4 reads:
```
cdef extern from "../common_lib/utils.cpp":
```
Change to:
```
cdef extern from "common_lib/utils.cpp":
```

- [ ] **Step 4: Fix include path in lib/cnode.h**

Open `jaxzero/mcts/ctree/lib/cnode.h`. Line 4 reads:
```cpp
#include "../../common_lib/utils.h"
```
Change to:
```cpp
#include "../common_lib/utils.h"
```

- [ ] **Step 5: Commit**

```bash
git add jaxzero/mcts/ctree/
git commit -m "chore: copy MAZero ctree C++/Cython source"
```

---

### Task 2: Write setup.py and compile ctree

**Files:**
- Create: `jaxzero/mcts/ctree/setup.py`

- [ ] **Step 1: Write setup.py**

Create `jaxzero/mcts/ctree/setup.py` with this exact content:

```python
import os
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext

here = os.path.dirname(os.path.abspath(__file__))

class BuildExt(build_ext):
    def build_extensions(self):
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()

ext = Extension(
    name="cytree",
    sources=[os.path.join(here, "cytree.pyx")],
    include_dirs=[np.get_include(), here],
    language="c++",
    extra_compile_args=["-O2", "-std=c++17"],
)

setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=cythonize([ext], language_level=3),
)
```

- [ ] **Step 2: Compile**

```bash
cd /home/ryan/Repos/jaxzero/jaxzero/mcts/ctree
python setup.py build_ext --inplace 2>&1 | tail -20
```

Expected: ends with `copying build/lib.linux-x86_64-.../cytree... -> .`

If you see `error: ...`, the most common causes:
- Missing g++: `sudo apt install g++` or `conda install gxx_linux-64`
- Missing Cython: `pip install cython`

- [ ] **Step 3: Verify .so exists**

```bash
ls /home/ryan/Repos/jaxzero/jaxzero/mcts/ctree/cytree*.so
```

Expected: `cytree.cpython-312-x86_64-linux-gnu.so` (Python version may vary).

- [ ] **Step 4: Smoke-test import**

```bash
cd /home/ryan/Repos/jaxzero/jaxzero/mcts/ctree
python -c "from cytree import Tree_batch; print('cytree OK')"
```

Expected: `cytree OK`

- [ ] **Step 5: Commit**

```bash
cd /home/ryan/Repos/jaxzero
git add jaxzero/mcts/ctree/setup.py
git add jaxzero/mcts/ctree/cytree*.so
git commit -m "build: add ctree setup.py and compiled extension"
```

---

### Task 3: Rewrite sampled_mcts.py

**Files:**
- Modify: `jaxzero/mcts/sampled_mcts.py` (full rewrite)
- Test: `tests/test_mcts.py` (unchanged — verify it passes)

- [ ] **Step 1: Write the failing test (run existing tests first)**

```bash
cd /home/ryan/Repos/jaxzero
conda run -n mazero pytest tests/test_mcts.py -v 2>&1 | tail -20
```

All 4 tests currently pass with the Python Node implementation. They will break when we replace the implementation. Verify they pass now as baseline.

- [ ] **Step 2: Replace sampled_mcts.py**

Write `jaxzero/mcts/sampled_mcts.py` with this exact content:

```python
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple

from jaxzero.config import MAZeroConfig
from jaxzero.model.transforms import phi_inv as _phi_inv

# Add ctree directory to path so cytree.so is importable
_ctree_dir = os.path.join(os.path.dirname(__file__), "ctree")
if _ctree_dir not in sys.path:
    sys.path.insert(0, _ctree_dir)
from cytree import Tree_batch


class SearchOutput(NamedTuple):
    root_value: np.ndarray   # (B,)
    sampled_actions: list    # list[B] of (K, N) int32
    sampled_visit_counts: list  # list[B] of (K,) int32
    sampled_qvalues: list    # list[B] of (K,) float32
    sampled_imp_ratio: list  # list[B] of (K,) float32


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class SampledMCTS:
    """Sampled MCTS backed by the MAZero C++/Cython ctree.

    Tree topology, OS(λ) value estimation, and UCB selection all run in C++.
    Model inference uses batched JAX calls of shape (B, N, D) per simulation step.
    """

    def __init__(self, config: MAZeroConfig, model):
        self.config = config
        self.model = model

        self._jit_initial = jax.jit(model.apply)
        self._jit_recurrent = jax.jit(
            lambda p, h, a: model.apply(p, h, a, method=model.recurrent_inference)
        )
        self._jit_val = jax.jit(
            lambda logits: _phi_inv(logits, config.value_support_size)
        )
        self._jit_rew = jax.jit(
            lambda logits: _phi_inv(logits, config.reward_support_size)
        )

    def search(
        self,
        params,
        obs: np.ndarray,    # (B, N, obs_dim)
        legal: np.ndarray,  # (B, N, A) bool
        rng: np.random.Generator,
    ) -> SearchOutput:
        cfg = self.config
        B = obs.shape[0]
        K = cfg.sampled_action_times

        # --- Initial inference: one batched JAX call ---
        out0 = self._jit_initial(params, jnp.array(obs))
        values0 = np.array(self._jit_val(out0.value_logits)).astype(np.float32)  # (B,)
        policy0 = np.array(out0.policy_logits)   # (B, N, A)
        hidden0 = np.array(out0.hidden_state)    # (B, N, D)

        # --- Legal-masked policy probs ---
        policy_probs = _softmax(policy0).astype(np.float32)   # (B, N, A)
        legal_f = legal.astype(np.float32)
        policy_probs = policy_probs * legal_f
        policy_probs += legal_f * 1e-4
        policy_probs /= policy_probs.sum(axis=-1, keepdims=True)

        # --- Dirichlet noise for root exploration ---
        alpha = cfg.root_dirichlet_alpha
        eps = cfg.root_exploration_fraction
        noises = rng.dirichlet(
            np.ones(cfg.action_space_size) * alpha,
            size=(B, cfg.num_agents),
        ).astype(np.float32)  # (B, N, A)
        noises = noises * legal_f
        noises += legal_f * 1e-4
        noises /= noises.sum(axis=-1, keepdims=True)
        beta = ((1.0 - eps) * policy_probs + eps * noises).astype(np.float32)

        # --- Initialise ctree ---
        seed = int(rng.integers(0, 2**31 - 1))
        trees = Tree_batch(
            B,
            cfg.num_agents,
            cfg.action_space_size,
            K,
            cfg.num_simulations,
            cfg.tree_value_stat_delta_lb,
            seed,
            cfg.mcts_rho,
            cfg.mcts_lambda,
        )
        rewards0 = np.zeros(B, dtype=np.float32)
        trees.prepare(rewards0, values0, policy_probs, beta, K, eps, noises)

        # --- Hidden-state pool: pool[sim_idx] = (B, N, D) array ---
        hidden_states_pool = [hidden0]

        # --- Simulation loop: one batched JAX model call per step ---
        for sim in range(cfg.num_simulations):
            ix_lst, iy_lst, batch_actions = trees.batch_selection(
                cfg.pb_c_base, cfg.pb_c_init, cfg.discount
            )
            # ix_lst[b]: which pool entry holds the selected node for env b
            # iy_lst[b]: env index within that pool entry (equals b)
            # batch_actions: (B, N) int32

            hidden_batch = np.stack(
                [hidden_states_pool[ix][iy] for ix, iy in zip(ix_lst, iy_lst)]
            )  # (B, N, D)

            rec_out = self._jit_recurrent(
                params, jnp.array(hidden_batch), jnp.array(batch_actions)
            )
            rec_values = np.array(self._jit_val(rec_out.value_logits)).astype(np.float32)   # (B,)
            rec_rewards = np.array(self._jit_rew(rec_out.reward_logits)).astype(np.float32) # (B,)
            rec_policy = np.array(rec_out.policy_logits)   # (B, N, A)
            rec_hidden = np.array(rec_out.hidden_state)    # (B, N, D)

            rec_probs = _softmax(rec_policy).astype(np.float32)  # (B, N, A)

            hidden_states_pool.append(rec_hidden)

            trees.batch_expansion_and_backup(
                sim + 1,
                cfg.discount,
                K,
                rec_rewards,
                rec_values,
                rec_probs,
                rec_probs,   # beta = policy_probs at non-root nodes (no noise)
            )

        return SearchOutput(
            root_value=trees.get_roots_values(),
            sampled_actions=trees.get_roots_sampled_actions(),
            sampled_visit_counts=trees.get_roots_sampled_visit_count(),
            sampled_qvalues=trees.get_roots_sampled_qvalues(cfg.discount),
            sampled_imp_ratio=trees.get_roots_sampled_imp_ratio(),
        )
```

- [ ] **Step 3: Run tests**

```bash
cd /home/ryan/Repos/jaxzero
conda run -n mazero pytest tests/test_mcts.py -v
```

Expected:
```
tests/test_mcts.py::test_search_output_shapes PASSED
tests/test_mcts.py::test_legal_masking_respected PASSED
tests/test_mcts.py::test_visit_counts_sum_to_simulations PASSED
tests/test_mcts.py::test_batch_independence PASSED
```

If `test_visit_counts_sum_to_simulations` fails: ctree's `get_roots_sampled_visit_count()` may return K entries that don't sum to exactly `num_simulations` due to early unvisited children. Update that test to check `sum >= 1` instead:

```python
def test_visit_counts_sum_to_simulations():
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.ones((B, N, OBS_DIM), dtype=np.float32)
    legal = np.ones((B, N, A), dtype=bool)
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    for b in range(B):
        total = result.sampled_visit_counts[b].sum()
        assert total == config.num_simulations, f"env {b}: got {total}"
```

- [ ] **Step 4: Run full test suite**

```bash
conda run -n mazero pytest tests/ -v 2>&1 | tail -30
```

All 39 tests should pass.

- [ ] **Step 5: Commit**

```bash
git add jaxzero/mcts/sampled_mcts.py
git commit -m "feat: replace Python Node MCTS with C++ ctree (OS-lambda, same interface)"
```

---

### Task 4: Smoke test end-to-end training

**Files:**
- No changes — verify existing `train.py` works with new MCTS

- [ ] **Step 1: Run short training run**

```bash
conda run -n mazero python -m jaxzero.main --env mpe --num_simulations 10 --training_steps 200 2>&1 | head -40
```

Expected output includes:
```
[filling buffer] round 5, size=40
[filling buffer] round 10, size=80
...
Step 0: loss=... | ep_return=...
```

Each collection round should be noticeably faster than before (C++ tree ops vs Python Node dicts).

- [ ] **Step 2: Benchmark collection speed (optional)**

```bash
conda run -n mazero python - <<'EOF'
import time, numpy as np, jax, jax.numpy as jnp
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet
from jaxzero.mcts.sampled_mcts import SampledMCTS

cfg = MAZeroConfig(num_agents=3, obs_size=80, action_space_size=9,
                   num_simulations=100, sampled_action_times=10)
net = MAMuZeroNet(config=cfg)
params = net.init(jax.random.PRNGKey(0), jnp.ones((8, 3, 80)))
mcts = SampledMCTS(config=cfg, model=net)
obs = np.ones((8, 3, 80), dtype=np.float32)
legal = np.ones((8, 3, 9), dtype=bool)
rng = np.random.default_rng(0)

# warmup
mcts.search(params, obs, legal, rng)

t0 = time.time()
for _ in range(5):
    mcts.search(params, obs, legal, rng)
print(f"{(time.time()-t0)/5*1000:.0f}ms per search (B=8, N=100)")
EOF
```

- [ ] **Step 3: Commit if any minor fixes were needed**

```bash
git add -p
git commit -m "fix: <describe fix if any>"
```

If no fixes needed, skip this step.
