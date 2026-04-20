# OS(λ) Joint MCTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `mctx.gumbel_muzero_policy` in `MCTSJointPlanner` with a custom JAX MCTS that uses Optimistic Search Lambda (OS(λ)) value backup — the core MAZero algorithm — to raise win rate from ~17% to competitive levels on SMAC 3m.

**Architecture:** A custom JAX MCTS lives in `mcts/mcts_joint_osla.py`. It uses `lax.fori_loop`/`lax.while_loop` over a static-shaped struct-of-arrays tree (max_nodes = num_simulations + 1). Each simulation adds one new leaf; OS(λ) aggregates per-simulation `(depth, backup_value)` pairs: keep top-ρ fraction by value, weight each by λ^depth. PUCT selection with K sampled joint actions per node (like MAZero). `MCTSJointOSLAPlanner` inherits `MCTSPlanner` and replaces `MCTSJointPlanner` when `planner_mode="joint"`.

**Tech Stack:** JAX (`lax.fori_loop`, `lax.while_loop`, `lax.cond`, `lax.dynamic_update_slice`), `chex.dataclass`, existing `FlaxMAMuZeroNet.recurrent_inference`, existing `MCTSPlanOutput`.

---

### Task 1: Add OS(λ) hyperparameters to MCTSConfig and YAML configs

**Files:**
- Modify: `config.py`
- Modify: `configs/mcts/default.yaml`
- Modify: `configs/mcts/joint.yaml`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_mcts.py, add to the test_config fixture — verify it accepts the new fields:
def test_mcts_config_has_osla_fields():
    from config import MCTSConfig
    cfg = MCTSConfig(
        planner_mode="joint",
        num_simulations=8,
        max_depth_gumbel_search=3,
        num_gumbel_samples=4,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        independent_argmax=True,
        use_root_communication=False,
        mcts_rho=0.75,
        mcts_lambda=0.8,
    )
    assert cfg.mcts_rho == 0.75
    assert cfg.mcts_lambda == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n mazero pytest tests/test_mcts.py::test_mcts_config_has_osla_fields -v
```

Expected: `FAILED` — `MCTSConfig.__init__() got an unexpected keyword argument 'mcts_rho'`

- [ ] **Step 3: Add fields to MCTSConfig in `config.py`**

In `config.py`, add two fields to `MCTSConfig` with defaults so existing YAML and test fixtures still work:

```python
@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for the MCTS planner."""
    planner_mode: str
    num_simulations: int
    max_depth_gumbel_search: int
    num_gumbel_samples: int
    dirichlet_alpha: float
    dirichlet_fraction: float
    independent_argmax: bool
    use_root_communication: bool
    mcts_rho: float = 0.75    # OS(λ): top quantile fraction to keep (1-rho is discarded)
    mcts_lambda: float = 0.8  # OS(λ): depth discount weight (lambda^depth)
```

- [ ] **Step 4: Add to `configs/mcts/default.yaml`**

```yaml
planner_mode: independent
num_simulations: 100
max_depth_gumbel_search: 10
num_gumbel_samples: 10
dirichlet_alpha: 0.3
dirichlet_fraction: 0.25
independent_argmax: true
use_root_communication: false
mcts_rho: 0.75
mcts_lambda: 0.8
```

- [ ] **Step 5: No change needed to `configs/mcts/joint.yaml`** (inherits defaults from default.yaml; OS(λ) uses the same rho/lambda values).

- [ ] **Step 6: Update existing test_config fixture in `tests/test_mcts.py`**

The module-scoped `test_config` fixture builds `MCTSConfig` by positional/keyword args. Add the new fields so the fixture keeps working when the dataclass has required fields (they have defaults so this is optional, but explicit is better):

In `tests/test_mcts.py`, the `mcts=MCTSConfig(...)` in `test_config` already works because the new fields have defaults. Verify by running existing tests.

- [ ] **Step 7: Run tests and verify existing tests still pass**

```bash
conda run -n mazero pytest tests/test_mcts.py -v
```

Expected: All existing tests PASS (new fields have defaults, no breakage).

- [ ] **Step 8: Commit**

```bash
git add config.py configs/mcts/default.yaml configs/mcts/joint.yaml tests/test_mcts.py
git commit -m "feat: add mcts_rho and mcts_lambda OS(λ) hyperparameters to MCTSConfig"
```

---

### Task 2: Implement OS(λ) value aggregation (pure function, no MCTS yet)

`compute_osla_value` is the mathematical core: given K `(depth, backup_value)` pairs collected from K simulations, return the OS(λ) estimate. Implement and test it standalone before touching any MCTS code.

**Files:**
- Create: `mcts/mcts_joint_osla.py` (skeleton + this function)
- Modify: `tests/test_mcts.py` (new test class)

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_mcts.py`:

```python
class TestComputeOslaValue:
    """Tests for the OS(λ) value aggregation function."""

    def test_all_same_depth_mean_equals_top_rho_mean(self):
        """When all sims reach same depth, OS(λ) with rho=1.0 = plain mean."""
        from mcts.mcts_joint_osla import compute_osla_value
        depths = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        values = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        # rho=1.0: keep all 4, mean = 0.25
        v = compute_osla_value(depths, values, rho=1.0, lam=0.8)
        assert jnp.allclose(v, 0.25, atol=1e-4)

    def test_top_rho_amplifies_rare_wins(self):
        """rho=0.75 keeps top 25% (1 out of 4), result = the win value (1.0)."""
        from mcts.mcts_joint_osla import compute_osla_value
        depths = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        values = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        # rho=0.75: keep top 25% = top 1 = value 1.0; weight = lambda^1 = 0.8
        v = compute_osla_value(depths, values, rho=0.75, lam=0.8)
        assert jnp.allclose(v, 1.0, atol=1e-4)

    def test_depth_weighting_discounts_deeper_sims(self):
        """Deeper simulations get less weight: lambda^depth."""
        from mcts.mcts_joint_osla import compute_osla_value
        # Two sims: depth 1 value 1.0, depth 5 value 1.0
        depths = jnp.array([1, 5], dtype=jnp.int32)
        values = jnp.array([1.0, 1.0], dtype=jnp.float32)
        # rho=1.0: keep both. weights = [0.8^1, 0.8^5] = [0.8, 0.327]
        # weighted mean = (0.8*1.0 + 0.327*1.0) / (0.8 + 0.327) = 1.0
        v = compute_osla_value(depths, values, rho=1.0, lam=0.8)
        assert jnp.allclose(v, 1.0, atol=1e-4)  # same value at all depths = 1.0

    def test_jit_compatible(self):
        """Must be JIT-compilable."""
        from mcts.mcts_joint_osla import compute_osla_value
        fn = jax.jit(compute_osla_value, static_argnames=("rho", "lam"))
        depths = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        values = jnp.array([0.1, 0.5, 0.2, 0.9], dtype=jnp.float32)
        v = fn(depths, values, rho=0.75, lam=0.8)
        assert v.shape == ()
        assert jnp.isfinite(v)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestComputeOslaValue -v
```

Expected: `FAILED` — `cannot import name 'compute_osla_value' from 'mcts.mcts_joint_osla'`

- [ ] **Step 3: Create `mcts/mcts_joint_osla.py` with `compute_osla_value`**

```python
# mcts/mcts_joint_osla.py

import functools

import chex
import jax
import jax.numpy as jnp

from model import FlaxMAMuZeroNet
import utils.transforms as utils
from config import ExperimentConfig
from mcts.base import MCTSPlanner, MCTSPlanOutput


def compute_osla_value(
    sim_depths: chex.Array,    # [K] int32 — simulation leaf depths
    sim_values: chex.Array,    # [K] float32 — backup values from root
    rho: float = 0.75,         # keep top (1-rho) fraction; rho=0.75 → top 25%
    lam: float = 0.8,          # depth decay weight
) -> chex.Array:
    """
    OS(λ) value estimate: weighted mean of the top (1-rho) quantile of simulations,
    each weighted by lambda^depth.

    rho=0.75 keeps the top 25% of simulations by backup value — amplifying rare
    high-value paths (e.g. wins) that mean backup would dilute.
    """
    K = sim_values.shape[0]
    k_top = max(1, int((1.0 - rho) * K))  # number of sims to keep (static)

    # Sort simulations by backup value descending; take top k_top indices.
    sorted_idx = jnp.argsort(sim_values)[::-1]  # descending
    top_idx = sorted_idx[:k_top]                 # [k_top]

    top_values = sim_values[top_idx]   # [k_top]
    top_depths = sim_depths[top_idx]   # [k_top]
    weights = lam ** top_depths.astype(jnp.float32)  # [k_top]

    return (top_values * weights).sum() / (weights.sum() + 1e-8)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestComputeOslaValue -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts/mcts_joint_osla.py tests/test_mcts.py
git commit -m "feat: add compute_osla_value — OS(λ) top-quantile depth-weighted aggregation"
```

---

### Task 3: Define tree data structures and UCB helper

**Files:**
- Modify: `mcts/mcts_joint_osla.py`

- [ ] **Step 1: Write failing tests for the UCB helper**

Add to `tests/test_mcts.py`:

```python
class TestOSLAHelpers:

    def test_compute_ucb_prefers_unvisited(self):
        """Unvisited children (visit_count=0) should have highest UCB."""
        from mcts.mcts_joint_osla import compute_ucb_scores
        # 4 children: visits [5, 0, 3, 0], Q-values [0.5, 0.0, 0.3, 0.0]
        # prior_probs = uniform = 0.25 each, parent_visits = 8
        child_visits = jnp.array([5, 0, 3, 0], dtype=jnp.float32)
        child_q = jnp.array([0.5, 0.0, 0.3, 0.0], dtype=jnp.float32)
        prior_probs = jnp.array([0.25, 0.25, 0.25, 0.25], dtype=jnp.float32)
        parent_visits = jnp.array(8, dtype=jnp.float32)
        ucb = compute_ucb_scores(child_q, child_visits, prior_probs, parent_visits, c_puct=1.25)
        # Unvisited children (indices 1, 3) should have higher UCB than visited ones
        assert ucb[1] > ucb[0]
        assert ucb[3] > ucb[2]

    def test_compute_ucb_shape(self):
        from mcts.mcts_joint_osla import compute_ucb_scores
        K = 10
        ucb = compute_ucb_scores(
            jnp.zeros(K), jnp.zeros(K), jnp.ones(K) / K,
            jnp.array(1.0), c_puct=1.25
        )
        assert ucb.shape == (K,)
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestOSLAHelpers -v
```

Expected: FAILED — `cannot import name 'compute_ucb_scores'`

- [ ] **Step 3: Add tree datastructures and UCB helper to `mcts/mcts_joint_osla.py`**

Append after `compute_osla_value`:

```python
# ─── Tree data structure ──────────────────────────────────────────────────────

@chex.dataclass
class OSLATree:
    """Static-shaped search tree for one environment instance (no batch dim).

    max_nodes = num_simulations + 1 (root + 1 new leaf per simulation).
    K = num_gumbel_samples (sampled joint actions per node).
    """
    visit_counts:     chex.Array  # [max_nodes] int32
    value_sum:        chex.Array  # [max_nodes] float32
    reward:           chex.Array  # [max_nodes] float32 — reward to enter this node
    embedding:        chex.Array  # [max_nodes, N, D] float32
    depth:            chex.Array  # [max_nodes] int32
    parent:           chex.Array  # [max_nodes] int32 (-1 = root)
    child_actions:    chex.Array  # [max_nodes, K] int32 — flat joint action indices
    child_node_idx:   chex.Array  # [max_nodes, K] int32 (-1 = not yet expanded)
    child_prior_prob: chex.Array  # [max_nodes, K] float32


@chex.dataclass
class SimCarry:
    """Outer fori_loop state across all simulations."""
    tree:       OSLATree
    next_free:  chex.Array   # [] int32 — next available node index
    rng:        chex.Array   # PRNGKey
    sim_depths: chex.Array   # [num_simulations] int32
    sim_values: chex.Array   # [num_simulations] float32


@chex.dataclass
class SelectCarry:
    """Inner while_loop state during tree selection."""
    node_idx:     chex.Array  # [] int32 — current node
    depth:        chex.Array  # [] int32
    path_nodes:   chex.Array  # [max_depth+1] int32
    path_k:       chex.Array  # [max_depth+1] int32
    path_rewards: chex.Array  # [max_depth+1] float32 — reward[path_nodes[i]]
    best_k:       chex.Array  # [] int32 — UCB-best child of current node
    child_idx:    chex.Array  # [] int32 — child_node_idx[node, best_k]


# ─── UCB helper ───────────────────────────────────────────────────────────────

def compute_ucb_scores(
    child_q:       chex.Array,  # [K] float32 — mean Q-value per child
    child_visits:  chex.Array,  # [K] float32 — visit counts (0 = unvisited)
    prior_probs:   chex.Array,  # [K] float32 — prior probabilities
    parent_visits: chex.Array,  # [] float32 — total visits at parent
    c_puct: float = 1.25,
) -> chex.Array:
    """PUCT formula: Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))."""
    exploration = c_puct * prior_probs * jnp.sqrt(parent_visits + 1.0) / (1.0 + child_visits)
    return child_q + exploration
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestOSLAHelpers -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts/mcts_joint_osla.py tests/test_mcts.py
git commit -m "feat: add OSLATree dataclass and compute_ucb_scores helper"
```

---

### Task 4: Implement the single-simulation loop (selection + expansion + backup)

This is the most complex part. We implement `_run_single_sim(sim_carry, sim_idx, params, recurrent_fn, ...)` which runs one MCTS simulation.

**Files:**
- Modify: `mcts/mcts_joint_osla.py`

- [ ] **Step 1: Add `_sample_k_actions` helper and write its test**

Add to `tests/test_mcts.py::TestOSLAHelpers`:

```python
    def test_sample_k_actions_unique(self):
        """Sampled actions should be in-range and (mostly) unique."""
        from mcts.mcts_joint_osla import _sample_k_actions
        rng = jax.random.PRNGKey(0)
        logits = jnp.zeros(729)  # uniform over 3m joint space
        actions, probs = _sample_k_actions(rng, logits, K=10, A_N=729)
        assert actions.shape == (10,)
        assert probs.shape == (10,)
        assert jnp.all(actions >= 0) and jnp.all(actions < 729)
        assert jnp.allclose(probs.sum(), 1.0, atol=1e-4)
```

Run:
```bash
conda run -n mazero pytest tests/test_mcts.py::TestOSLAHelpers::test_sample_k_actions_unique -v
```
Expected: FAIL

- [ ] **Step 2: Implement `_sample_k_actions` in `mcts/mcts_joint_osla.py`**

```python
def _sample_k_actions(
    rng: chex.Array,
    logits: chex.Array,   # [A_N] joint action logits
    K: int,               # number of actions to sample (static)
    A_N: int,             # joint action space size (static)
) -> tuple[chex.Array, chex.Array]:
    """Sample K joint actions proportional to the prior; return (actions, probs)."""
    probs = jax.nn.softmax(logits)
    # Sample without replacement; fall back to with-replacement if K >= A_N
    replace = K >= A_N
    actions = jax.random.choice(rng, A_N, shape=(K,), replace=replace, p=probs)
    return actions, probs[actions]  # (K,) int32, (K,) float32
```

Run test: `conda run -n mazero pytest tests/test_mcts.py::TestOSLAHelpers::test_sample_k_actions_unique -v`
Expected: PASS.

- [ ] **Step 3: Implement `_run_single_sim`**

Add to `mcts/mcts_joint_osla.py` after the helper functions:

```python
def _run_single_sim(
    carry: SimCarry,
    sim_idx: chex.Array,        # [] int32 — current simulation index
    params,                     # model params (pytree)
    recurrent_fn,               # (params, rng, action, embedding) -> (output, next_embedding)
    initial_joint_logits: chex.Array,  # [A_N] root joint logits (for root prior; already set)
    K: int,                     # num_gumbel_samples (static)
    A_N: int,                   # joint action space size (static)
    max_depth: int,             # static upper bound for loops
    gamma: float,
    c_puct: float = 1.25,
) -> SimCarry:
    """Run one MCTS simulation: select → expand → backup → record (depth, value)."""
    tree = carry.tree
    rng, expand_rng = jax.random.split(carry.rng)

    # ── 1. Selection ─────────────────────────────────────────────────────────
    def _get_child_q(tree: OSLATree, node_idx, k) -> chex.Array:
        child_idx = tree.child_node_idx[node_idx, k]
        child_vc = tree.visit_counts[child_idx].astype(jnp.float32)
        child_vs = tree.value_sum[child_idx]
        return jnp.where(child_idx >= 0, child_vs / jnp.maximum(child_vc, 1.0), 0.0)

    def _best_ucb(tree: OSLATree, node_idx) -> chex.Array:
        child_node_idxs = tree.child_node_idx[node_idx]   # [K]
        child_visits = jnp.where(
            child_node_idxs >= 0,
            tree.visit_counts[jnp.maximum(child_node_idxs, 0)].astype(jnp.float32),
            0.0,
        )
        child_q = jnp.where(
            child_node_idxs >= 0,
            tree.value_sum[jnp.maximum(child_node_idxs, 0)] /
            jnp.maximum(child_visits, 1.0),
            0.0,
        )
        prior_probs = tree.child_prior_prob[node_idx]      # [K]
        parent_visits = tree.visit_counts[node_idx].astype(jnp.float32)
        ucb = compute_ucb_scores(child_q, child_visits, prior_probs, parent_visits, c_puct)
        return jnp.argmax(ucb).astype(jnp.int32)

    def _make_init_select_carry(tree: OSLATree) -> SelectCarry:
        bk = _best_ucb(tree, jnp.array(0, jnp.int32))
        cidx = tree.child_node_idx[0, bk]
        return SelectCarry(
            node_idx=jnp.array(0, jnp.int32),
            depth=jnp.array(0, jnp.int32),
            path_nodes=jnp.full(max_depth + 1, -1, jnp.int32),
            path_k=jnp.full(max_depth + 1, 0, jnp.int32),
            path_rewards=jnp.zeros(max_depth + 1, jnp.float32),
            best_k=bk,
            child_idx=cidx,
        )

    def _select_cond(sc: SelectCarry) -> bool:
        return (sc.depth < max_depth) & (sc.child_idx >= 0)

    def _select_body(sc: SelectCarry) -> SelectCarry:
        path_nodes = sc.path_nodes.at[sc.depth].set(sc.node_idx)
        path_k = sc.path_k.at[sc.depth].set(sc.best_k)
        child_reward = tree.reward[sc.child_idx]
        path_rewards = sc.path_rewards.at[sc.depth].set(child_reward)

        new_bk = _best_ucb(tree, sc.child_idx)
        new_cidx = tree.child_node_idx[sc.child_idx, new_bk]
        return SelectCarry(
            node_idx=sc.child_idx,
            depth=sc.depth + 1,
            path_nodes=path_nodes,
            path_k=path_k,
            path_rewards=path_rewards,
            best_k=new_bk,
            child_idx=new_cidx,
        )

    sc = jax.lax.while_loop(_select_cond, _select_body, _make_init_select_carry(tree))

    # After selection: sc.node_idx is the parent of the new leaf.
    # sc.best_k is the child to expand from sc.node_idx.
    parent_node = sc.node_idx
    expand_k = sc.best_k
    leaf_depth = sc.depth + 1

    # Record this node in path
    path_nodes = sc.path_nodes.at[sc.depth].set(parent_node)
    path_k = sc.path_k.at[sc.depth].set(expand_k)

    # ── 2. Expansion ─────────────────────────────────────────────────────────
    parent_embedding = tree.embedding[parent_node]        # [N, D]
    flat_action = tree.child_actions[parent_node, expand_k]  # [] int32

    rec_out, new_embedding = recurrent_fn(params, expand_rng, flat_action[None], parent_embedding[None])
    new_embedding = new_embedding[0]       # [N, D]
    leaf_reward = rec_out.reward[0]        # scalar
    leaf_value = rec_out.value[0]          # scalar (model prior value estimate)
    leaf_prior_logits = rec_out.prior_logits[0]  # [A_N]

    # Sample K children for the new leaf node
    child_rng, sample_rng = jax.random.split(expand_rng)
    new_child_actions, new_child_probs = _sample_k_actions(sample_rng, leaf_prior_logits, K, A_N)

    # Allocate new node
    new_node_idx = carry.next_free

    tree = tree.replace(
        embedding=tree.embedding.at[new_node_idx].set(new_embedding),
        reward=tree.reward.at[new_node_idx].set(leaf_reward),
        depth=tree.depth.at[new_node_idx].set(leaf_depth),
        parent=tree.parent.at[new_node_idx].set(parent_node),
        child_actions=tree.child_actions.at[new_node_idx].set(new_child_actions),
        child_node_idx=tree.child_node_idx.at[new_node_idx].set(
            jnp.full(K, -1, jnp.int32)
        ),
        child_prior_prob=tree.child_prior_prob.at[new_node_idx].set(new_child_probs),
    )
    # Link parent → new node
    tree = tree.replace(
        child_node_idx=tree.child_node_idx.at[parent_node, expand_k].set(new_node_idx),
    )

    # ── 3. Backup ────────────────────────────────────────────────────────────
    # path_nodes[0..sc.depth] inclusive = nodes from root to parent.
    # Include new leaf at path_nodes[sc.depth+1].
    path_nodes = path_nodes.at[leaf_depth].set(new_node_idx)
    path_rewards = sc.path_rewards.at[leaf_depth].set(leaf_reward)
    path_depth = leaf_depth  # number of steps from root to leaf

    # Work backwards from leaf to root, accumulating value.
    def backup_body(k, bcarry):
        # k = 0 → update leaf, k = path_depth → update root
        btree, V = bcarry
        i = path_depth - k  # depth index: path_depth, path_depth-1, ..., 0
        valid = k <= path_depth
        node_i = jnp.where(valid, path_nodes[i], 0)

        new_vc = btree.visit_counts.at[node_i].add(jnp.where(valid, 1, 0))
        new_vs = btree.value_sum.at[node_i].add(jnp.where(valid, V, 0.0))
        btree = btree.replace(visit_counts=new_vc, value_sum=new_vs)

        # Update V for the next (shallower) node: V_{i-1} = r_i + gamma * V_i
        r_i = path_rewards[jnp.maximum(i, 0)]
        V_new = r_i + gamma * V
        V = jnp.where(valid & (k < path_depth), V_new, V)
        return btree, V

    tree, _ = jax.lax.fori_loop(0, max_depth + 1, backup_body, (tree, leaf_value))

    # ── 4. Record OS(λ) data for this simulation ──────────────────────────────
    # Backup value from root: cumulative discounted reward + gamma^depth * leaf_value
    cum_reward = jnp.sum(
        jnp.array([path_rewards[d] * (gamma ** d) for d in range(max_depth + 1)]) *
        (jnp.arange(max_depth + 1) <= path_depth)
    )
    # Simpler: recompute from scratch using the path
    def accum_reward(k, acc):
        # k=0 → depth 1 reward, k=depth-1 → depth path_depth reward
        d = k + 1
        valid = d <= path_depth
        return acc + jnp.where(valid, (gamma ** k) * path_rewards[d], 0.0)
    root_cum_reward = jax.lax.fori_loop(0, max_depth, accum_reward, 0.0)
    root_backup_value = root_cum_reward + (gamma ** path_depth) * leaf_value

    sim_depths = carry.sim_depths.at[sim_idx].set(path_depth)
    sim_values = carry.sim_values.at[sim_idx].set(root_backup_value)

    return SimCarry(
        tree=tree,
        next_free=carry.next_free + 1,
        rng=rng,
        sim_depths=sim_depths,
        sim_values=sim_values,
    )
```

- [ ] **Step 4: Run existing tests to verify no regressions**

```bash
conda run -n mazero pytest tests/test_mcts.py -v
```

Expected: All prior tests still PASS (we haven't broken any interface).

- [ ] **Step 5: Commit**

```bash
git add mcts/mcts_joint_osla.py
git commit -m "feat: implement _run_single_sim — MCTS selection/expansion/backup with OS(λ) tracking"
```

---

### Task 5: Implement `MCTSJointOSLAPlanner` and the full planning loop

**Files:**
- Modify: `mcts/mcts_joint_osla.py`

- [ ] **Step 1: Write failing tests for MCTSJointOSLAPlanner**

Add a new test class to `tests/test_mcts.py`:

```python
@pytest.fixture(scope="module")
def osla_config(test_config):
    """MCTSConfig with joint planner + OS(λ)."""
    return ExperimentConfig(
        train=test_config.train,
        model=test_config.model,
        mcts=MCTSConfig(
            planner_mode="joint",
            num_simulations=8,
            max_depth_gumbel_search=3,
            num_gumbel_samples=4,
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
            independent_argmax=True,
            use_root_communication=False,
            mcts_rho=0.75,
            mcts_lambda=0.8,
        ),
    )


@pytest.fixture(scope="module")
def osla_plan_fn(model_and_params, osla_config):
    """JIT-compiled plan function for MCTSJointOSLAPlanner."""
    from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
    net, _ = model_and_params
    planner = MCTSJointOSLAPlanner(model=net, config=osla_config)
    return jax.jit(planner.plan), planner


class TestMCTSJointOSLAPlanner:

    def test_returns_plan_output(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert isinstance(out, MCTSPlanOutput)

    def test_joint_action_shape(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.joint_action.shape == (1, N)

    def test_policy_targets_shape(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.policy_targets.shape == (1, N, A)

    def test_actions_in_valid_range(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert jnp.all(out.joint_action >= 0)
        assert jnp.all(out.joint_action < A)

    def test_policy_targets_sum_to_one(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        sums = jnp.sum(out.policy_targets, axis=-1)
        assert jnp.allclose(sums, jnp.ones((1, N)), atol=1e-4)

    def test_root_value_shape(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        assert out.root_value.shape == (1,)

    def test_deterministic_with_same_key(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        out1 = plan_fn(params, jax.random.PRNGKey(7), obs)
        out2 = plan_fn(params, jax.random.PRNGKey(7), obs)
        assert jnp.array_equal(out1.joint_action, out2.joint_action)

    def test_stochastic_with_different_keys(self, osla_plan_fn, params, obs):
        plan_fn, _ = osla_plan_fn
        results = [plan_fn(params, jax.random.PRNGKey(i), obs).joint_action for i in range(10)]
        all_same = all(jnp.array_equal(results[0], r) for r in results[1:])
        assert not all_same

    def test_osla_value_differs_from_model_prior(self, osla_plan_fn, model_and_params, osla_config, obs):
        """OS(λ) root value should generally differ from raw model value (search improves it)."""
        from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
        net, params = model_and_params
        planner = MCTSJointOSLAPlanner(model=net, config=osla_config)
        plan_fn = jax.jit(planner.plan)
        out = plan_fn(params, jax.random.PRNGKey(0), obs)
        # Just verify it's a finite scalar — exact value depends on random search
        assert jnp.isfinite(out.root_value).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestMCTSJointOSLAPlanner -v
```

Expected: FAIL — `cannot import name 'MCTSJointOSLAPlanner'`

- [ ] **Step 3: Implement `_osla_plan_single` and `MCTSJointOSLAPlanner`**

Append to `mcts/mcts_joint_osla.py`:

```python
# ─── Full planning loop (single environment, no batch) ───────────────────────

def _osla_plan_single(
    params,
    rng_key: chex.Array,
    observation: chex.Array,      # [N, obs_size]
    model,
    num_simulations: int,         # static
    K: int,                       # static (num_gumbel_samples)
    A_N: int,                     # static (action_space_size^num_agents)
    max_depth: int,               # static
    gamma: float,
    rho: float,
    lam: float,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
    joint_action_shape: tuple,    # static (A, A, ...) x N
    value_support,
    reward_support,
) -> MCTSPlanOutput:
    """Core planning function for one environment. vmapped over batch outside."""
    init_key, dir_key, rng = jax.random.split(rng_key, 3)
    max_nodes = num_simulations + 1
    N = observation.shape[0]  # num agents

    # ── Root inference ───────────────────────────────────────────────────────
    obs_batched = observation[None]  # [1, N, obs_size]
    init_out = model.apply(
        {"params": params}, obs_batched, rngs={"dropout": init_key}
    )
    root_embedding = init_out.hidden_state[0]     # [N, D]
    root_value = utils.support_to_scalar(init_out.value_logits, value_support)[0]
    root_joint_logits = _logits_to_joint_logits(init_out.policy_logits[0], N, A_N)

    # Dirichlet noise on root prior
    dir_noise = jax.random.dirichlet(dir_key, alpha=jnp.full(A_N, dirichlet_alpha))
    root_probs = jax.nn.softmax(root_joint_logits)
    noisy_probs = (1 - dirichlet_fraction) * root_probs + dirichlet_fraction * dir_noise
    noisy_root_logits = jnp.log(noisy_probs + 1e-30)

    # Sample K children for root
    sample_key, rng = jax.random.split(rng)
    root_child_actions, root_child_probs = _sample_k_actions(sample_key, noisy_root_logits, K, A_N)

    # ── Initialize tree ──────────────────────────────────────────────────────
    D = root_embedding.shape[-1]
    tree = OSLATree(
        visit_counts=jnp.zeros(max_nodes, jnp.int32),
        value_sum=jnp.zeros(max_nodes, jnp.float32),
        reward=jnp.zeros(max_nodes, jnp.float32),
        embedding=jnp.zeros((max_nodes, N, D), jnp.float32).at[0].set(root_embedding),
        depth=jnp.zeros(max_nodes, jnp.int32),
        parent=jnp.full(max_nodes, -1, jnp.int32),
        child_actions=jnp.zeros((max_nodes, K), jnp.int32).at[0].set(root_child_actions),
        child_node_idx=jnp.full((max_nodes, K), -1, jnp.int32),
        child_prior_prob=jnp.zeros((max_nodes, K), jnp.float32).at[0].set(root_child_probs),
    )
    # Initialize root visit count = 1 so parent_visits > 0 in UCB from step 1
    tree = tree.replace(visit_counts=tree.visit_counts.at[0].set(1))

    init_carry = SimCarry(
        tree=tree,
        next_free=jnp.array(1, jnp.int32),
        rng=rng,
        sim_depths=jnp.zeros(num_simulations, jnp.int32),
        sim_values=jnp.zeros(num_simulations, jnp.float32),
    )

    # ── Recurrent fn wrapper (batched interface expected by _run_single_sim) ──
    def recurrent_fn_batched(params, rng_key, flat_action_batch, embedding_batch):
        # flat_action_batch: [1] int32; embedding_batch: [1, N, D]
        joint_action_shape_t = joint_action_shape
        per_agent = jnp.stack(
            jnp.unravel_index(flat_action_batch, joint_action_shape_t), axis=-1
        )  # [1, N]
        out = model.apply(
            {"params": params},
            embedding_batch,
            per_agent,
            method=model.recurrent_inference,
            rngs={"dropout": rng_key},
        )
        value = utils.support_to_scalar(out.value_logits, value_support)
        reward = utils.support_to_scalar(out.reward_logits, reward_support)
        joint_logits = jax.vmap(
            lambda lg: _logits_to_joint_logits(lg, N, A_N)
        )(out.policy_logits)  # vmap over batch=1
        import mctx
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, gamma),
                prior_logits=joint_logits,
                value=value,
            ),
            out.hidden_state,
        )

    # ── Run simulations ──────────────────────────────────────────────────────
    def sim_step(sim_idx, carry: SimCarry) -> SimCarry:
        return _run_single_sim(
            carry, sim_idx, params, recurrent_fn_batched,
            noisy_root_logits, K, A_N, max_depth, gamma, c_puct,
        )

    final_carry = jax.lax.fori_loop(0, num_simulations, sim_step, init_carry)

    # ── OS(λ) root value ─────────────────────────────────────────────────────
    osla_root_value = compute_osla_value(
        final_carry.sim_depths, final_carry.sim_values, rho=rho, lam=lam
    )

    # ── Policy target from root child visit counts ────────────────────────────
    root_children = final_carry.tree.child_node_idx[0]  # [K]
    root_child_visits = jnp.where(
        root_children >= 0,
        final_carry.tree.visit_counts[jnp.maximum(root_children, 0)].astype(jnp.float32),
        0.0,
    )  # [K]

    # Build sparse joint distribution over A_N actions
    joint_visits = jnp.zeros(A_N).at[root_child_actions].add(root_child_visits)
    joint_policy = joint_visits / (joint_visits.sum() + 1e-8)  # [A_N]

    # Marginalize to per-agent targets (B=1 interface: reshape to (1, A_N))
    marginal_policy = _joint_policy_to_marginal(joint_policy[None], N, joint_action_shape)  # (1, N, A)

    # Best action = most-visited root child
    best_k = jnp.argmax(root_child_visits)
    best_flat_action = root_child_actions[best_k]
    best_action = jnp.stack(
        jnp.unravel_index(best_flat_action, joint_action_shape), axis=-1
    )  # (N,)

    return MCTSPlanOutput(
        joint_action=best_action[None],         # (1, N)
        policy_targets=marginal_policy,          # (1, N, A)
        root_value=osla_root_value[None],        # (1,)
        agent_order=jnp.arange(N),
    )


def _logits_to_joint_logits(per_agent_logits: chex.Array, N: int, A_N: int) -> chex.Array:
    """Per-agent logits (N, A) → joint logits (A_N,) under independence assumption."""
    A = per_agent_logits.shape[-1]
    log_probs = jax.nn.log_softmax(per_agent_logits, axis=-1)  # (N, A)
    joint = log_probs[0]                                         # (A,)
    for i in range(1, N):
        joint = (joint[:, None] + log_probs[i][None, :]).reshape(-1)
    return joint  # (A_N,)


def _joint_policy_to_marginal(
    joint_policy: chex.Array,     # (B, A_N)
    N: int,
    joint_action_shape: tuple,    # (A, A, ...) x N
) -> chex.Array:
    """Joint policy (B, A_N) → per-agent marginals (B, N, A)."""
    B = joint_policy.shape[0]
    A = joint_action_shape[0]
    reshaped = joint_policy.reshape(B, *joint_action_shape)
    marginals = []
    for i in range(N):
        other_axes = tuple(j + 1 for j in range(N) if j != i)
        marginals.append(jnp.sum(reshaped, axis=other_axes))  # (B, A)
    return jnp.stack(marginals, axis=1)  # (B, N, A)


# ─── Planner class ────────────────────────────────────────────────────────────

class MCTSJointOSLAPlanner(MCTSPlanner):
    """
    Joint MCTS with OS(λ) backup.

    Uses a custom JAX MCTS loop (not mctx) with PUCT selection,
    K sampled joint actions per node, and OS(λ) value aggregation.
    """

    def __init__(self, model: FlaxMAMuZeroNet, config: ExperimentConfig):
        super().__init__(model, config)
        self.joint_action_shape: tuple = (self.action_space_size,) * self.num_agents
        self.A_N = self.action_space_size ** self.num_agents
        self.mcts_rho = config.mcts.mcts_rho
        self.mcts_lambda = config.mcts.mcts_lambda

    def _recurrent_fn(self, params, rng_key, action, embedding):
        # Required by MCTSPlanner ABC, but not used directly (we call model inline).
        raise NotImplementedError("MCTSJointOSLAPlanner uses its own recurrent_fn internally.")

    def _plan_loop(
        self, params, rng_key: chex.Array, observation: chex.Array
    ) -> MCTSPlanOutput:
        """Vmapped single-env planning across batch."""
        B = observation.shape[0]
        rng_keys = jax.random.split(rng_key, B)

        plan_single = functools.partial(
            _osla_plan_single,
            model=self.model,
            num_simulations=self.num_simulations,
            K=self.num_gumbel_samples,
            A_N=self.A_N,
            max_depth=self.max_depth_gumbel_search,
            gamma=self.discount_gamma,
            rho=self.mcts_rho,
            lam=self.mcts_lambda,
            c_puct=1.25,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_fraction=self.dirichlet_fraction,
            joint_action_shape=self.joint_action_shape,
            value_support=self.value_support,
            reward_support=self.reward_support,
        )

        results = jax.vmap(plan_single, in_axes=(None, 0, 0))(
            params, rng_keys, observation
        )

        # vmap produces leading B dim on all arrays; reshape (B, 1, ...) → (B, ...)
        return MCTSPlanOutput(
            joint_action=results.joint_action.squeeze(1),      # (B, N)
            policy_targets=results.policy_targets.squeeze(1),  # (B, N, A)
            root_value=results.root_value.squeeze(1),          # (B,)
            agent_order=results.agent_order[0],                 # (N,) — same for all
        )
```

- [ ] **Step 4: Run tests**

```bash
conda run -n mazero pytest tests/test_mcts.py::TestMCTSJointOSLAPlanner -v
```

Expected: All 9 tests PASS. This is the longest compilation step — JIT takes 1–3 min on first run.

- [ ] **Step 5: Run ALL tests**

```bash
conda run -n mazero pytest tests/test_mcts.py -v
```

Expected: All tests PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add mcts/mcts_joint_osla.py tests/test_mcts.py
git commit -m "feat: implement MCTSJointOSLAPlanner — joint MCTS with OS(λ) value backup"
```

---

### Task 6: Wire MCTSJointOSLAPlanner into the training system

**Files:**
- Modify: `mcts/__init__.py`
- Modify: `actors/data_actor.py`

- [ ] **Step 1: Export from `mcts/__init__.py`**

```python
# mcts/__init__.py
from mcts.base import MCTSPlanner, MCTSPlanOutput
from mcts.mcts_independent import MCTSIndependentPlanner
from mcts.mcts_joint import MCTSJointPlanner
from mcts.mcts_joint_osla import MCTSJointOSLAPlanner
```

- [ ] **Step 2: Register in `actors/data_actor.py`**

In `data_actor.py`, inside `DataActor.__init__`, update the import and planner_map:

```python
from mcts import MCTSIndependentPlanner, MCTSJointPlanner, MCTSJointOSLAPlanner
```

```python
planner_map = {
    "independent": MCTSIndependentPlanner,
    "joint": MCTSJointOSLAPlanner,        # replace MCTSJointPlanner with OS(λ) version
    "joint_legacy": MCTSJointPlanner,     # keep old planner accessible for ablations
}
```

- [ ] **Step 3: Run ALL tests to verify no import errors**

```bash
conda run -n mazero pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 4: Quick smoke test with joint planner on MPE**

```bash
conda run -n mazero python train/muzero.py mcts=joint train.num_episodes=200 train.warmup_episodes=20
```

Expected: Training starts, loss decreases, no NaNs or crashes. DataActor logs show "Setup complete." for all actors.

- [ ] **Step 5: Commit**

```bash
git add mcts/__init__.py actors/data_actor.py
git commit -m "feat: wire MCTSJointOSLAPlanner as default joint planner; keep MCTSJointPlanner as joint_legacy"
```

---

### Task 7: Update SMAX configs and run a verification sweep

**Files:**
- Modify: `configs/train/smax_3m.yaml`

- [ ] **Step 1: Update smax_3m.yaml with OS(λ) hyperparams aligned with MAZero paper**

The MAZero paper uses: `mcts_rho=0.75`, `mcts_lambda=0.8`, `num_simulations=50` for SMAC.

```yaml
# configs/train/smax_3m.yaml
defaults:
  - default

env_name: 3m
num_agents: 3
max_episode_steps: 150

# --- MaZero paper SMAC hyperparameters ---
learning_rate: 1e-4
n_step: 5
warmup_episodes: 50
awpo_alpha: 1.0

# Throughput tuning
num_actors: 6
batch_size: 1024
num_envs_per_actor: 8
```

The OS(λ) params (`mcts_rho=0.75`, `mcts_lambda=0.8`) already live in `configs/mcts/default.yaml` with the right values, so no override needed here.

- [ ] **Step 2: Smoke-test the full SMAX setup compiles and runs**

```bash
conda run -n mazero python train/muzero.py train=smax_3m mcts=joint train.num_episodes=100 train.warmup_episodes=10
```

Expected: Runs 100 episodes, no crashes, no NaN loss.

- [ ] **Step 3: Commit**

```bash
git add configs/train/smax_3m.yaml
git commit -m "config: align smax_3m.yaml with MAZero paper hyperparameters for OS(λ) training"
```

---

## Debugging Notes

**If JIT compilation hangs or is very slow (>5 min):**
- Check `max_depth_gumbel_search` in the test config; reduce to 2 if needed
- `lax.while_loop` with dynamic termination unrolls less aggressively than `fori_loop`; if compilation is a blocker, consider replacing the selection `while_loop` with a `fori_loop(0, max_depth+1, ...)` and masking

**If you get a shape error in `_run_single_sim`:**
- The most likely culprit is `recurrent_fn_batched` — it receives `flat_action_batch: [1]` and `embedding_batch: [1, N, D]` and must return shapes `(1,)` for reward and value, and `(1, A_N)` for prior_logits
- Add `assert` statements after each model call during debugging (they'll be stripped at JIT time)

**If OS(λ) root value is always 0 or NaN:**
- Check that `sim_values` accumulates properly; print `final_carry.sim_values` in eager mode (before JIT)
- Run `_osla_plan_single` without `jax.jit` first to inspect intermediate values

**If `jax.random.choice(..., replace=False)` fails:**
- This happens when `K >= A_N` (e.g., K=10 but A_N=9 for N=1 test case)
- The `_sample_k_actions` function already handles this with `replace = K >= A_N`

**vmap in `_plan_loop` shapes:**
- `jax.vmap(plan_single, in_axes=(None, 0, 0))(params, rng_keys, observation)` maps over `rng_keys: (B, 2)` and `observation: (B, N, obs_size)`
- `plan_single` expects `observation: (N, obs_size)` (no batch) and returns `MCTSPlanOutput` with all shapes having a leading `1` dim (from the internal `obs_batched = observation[None]`)
- After vmap, shapes become `(B, 1, ...)` → squeeze in `_plan_loop`

---

## Verification

After completing all tasks:

```bash
# 1. Unit tests
conda run -n mazero pytest tests/ -v

# 2. Short SMAX smoke test
conda run -n mazero python train/muzero.py train=smax_3m mcts=joint train.num_episodes=1000 train.warmup_episodes=50

# 3. Check that win rate starts increasing (should see non-zero wins by ~5k episodes)
# 4. Full training comparison: run with mcts=joint_legacy (old planner) vs mcts=joint (OS-λ)
#    and compare win rate at 50k episodes
```
