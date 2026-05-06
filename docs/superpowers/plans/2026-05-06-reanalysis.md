# Reanalysis Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement actual MCTS reanalysis in `ReanalyzeWorker.make_batch` so policy targets (visit counts, Q-values, sampled actions) are recomputed with current target params each training step, matching MAZero's core mechanism.

**Architecture:** `make_batch` currently ignores `params` and reads stale stored game data for all targets. We add a reanalysis pass: batch all `re_num × (U+1)` positions into a single MCTS call, unpack and pad results, then use fresh stats for policy targets while keeping stored obs/actions/rewards/value-bootstrap unchanged. Also fix missing `lax.stop_gradient` on the AWPO baseline.

**Tech Stack:** JAX, Flax, NumPy, jaxzero ctree (cytree), pytest

---

## File Map

| File | Change |
|------|--------|
| `jaxzero/reanalyze.py` | Core change: reanalysis in `make_batch` |
| `jaxzero/train.py` | Fix stop_gradient on AWPO baseline; add reanalysis JIT warmup |
| `tests/test_reanalyze.py` | Add reanalysis behavior tests |

---

### Task 1: Fix stop_gradient on AWPO baseline

The advantage `Q - V_net` in `awpo_sharp_loss` uses the live model's `value_logits` as the baseline. Without `stop_gradient`, gradients from the policy loss flow back through `V_net` into the value parameters, creating unintended coupling between the policy and value objectives.

**Files:**
- Modify: `jaxzero/train.py:104` and `jaxzero/train.py:132`

- [ ] **Step 1: Write failing test**

Add to `tests/test_reanalyze.py` (or `tests/test_train.py`):

```python
def test_policy_loss_no_gradient_through_baseline():
    """Policy gradient must not flow through V_net baseline into value params."""
    import jax
    import jax.numpy as jnp
    from jaxzero.train import make_update_fn
    from jaxzero.model.networks import MAMuZeroNet

    config = MAZeroConfig(
        num_agents=N, obs_size=OBS_DIM, action_space_size=A,
        batch_size=B, unroll_steps=2, td_steps=2,
        use_reanalyze=False, num_simulations=5, sampled_action_times=K,
        min_replay_size=5, value_loss_coeff=0.0,  # zero out value loss
        reward_loss_coeff=0.0, consistency_coeff=0.0,
    )
    net = MAMuZeroNet(config=config)
    params = net.init(jax.random.PRNGKey(0), jnp.ones((1, N, OBS_DIM)))

    worker = ReanalyzeWorker(config=config, model=net)
    ctx = make_buffer_ctx(config)
    batch = worker.make_batch(ctx, params)

    update_fn = make_update_fn(net, config)
    # With only policy loss active, grads to value MLP output should be zero
    # if stop_gradient is correctly placed on V_net baseline.
    _, grads, _, _ = update_fn(params, batch)
    value_out_grad = jax.tree_util.tree_leaves(grads['params']['prediction_net']['value_mlp']['output']['kernel'])
    # Gradient should be zero or near-zero since only policy_loss is active
    # and it must not flow through the baseline
    assert np.allclose(value_out_grad[0], 0.0, atol=1e-6), (
        f"Policy loss gradient leaked into value_mlp output: max={np.abs(value_out_grad[0]).max()}"
    )
```

Run: `python -m pytest tests/test_reanalyze.py::test_policy_loss_no_gradient_through_baseline -v`

Expected: FAIL (gradient IS leaking currently).

- [ ] **Step 2: Add stop_gradient in train.py**

In `jaxzero/train.py`, change lines 104 and 132:

```python
# Line 104 — was:
v_pred_0 = phi_inv(out0.value_logits, S_v)  # (B,)
# Change to:
v_pred_0 = phi_inv(lax.stop_gradient(out0.value_logits), S_v)  # (B,)
```

```python
# Line 132 — was:
v_pred_k = phi_inv(out_k.value_logits, S_v)  # (B,)
# Change to:
v_pred_k = phi_inv(lax.stop_gradient(out_k.value_logits), S_v)  # (B,)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_reanalyze.py::test_policy_loss_no_gradient_through_baseline -v`

Expected: PASS.

- [ ] **Step 4: Run full test suite to check no regressions**

Run: `python -m pytest tests/ -v -x`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add jaxzero/train.py tests/test_reanalyze.py
git commit -m "fix: stop_gradient on AWPO baseline to prevent policy→value gradient leakage"
```

---

### Task 2: Add `_pad_to_k` helper in reanalyze.py

Reanalysis and the existing `_store_search_stats` in `train.py` both need to pad ctree outputs to fixed K and normalize visit counts. Extract a shared helper in `reanalyze.py` to avoid duplicating this logic.

**Files:**
- Modify: `jaxzero/reanalyze.py` (add helper at module level)

- [ ] **Step 1: Write failing test**

Add to `tests/test_reanalyze.py`:

```python
from jaxzero.reanalyze import _pad_to_k

def test_pad_to_k_pads_shorter():
    visits = np.array([10.0, 5.0, 3.0], dtype=np.float32)
    actions = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=np.int32)  # (3, N=3)
    qvals = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    K, N_agents = 5, 3

    pol, sa, qv, mask = _pad_to_k(visits, actions, qvals, K, N_agents)

    assert pol.shape == (K,)
    assert sa.shape == (K, N_agents)
    assert qv.shape == (K,)
    assert mask.shape == (K,)
    assert mask.dtype == bool
    # first 3 valid, last 2 padded
    assert mask[:3].all() and not mask[3:].any()
    # visit counts normalized to sum=1 over valid entries
    np.testing.assert_allclose(pol[:3].sum(), 1.0, atol=1e-5)
    # padded entries are zero
    assert pol[3:].sum() == 0.0
    assert sa[3:].sum() == 0


def test_pad_to_k_exact_k():
    visits = np.array([4.0, 3.0, 2.0, 1.0, 0.5], dtype=np.float32)
    actions = np.zeros((5, 2), dtype=np.int32)
    qvals = np.zeros(5, dtype=np.float32)
    K, N_agents = 5, 2

    pol, sa, qv, mask = _pad_to_k(visits, actions, qvals, K, N_agents)

    assert mask.all()
    np.testing.assert_allclose(pol.sum(), 1.0, atol=1e-5)
```

Run: `python -m pytest tests/test_reanalyze.py::test_pad_to_k_pads_shorter tests/test_reanalyze.py::test_pad_to_k_exact_k -v`

Expected: FAIL (`_pad_to_k` not defined).

- [ ] **Step 2: Add `_pad_to_k` to reanalyze.py**

Add at the top of `jaxzero/reanalyze.py`, after imports:

```python
def _pad_to_k(
    visits: np.ndarray,   # (K_actual,)
    actions: np.ndarray,  # (K_actual, N)
    qvals: np.ndarray,    # (K_actual,)
    K: int,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad ctree sampled-action outputs to fixed width K. Returns (pol, sa, qv, mask)."""
    K_actual = len(visits)
    pad = K - K_actual
    if pad > 0:
        visits = np.concatenate([visits, np.zeros(pad, dtype=np.float32)])
        actions = np.concatenate([actions, np.zeros((pad, N), dtype=np.int32)], axis=0)
        qvals = np.concatenate([qvals, np.zeros(pad, dtype=np.float32)])
        mask = np.concatenate([np.ones(K_actual, dtype=bool), np.zeros(pad, dtype=bool)])
    else:
        mask = np.ones(K, dtype=bool)
    total = visits[:K_actual].sum()
    pol = visits.copy()
    pol[:K_actual] = pol[:K_actual] / (total + 1e-8)
    return pol, actions, qvals, mask
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_reanalyze.py::test_pad_to_k_pads_shorter tests/test_reanalyze.py::test_pad_to_k_exact_k -v`

Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add jaxzero/reanalyze.py tests/test_reanalyze.py
git commit -m "feat: add _pad_to_k helper in reanalyze for ctree output padding"
```

---

### Task 3: Implement reanalysis in `ReanalyzeWorker.make_batch`

The core fix: run MCTS with current `params` for reanalyzed positions, replace stored policy targets with fresh ones.

**Files:**
- Modify: `jaxzero/reanalyze.py`

- [ ] **Step 1: Write failing test for reanalysis behavior**

Add to `tests/test_reanalyze.py`:

```python
from unittest.mock import patch
from jaxzero.mcts.sampled_mcts import SearchOutput

def _make_mock_search_result(B_flat, K, N):
    """Returns a SearchOutput where each position has K uniform visits to action 0."""
    return SearchOutput(
        root_value=np.zeros(B_flat, dtype=np.float32),
        sampled_actions=[np.zeros((K, N), dtype=np.int32) for _ in range(B_flat)],
        sampled_visit_counts=[np.ones(K, dtype=np.float32) for _ in range(B_flat)],
        sampled_qvalues=[np.full(K, 99.0, dtype=np.float32) for _ in range(B_flat)],
        sampled_imp_ratio=[np.ones(K, dtype=np.float32) for _ in range(B_flat)],
    )


def test_reanalysis_replaces_stored_policy_targets():
    """When use_reanalyze=True, make_batch must use fresh MCTS targets, not stored ones."""
    config = make_config(use_reanalyze=True)
    net = MAMuZeroNet(config=config)
    params = net.init(jax.random.PRNGKey(0), np.ones((1, N, OBS_DIM), dtype=np.float32))

    worker = ReanalyzeWorker(config=config, model=net)
    assert worker._mcts is not None

    U = config.unroll_steps
    B_flat = B * (U + 1)  # all positions in one MCTS call

    mock_result = _make_mock_search_result(B_flat, K, N)

    with patch.object(worker._mcts, 'search', return_value=mock_result) as mock_search:
        ctx = make_buffer_ctx(config)
        batch = worker.make_batch(ctx, params)

    # search must be called exactly once with all B*(U+1) positions
    mock_search.assert_called_once()
    call_obs = mock_search.call_args[0][1]   # second positional arg = obs
    assert call_obs.shape == (B_flat, N, OBS_DIM), (
        f"Expected obs shape ({B_flat}, {N}, {OBS_DIM}), got {call_obs.shape}"
    )

    # Q-values must be 99.0 (from mock), not stored random values
    assert np.all(batch.target_qvalues == 99.0), (
        f"Reanalyzed Q-values should be 99.0 but got: {batch.target_qvalues[:2, 0, :3]}"
    )

    # Visit counts must be uniform 1/K (mock returns ones, normalized)
    expected_pol = 1.0 / K
    np.testing.assert_allclose(batch.target_policies, expected_pol, atol=1e-5)


def test_no_reanalysis_uses_stored_targets():
    """When use_reanalyze=False, make_batch must NOT call MCTS."""
    config = make_config(use_reanalyze=False)
    net = MAMuZeroNet(config=config)
    params = net.init(jax.random.PRNGKey(0), np.ones((1, N, OBS_DIM), dtype=np.float32))

    worker = ReanalyzeWorker(config=config, model=net)
    assert worker._mcts is None  # no MCTS when use_reanalyze=False

    ctx = make_buffer_ctx(config)
    # In make_game(), all qvalues are random; stored stats should appear unchanged
    games, positions, _, _ = ctx
    stored_qvals_0 = games[0].sampled_qvalues[int(positions[0])]

    batch = worker.make_batch(ctx, params)

    np.testing.assert_array_equal(
        batch.target_qvalues[0, 0],
        stored_qvals_0,
    )
```

Run: `python -m pytest tests/test_reanalyze.py::test_reanalysis_replaces_stored_policy_targets tests/test_reanalyze.py::test_no_reanalysis_uses_stored_targets -v`

Expected: both FAIL (reanalysis not implemented).

- [ ] **Step 2: Implement reanalysis in `make_batch`**

Replace the full `make_batch` method in `jaxzero/reanalyze.py`:

```python
def make_batch(self, buffer_context, params: Any) -> BatchData:
    games, positions, indices, weights = buffer_context
    B = len(games)
    U = self.config.unroll_steps
    K = self.config.sampled_action_times
    N = self.config.num_agents

    # How many of the B games get reanalyzed policy targets
    re_num = int(np.ceil(B * self.config.revisit_policy_search_rate))
    if self._mcts is None:
        re_num = 0

    # --- Phase 1: Run MCTS reanalysis for re_num * (U+1) positions ---
    fresh_policies = None   # (re_num, U+1, K)
    fresh_qvalues = None    # (re_num, U+1, K)
    fresh_masks = None      # (re_num, U+1, K)
    fresh_sa = None         # (re_num, U+1, K, N)

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

    # --- Phase 2: Build batch, using fresh or stored stats ---
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
            pol_b = fresh_policies[b]    # (U+1, K) fresh visit counts
            qv_b = fresh_qvalues[b]      # (U+1, K) fresh Q-values
            mask_b = fresh_masks[b]      # (U+1, K) fresh masks
            sa_b = fresh_sa[b]           # (U+1, K, N) fresh sampled actions
        else:
            sa_b = np.stack([
                game.sampled_actions[min(pos + k, T - 1)]
                for k in range(U + 1)
            ])  # (U+1, K, N)

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
```

- [ ] **Step 3: Run reanalysis tests to verify they pass**

Run: `python -m pytest tests/test_reanalyze.py::test_reanalysis_replaces_stored_policy_targets tests/test_reanalyze.py::test_no_reanalysis_uses_stored_targets -v`

Expected: both PASS.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v -x`

Expected: all tests pass. If `test_batch_shapes_no_reanalyze` fails, verify `use_reanalyze=False` path still uses stored data.

- [ ] **Step 5: Commit**

```bash
git add jaxzero/reanalyze.py tests/test_reanalyze.py
git commit -m "feat: implement MCTS reanalysis in ReanalyzeWorker.make_batch"
```

---

### Task 4: Add reanalysis JIT warmup in train.py

`SampledMCTS.search` JIT-compiles on first call for a given batch size. Reanalysis uses batch size `B*(U+1)` (e.g. 256*6=1536) which differs from the collection batch size. First-call compilation mid-training causes a stall. Warmup it before the training loop.

**Files:**
- Modify: `jaxzero/train.py` (inside the `train()` function, after the existing warmup)

- [ ] **Step 1: Add reanalysis warmup to train()**

In `jaxzero/train.py`, find the block starting with:
```python
    # Warmup JIT: compile initial + recurrent inference at the collection batch size
```

Add after the existing `mcts.search(...)` warmup call (around line 322):

```python
    # Warmup JIT for reanalysis batch size: B*(U+1) positions per batch call
    if config.use_reanalyze:
        _B_re = config.batch_size * (config.unroll_steps + 1)
        _obs_re = np.ones((_B_re, config.num_agents, config.obs_size), dtype=np.float32)
        _legal_re = np.ones((_B_re, config.num_agents, config.action_space_size), dtype=bool)
        print(f"Compiling JAX reanalysis MCTS (batch={_B_re}, one-time)...")
        mcts.search(params, _obs_re, _legal_re, np.random.default_rng(1))
        print("Reanalysis compilation done.")
```

- [ ] **Step 2: Smoke-test warmup by running one training step**

Run: `python -m jaxzero.main --env 3m 2>&1 | head -20`

Expected output includes both compilation messages:
```
Compiling JAX model (one-time)...
Compilation done.
Compiling JAX reanalysis MCTS (batch=1536, one-time)...
Reanalysis compilation done.
```

- [ ] **Step 3: Commit**

```bash
git add jaxzero/train.py
git commit -m "perf: warmup JIT for reanalysis MCTS batch size to avoid mid-training stall"
```

---

### Task 5: Verify end-to-end: reanalysis runs and policy loss trends down

Run a short training session and confirm policy loss is decreasing, not plateauing.

**Files:**
- No file changes — observation only

- [ ] **Step 1: Run short training, capture loss log**

```bash
python -m jaxzero.main --env 3m 2>&1 | grep "Step " | head -50
```

Expected: policy loss should be declining across the first 5000 steps, not flatlined at ~4.1.

Observe that the "p=" values decrease over time. Example healthy trajectory:
```
Step 0:   ... p=5.2 ...
Step 100: ... p=4.8 ...
Step 500: ... p=3.9 ...
```

If policy loss still plateaus at 4.1, check:
1. `config.use_reanalyze` is True (it is by default)
2. `config.revisit_policy_search_rate` is 0.99
3. `self._mcts` is not None in the worker
4. The mock was removed and real MCTS is called

- [ ] **Step 2: Final commit if no regressions**

```bash
git add .
git commit -m "feat: reanalysis working end-to-end, policy loss declining"
```

---

## Self-Review

**Spec coverage:**
- [x] Reanalysis implemented in `make_batch`
- [x] Uses `params` (target_params from caller)
- [x] Re-runs MCTS for `ceil(B * revisit_policy_search_rate)` positions
- [x] Covers all `U+1` unroll positions per game
- [x] Fresh visit counts, Q-values, sampled actions used for policy targets
- [x] Stored obs/actions/rewards/values unchanged (value reanalysis is a separate concern)
- [x] Stop-gradient on AWPO baseline fixed
- [x] JIT warmup for new batch size
- [x] Tests for both reanalysis and no-reanalysis paths

**Placeholder scan:** None found — all code blocks are complete.

**Type consistency:**
- `_pad_to_k` returns `(pol, sa, qv, mask)` — used with same order in Task 3 ✓
- `SearchOutput.sampled_visit_counts`, `.sampled_actions`, `.sampled_qvalues` — match `sampled_mcts.py:SearchOutput` definition ✓
- `fresh_policies`, `fresh_qvalues`, `fresh_masks`, `fresh_sa` shapes consistent with `BatchData` fields ✓
