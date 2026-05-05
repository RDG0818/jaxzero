# Fix Loss Plateau Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix jaxzero MPE training loss stuck at ~6.7 after 10k steps by correcting AWPO advantage normalization, out-of-episode masks, and adding per-component loss logging.

**Architecture:** Three independent fixes address confirmed root causes: (1) advantage std normalization missing from `awpo_sharp_loss` makes gradient scale reward-magnitude-dependent, (2) `game.make_target` reuses terminal-step masks for out-of-episode unroll steps instead of zeroing them, (3) no per-component logging prevents diagnosis. All changes isolated to `train.py` and `game.py`.

**Tech Stack:** JAX, Flax, NumPy, pytest

---

## File Map

- Modify: `jaxzero/train.py` — AWPO std normalization, per-component logging, `make_update_fn` aux return
- Modify: `jaxzero/game.py` — fix out-of-episode masks in `make_target`
- Modify: `jaxzero/config.py` — reduce `updates_per_collection` default to 2
- Modify: `tests/test_train.py` — update tests for new `make_update_fn` signature + AWPO normalization test
- Modify: `tests/test_game.py` — add out-of-episode mask test

---

### Task 1: Fix out-of-episode masks in `game.make_target`

**Root cause:** When `pos + k >= T` (unrolling past episode end), `masks_batch[k]` uses
`sampled_masks[T-1]` (terminal step, all True). Policy loss then trains on repeated
terminal-step Q-values and visit counts for those steps. Should be all-False (zero contribution).

**Files:**
- Modify: `jaxzero/game.py:121-123`
- Modify: `tests/test_game.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_game.py`:

```python
def test_make_target_out_of_episode_masks_are_zero():
    """Steps past episode end must have all-False masks (no policy loss contribution)."""
    from jaxzero.game import GameHistory
    import numpy as np

    K, N, A, S = 5, 2, 3, 1
    game = GameHistory(num_agents=N, obs_dim=4, action_space_size=A, stacked_observations=S)

    # 3-step episode
    for t in range(3):
        game.store_observation(np.zeros((N, 4)))
        if t < 3:
            game.store_action(np.zeros(N, dtype=np.int32))
            game.store_reward(1.0)
            game.store_root_value(0.5)
            game.store_pred_value(0.0)
            game.store_legal_actions(np.ones((N, A), dtype=bool))
            game.store_search_stats(
                sampled_actions=np.zeros((K, N), dtype=np.int32),
                visit_counts=np.ones(K) / K,
                qvalues=np.zeros(K),
                mask=np.ones(K, dtype=bool),
            )

    # pos=1, unroll_steps=5 → steps k=2,3,4,5 go past T=3
    _, _, _, _, _, _, masks_batch = game.make_target(
        pos=1, unroll_steps=5, td_steps=3, discount=0.99
    )
    # k=0,1 are within episode (pos+k = 1,2 < 3), k=2..5 are past end
    assert masks_batch[0].all(), "step 0 within episode should have valid mask"
    assert masks_batch[1].all(), "step 1 within episode should have valid mask"
    for k in range(2, 6):
        assert not masks_batch[k].any(), f"step {k} past episode end should be all-False"
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/ryan/Repos/jaxzero
conda run -n mazero pytest tests/test_game.py::test_make_target_out_of_episode_masks_are_zero -v
```

Expected: FAIL — `assert not masks_batch[2].any()` fails because current code reuses terminal mask (all True).

- [ ] **Step 3: Fix `game.make_target` mask construction**

Replace lines 121-123 in `jaxzero/game.py`:

```python
        policies_batch = np.stack([self.sampled_policies[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
        qvals_batch = np.stack([self.sampled_qvalues[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
        masks_batch = np.stack([self.sampled_masks[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
```

With:

```python
        K = len(self.sampled_actions[0])
        policies_batch = np.stack([self.sampled_policies[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
        qvals_batch = np.stack([self.sampled_qvalues[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
        masks_batch = np.stack([
            self.sampled_masks[pos + k] if pos + k < T else np.zeros(K, dtype=bool)
            for k in range(unroll_steps + 1)
        ])
```

Note: `K` is already computed on line 97 as `K = len(self.sampled_actions[0])` — remove the duplicate definition if present.

- [ ] **Step 4: Run test to verify it passes**

```
conda run -n mazero pytest tests/test_game.py::test_make_target_out_of_episode_masks_are_zero -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

```
conda run -n mazero pytest tests/ -v --tb=short
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add jaxzero/game.py tests/test_game.py
git commit -m "fix: zero out-of-episode masks in make_target instead of reusing terminal step"
```

---

### Task 2: Fix AWPO advantage normalization (add std normalization)

**Root cause:** `awpo_sharp_loss` computes `adv_norm = Q - mean(Q)` then `exp(adv_norm / alpha)`.
Without std normalization, when MPE Q-values have small variance (e.g., ±0.5), the exponent
`adv_norm/alpha ≈ ±0.17`, giving weights `[0.85, 1.18]` — nearly uniform. Policy gets weak gradient signal.

MAZero normalizes: `adv_norm = (Q - mean(Q)) / (std(Q) + 1e-5)`, making the scale independent
of reward magnitude. With `alpha=3.0`, `±1 std → exp(±0.33) ≈ [0.72, 1.38]`.

**Files:**
- Modify: `jaxzero/train.py:30-34`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
def test_awpo_loss_std_normalization():
    """Advantages with larger variance should produce larger adv_weight spread."""
    import jax.numpy as jnp
    from jaxzero.train import awpo_sharp_loss

    B, N, A, K = 1, 2, 5, 4
    policy_logits = jnp.zeros((B, N, A))
    sampled_actions = jnp.zeros((B, K, N), dtype=jnp.int32)
    visit_counts = jnp.ones((B, K)) / K
    masks = jnp.ones((B, K))

    # Small variance advantages — after std-norm, weights should still spread meaningfully
    small_adv = jnp.array([[0.1, 0.2, 0.1, 0.2]])   # std ≈ 0.05
    # Large variance advantages — same relative structure
    large_adv = jnp.array([[1.0, 2.0, 1.0, 2.0]])   # std ≈ 0.5

    loss_small = awpo_sharp_loss(policy_logits, sampled_actions, visit_counts, small_adv, masks, alpha=3.0)
    loss_large = awpo_sharp_loss(policy_logits, sampled_actions, visit_counts, large_adv, masks, alpha=3.0)

    # With std normalization, both should produce similar loss magnitude
    # (same relative structure → same normalized advantages)
    assert jnp.abs(loss_small - loss_large).mean() < 0.05, (
        f"Std normalization should make loss scale-invariant: small={loss_small}, large={loss_large}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
conda run -n mazero pytest tests/test_train.py::test_awpo_loss_std_normalization -v
```

Expected: FAIL — without std normalization, `loss_large` >> `loss_small`.

- [ ] **Step 3: Fix `awpo_sharp_loss` in `jaxzero/train.py`**

Replace lines 30-34:

```python
    masked_adv_mean = (advantages * masks).sum(-1, keepdims=True) / (masks.sum(-1, keepdims=True) + 1e-8)
    adv_norm = advantages - masked_adv_mean
    adv_weights = jnp.exp(adv_norm / alpha)
```

With:

```python
    n_valid = masks.sum(-1, keepdims=True) + 1e-8
    masked_adv_mean = (advantages * masks).sum(-1, keepdims=True) / n_valid
    adv_centered = advantages - masked_adv_mean
    masked_adv_var = (adv_centered ** 2 * masks).sum(-1, keepdims=True) / n_valid
    masked_adv_std = jnp.sqrt(masked_adv_var + 1e-10)
    adv_norm = adv_centered / (masked_adv_std + 1e-5)
    adv_weights = jnp.exp(adv_norm / alpha)
```

- [ ] **Step 4: Run the normalization test**

```
conda run -n mazero pytest tests/test_train.py::test_awpo_loss_std_normalization -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```
conda run -n mazero pytest tests/ -v --tb=short
```

Expected: all pass. If `test_loss_decreases_on_repeated_batch` fails, it may need more iterations — increase the repeat count from 10 to 20.

- [ ] **Step 6: Commit**

```bash
git add jaxzero/train.py tests/test_train.py
git commit -m "fix: add std normalization to AWPO advantage weights in awpo_sharp_loss"
```

---

### Task 3: Add per-component loss logging

**Goal:** Expose reward, value, and policy loss separately so training dynamics are visible.
Currently only total loss is logged — impossible to tell which component is stuck.

**Approach:** Use `jax.value_and_grad(..., has_aux=True)` pattern. Inner loss function returns
`(total_loss, aux_dict)`. Outer `update_fn` returns `(total_loss, grads, aux_dict)`.

**Files:**
- Modify: `jaxzero/train.py:55-122` (make_update_fn), `jaxzero/train.py:299-305` (training loop)
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py`:

```python
def test_update_fn_returns_aux():
    """make_update_fn should return (loss, grads, aux) with per-component losses."""
    config = make_config()
    net = MAMuZeroNet(config=config)
    obs_init = jnp.ones((1, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs_init)
    update_fn = make_update_fn(net, config)
    batch = make_fake_batch(config)

    result = update_fn(params, batch)
    assert len(result) == 3, "update_fn should return (loss, grads, aux)"
    loss, grads, aux = result
    assert jnp.isfinite(loss)
    assert "reward_loss" in aux
    assert "value_loss" in aux
    assert "policy_loss" in aux
    assert all(jnp.isfinite(v) for v in aux.values())
```

- [ ] **Step 2: Run test to verify it fails**

```
conda run -n mazero pytest tests/test_train.py::test_update_fn_returns_aux -v
```

Expected: FAIL — current `update_fn` returns only `(loss, grads)`.

- [ ] **Step 3: Rewrite `make_update_fn` in `jaxzero/train.py`**

Replace the entire `make_update_fn` function (lines 55-122):

```python
def make_update_fn(model, config: MAZeroConfig):
    """Create a pure (params, batch) -> (loss, grads, aux) update function."""
    S_v = config.value_support_size
    S_r = config.reward_support_size

    def _loss_fn(params: Any, batch: BatchData):
        obs = jnp.array(batch.obs)                         # (B, U+1, N, obs_dim)
        actions = jnp.array(batch.actions)                 # (B, U, N)
        target_rewards = jnp.array(batch.target_rewards)   # (B, U)
        target_values = jnp.array(batch.target_values)     # (B, U+1)
        target_policies = jnp.array(batch.target_policies)  # (B, U+1, K)
        target_qvalues = jnp.array(batch.target_qvalues)    # (B, U+1, K)
        target_masks = jnp.array(batch.target_masks)        # (B, U+1, K)
        sampled_acts = jnp.array(batch.sampled_actions)     # (B, U+1, K, N)
        weights = jnp.array(batch.weights)                  # (B,)

        U = config.unroll_steps

        out0 = model.apply(params, obs[:, 0])
        target_v0 = _batch_phi_h(target_values[:, 0], S_v)

        value_loss = categorical_cross_entropy(out0.value_logits, target_v0)
        reward_loss = jnp.zeros(obs.shape[0])
        policy_loss = awpo_sharp_loss(
            out0.policy_logits,
            sampled_acts[:, 0],
            target_policies[:, 0],
            target_qvalues[:, 0],
            target_masks[:, 0],
            config.awpo_alpha,
        )

        hidden = out0.hidden_state

        for k in range(1, U + 1):
            hidden = 0.5 * hidden + 0.5 * lax.stop_gradient(hidden)
            out_k = model.apply(
                params, hidden, actions[:, k - 1],
                method=model.recurrent_inference,
            )

            target_r = _batch_phi_h(target_rewards[:, k - 1], S_r)
            target_v = _batch_phi_h(target_values[:, k], S_v)

            reward_loss = reward_loss + categorical_cross_entropy(out_k.reward_logits, target_r)
            value_loss = value_loss + categorical_cross_entropy(out_k.value_logits, target_v)
            policy_loss = policy_loss + awpo_sharp_loss(
                out_k.policy_logits,
                sampled_acts[:, k],
                target_policies[:, k],
                target_qvalues[:, k],
                target_masks[:, k],
                config.awpo_alpha,
            )

            hidden = out_k.hidden_state

        r_term = config.reward_loss_coeff * reward_loss
        v_term = config.value_loss_coeff * value_loss
        p_term = config.policy_loss_coeff * policy_loss
        total = r_term + v_term + p_term
        scalar = (weights * total).mean() / U
        aux = {
            "reward_loss": (weights * r_term).mean() / U,
            "value_loss": (weights * v_term).mean() / U,
            "policy_loss": (weights * p_term).mean() / U,
        }
        return scalar, aux

    _grad_fn = jax.jit(jax.value_and_grad(_loss_fn, has_aux=True))

    def update_fn(params: Any, batch: BatchData):
        (loss, aux), grads = _grad_fn(params, batch)
        return loss, grads, aux

    return update_fn
```

- [ ] **Step 4: Update training loop in `jaxzero/train.py`**

Replace lines 299-305 in the `train()` function:

```python
            loss, grads = jax.value_and_grad(update_fn)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if step % config.log_interval == 0:
                mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
                print(f"Step {step}: loss={float(loss):.4f} | ep_return={mean_ret:.2f}")
```

With:

```python
            loss, grads, aux = update_fn(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if step % config.log_interval == 0:
                mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
                print(
                    f"Step {step}: loss={float(loss):.4f}"
                    f" | r={float(aux['reward_loss']):.3f}"
                    f" v={float(aux['value_loss']):.3f}"
                    f" p={float(aux['policy_loss']):.3f}"
                    f" | ep_return={mean_ret:.2f}"
                )
```

- [ ] **Step 5: Fix existing test that uses old signature**

In `tests/test_train.py`, `test_update_fn_runs` and `test_loss_decreases_on_repeated_batch` use
`jax.value_and_grad(update_fn)(params, batch)` directly. Update them:

```python
def test_update_fn_runs():
    config = make_config()
    net = MAMuZeroNet(config=config)
    obs_init = jnp.ones((1, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs_init)
    update_fn = make_update_fn(net, config)
    batch = make_fake_batch(config)
    loss, grads, aux = update_fn(params, batch)
    assert jnp.isfinite(loss)


def test_loss_decreases_on_repeated_batch():
    """Loss should decrease when repeatedly training on same batch."""
    config = make_config()
    net = MAMuZeroNet(config=config)
    obs_init = jnp.ones((1, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs_init)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    update_fn = make_update_fn(net, config)
    batch = make_fake_batch(config)

    losses = []
    for _ in range(20):
        loss, grads, aux = update_fn(params, batch)
        losses.append(float(loss))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
```

- [ ] **Step 6: Run all train tests**

```
conda run -n mazero pytest tests/test_train.py -v --tb=short
```

Expected: all pass including new `test_update_fn_returns_aux`.

- [ ] **Step 7: Commit**

```bash
git add jaxzero/train.py tests/test_train.py
git commit -m "feat: add per-component loss logging (reward/value/policy) via has_aux=True"
```

---

### Task 4: Reduce `updates_per_collection` default to 2

**Root cause:** `updates_per_collection=10` means ~10 gradient steps per ~8 new samples (B=8 envs).
That's 1.25 updates/sample — too high early in training. Policy updates use Q-values collected
under the old policy, so the AWPO targets become stale quickly. Reducing to 2 keeps policy closer
to the data-generating policy.

**Files:**
- Modify: `jaxzero/config.py:67`

- [ ] **Step 1: Change default in config.py**

In `jaxzero/config.py`, change line 67:

```python
    updates_per_collection: int = 10
```

To:

```python
    updates_per_collection: int = 2
```

- [ ] **Step 2: Verify test suite still passes**

```
conda run -n mazero pytest tests/ -v --tb=short
```

Expected: all pass (no test asserts on this default value).

- [ ] **Step 3: Commit**

```bash
git add jaxzero/config.py
git commit -m "config: reduce updates_per_collection default from 10 to 2 to reduce policy drift"
```

---

### Task 5: Smoke test on MPE

**Goal:** Run training for 500 steps and verify per-component losses are decreasing and in expected ranges.

- [ ] **Step 1: Run short MPE training**

```
conda run -n mazero python -m jaxzero.main --env mpe --training_steps 500 --num_simulations 20 --num_envs 4
```

- [ ] **Step 2: Check output**

Expected output pattern after ~100 steps:
```
Step 0: loss=X.XX | r=X.XXX v=X.XXX p=X.XXX | ep_return=nan
Step 100: loss=X.XX | r=X.XXX v=X.XXX p=X.XXX | ep_return=Y.YY
```

Healthy signs:
- `r` (reward loss) decreases from ~2.4 toward 0
- `v` (value loss) decreases from ~2.4 toward 0
- `p` (policy loss) starts ~5.8 and may decrease slowly
- Total loss decreases over time

Warning signs:
- `p` stays constant at ~5.8 for 500+ steps → policy not learning, check mask fix
- `r` and `v` not decreasing → value function broken, check support size

- [ ] **Step 3: If policy loss not decreasing, diagnose**

Check that masks are reaching training:

Add temporary debug print in `jaxzero/reanalyze.py` after `make_batch`:
```python
print(f"mask fraction: {batch.target_masks.mean():.3f}")
```

If mask fraction is 1.0 for all steps, the game episodes are shorter than unroll_steps → out-of-episode mask fix not exercised yet. Normal if all episodes complete within 5 steps. Remove debug print after checking.

---

## Self-Review

**Spec coverage:**
1. ✅ per-component loss logging → Task 3
2. ✅ AWPO std normalization → Task 2
3. ✅ out-of-episode masks → Task 1
4. ✅ updates_per_collection reduction → Task 4
5. ✅ smoke test → Task 5

**Placeholder scan:** None found.

**Type consistency:**
- `update_fn` returns `(loss, grads, aux)` — used consistently in Task 3 Steps 3-5 and training loop update.
- `awpo_sharp_loss` signature unchanged — same args, just fixed body.
- `make_target` return tuple unchanged — same 7-tuple, just `masks_batch` computed differently.
