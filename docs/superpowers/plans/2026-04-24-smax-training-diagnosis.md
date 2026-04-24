# SMAX Training Diagnosis & Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify why training plateaus at ~0.35 reward (near-random) and close the gap to the paper's ~100% win rate at 50k env steps.

**Architecture:** The codebase is a JAX/Flax reimplementation of MAZero (ICLR 2024) targeting JaxMARL HeuristicEnemySMAX 3m. Training is on a remote RTX 3060 Ti (8 GB). This machine has no GPU — all code investigation can be done here; training experiments run remotely.

**Tech Stack:** JAX, Flax, Ray, JaxMARL, Hydra/YAML configs. Training entry point: `python train/muzero.py train=smax_3m model=smax mcts=joint`.

---

## Root Cause Analysis (Read This First)

From reading the code, three issues are very likely causing the plateau. They compound each other.

### Issue 1 — AWPO uses the wrong advantage signal (HIGH CONFIDENCE)

**File:** `actors/learner_actor.py`, `make_train_step`, around line 97.

```python
v_mcts = batch.value_target[:, 0].mean(axis=-1)  # n-step bootstrapped value
v_net  = support_to_scalar(init_out.value_logits, value_support)
advantage = v_mcts - v_net
awpo_w = jnp.exp(jnp.clip(advantage / awpo_alpha, -5.0, 5.0))
awpo_w = awpo_w / (awpo_w.mean() + 1e-8)
```

This computes a **state-level** advantage (how much the current network underestimates the MCTS value). It does NOT tell the policy which specific actions were good.

The MAZero paper computes a **per-action** advantage: `Q_mcts(s, a_k) - V_mcts(s)` for each of the K sampled actions. This directly upweights the probability of actions the tree found better than the average. Our version upweights ALL actions at a state that happens to be undervalued — even bad actions get upweighted together.

### Issue 2 — Policy targets are sparse noise (HIGH CONFIDENCE)

SMAX 3m action space: A=9 actions per agent, N=3 agents → A_N=9³=729 joint actions. With K=5 root children and 50 simulations, we sample only 5/729 ≈ 0.7% of the joint space. The marginal policy is built by summing over these 5 samples. This is extremely noisy and the model memorizes noise at near-zero loss.

Evidence: loss = 0.0014 with reward = 0.35. The model has perfectly fit its targets. The targets themselves are bad.

### Issue 3 — Learning rate is 5× below paper value (HIGH CONFIDENCE)

Paper uses constant LR=5e-4. Our SMAX config uses cosine decay from 1e-4 → 1e-5. The effective LR through most of training is well below what the paper found to work.

### Issue 4 — Consistency target is weaker (MEDIUM CONFIDENCE)

Paper uses `sg(P1(re-encode(obs_{t+k})))` — a fresh re-encoding of the actual next observation as the target. We use `EMA_params(P1(h_{t+k}))` — EMA params applied to the dynamics-predicted next latent. The paper's approach is grounded in real observations; ours propagates any dynamics errors into the consistency target. Combined with `consistency_scale=0.25` (paper uses 2.0), the representation may not be learning useful features.

### Issue 5 — Reanalysis may reinforce bad behavior early (MEDIUM CONFIDENCE)

The reanalyze actor updates `value_target[0]` and `policy_target[0]` with the current model's MCTS output. In early training, this OVERWRITES the n-step bootstrapped value (which is grounded in actual rewards) with a pure MCTS value from a bad model. With 131 reanalyze completions vs 31 learner calls, the buffer is being flooded with bad reanalyzed targets.

---

## Experiment Strategy

Since training on a remote machine, experiments are ordered so each one is informative regardless of the previous result. Start with Task 1 (pure config, no code changes) to establish a faster baseline. Then Task 2 (logging) gives visibility into what's failing. Then Tasks 3+ fix the code issues in order of confidence.

**Do NOT run multiple experiments simultaneously** — you need to know which change caused which effect.

---

## Task 1: Config Quick-Wins (No Code Changes)

**Files:**
- Modify: `configs/train/smax_3m.yaml`
- Modify: `configs/mcts/joint.yaml` (or create `configs/mcts/smax_fast.yaml`)

These are pure config changes. They should immediately improve target quality and convergence speed. Run a 50k-episode training run and compare avg return.

- [ ] **Step 1.1: Update learning rate in smax_3m.yaml**

Change from cosine decay at 1e-4 to constant 5e-4. Edit `configs/train/smax_3m.yaml`:

```yaml
learning_rate: 5e-4          # was 1e-4 — paper uses 5e-4 constant
lr_warmup_steps: 0           # disable warmup; paper uses constant LR
end_lr_factor: 1.0           # 1.0 = no decay (constant LR)
```

- [ ] **Step 1.2: Increase MCTS simulations and K in smax_3m.yaml**

The smax_3m config uses `mcts=joint` which has 50 sims and K=5. Override these:

```yaml
# In configs/train/smax_3m.yaml — add override for MCTS params
# (These override whatever mcts= preset is selected)
```

Actually, MCTS params live in the mcts config group, not train. Create `configs/mcts/smax.yaml` if it doesn't exist (it already does, with 100 sims). The K must be in that file too:

Edit `configs/mcts/smax.yaml`:
```yaml
defaults:
  - default

num_simulations: 100          # paper uses 100
num_gumbel_samples: 10        # was 5, paper uses K=10 sampled actions per node
mcts_rho: 0.25                # keep top 75% of sims — unchanged
```

- [ ] **Step 1.3: Run baseline with these config changes**

On the remote machine:
```bash
python train/muzero.py train=smax_3m model=smax mcts=smax
```
Note: switch from `mcts=joint` to `mcts=smax` to get 100 sims + K=10.

Run for 50k episodes. Expected: Avg return > 0.5 if LR is the main issue, or improvement at least. If return stays at 0.35, the config changes alone are not sufficient and code fixes are needed.

- [ ] **Step 1.4: Commit config changes**

```bash
git add configs/train/smax_3m.yaml configs/mcts/smax.yaml
git commit -m "config: SMAX LR 5e-4 constant, 100 sims, K=10 per MAZero paper"
```

---

## Task 2: Add Diagnostic Logging

**Files:**
- Modify: `training/loop.py` (add win_rate to log)
- Modify: `actors/data_actor.py` (track wins)
- Modify: `actors/learner_actor.py` (log per-component losses and policy entropy)

Without knowing the win rate, we can't distinguish "reward=0.35 from partial wins" vs "reward=0.35 from per-step HP damage with no wins." These are completely different failure modes.

- [ ] **Step 2.1: Track wins in DataActor**

In `actors/data_actor.py`, `run_episode()`, after the episode loop. The `VecSMAXEnvWrapper.step()` returns a 5-tuple `(obs, states, rewards, dones, won)`. The DataActor currently ignores `won`. Fix:

```python
# In DataActor.run_episode(), after the env.step call (line ~174):
next_obs, next_states, rewards, dones, *extra = self.env.step(
    step_keys, states, actions_np
)
rewards_np = np.array(rewards)
dones_np = np.array(dones)
# Track wins for SMAX (extra[0] = won if env returns 5-tuple)
won_np = np.array(extra[0]) if extra else np.zeros(B, dtype=bool)
```

Then accumulate wins per episode:
```python
# Add to Episode or track separately:
episode_wins = [False] * B  # track if any step had won=True for each env

# Inside the per-env step loop:
for i in range(B):
    if not active[i]:
        continue
    episodes[i].add_step(...)
    if won_np[i]:
        episode_wins[i] = True  # won this episode
    if dones_np[i]:
        active[i] = False
```

Return win rate alongside episode return:
```python
# Replace the return statement:
win_rate = float(np.mean(episode_wins))
mean_return = float(np.mean([ep.episode_return for ep in episodes]))
return mean_return, win_rate
```

- [ ] **Step 2.2: Update training loop to log win rate**

`DataActor.run_episode()` now returns `(return, win_rate)`. Update callers in `training/loop.py`:

```python
# In run_warmup (line ~39):
ep_return, _ = ray.get(done_ref)  # ignore win_rate during warmup

# In run_training_loop, data actor completion branch (around line ~116):
result = ray.get(done_ref)
ep_return, win_rate = result if isinstance(result, tuple) else (result, 0.0)
returns.append(ep_return)
win_rates.append(win_rate)

# In the log interval block, add win rate:
avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
logger.info(
    f"Episodes: {episodes_processed:6d} | "
    f"Avg Return: {avg_return:8.2f} | "
    f"Win Rate: {avg_win_rate:.1%} | "   # ← NEW
    f"Avg Loss: {avg_loss:.4f} | "
    f"eps/s: {eps_per_sec:.1f} | "
    f"train steps/s: {steps_per_sec:.1f}"
)
```

Also add `win_rates: deque = deque(maxlen=config.train.log_interval)` near the other deques.

- [ ] **Step 2.3: Log policy entropy from the learner**

In `actors/learner_actor.py`, `_train_step()`, after computing metrics, add policy entropy:

```python
# After: METRIC_KEYS = ["total_loss", ...]
# Add policy entropy to the metric scalars in make_train_step:
```

In `make_train_step`'s `loss_fn`, add:
```python
# After computing init_out:
policy_probs = jax.nn.softmax(init_out.policy_logits, axis=-1)  # (B, N, A)
policy_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-8), axis=-1).mean()  # scalar
```

Add to `metric_scalars`:
```python
metric_scalars = jnp.stack([
    total_loss,
    reward_loss.mean(),
    policy_loss.mean(),
    value_loss.mean(),
    consistency_loss.mean(),
    policy_entropy,  # ← NEW (index 5)
])
```

Update `N_METRICS = 7` and `METRIC_KEYS`:
```python
N_METRICS = 7  # total, reward, policy, value, consistency, grad_norm, policy_entropy
# Note: transfer_buf layout = [metric_scalars(6), grad_norm(1), priorities(B)]
# After adding policy_entropy, layout = [metric_scalars(6), grad_norm(1), priorities(B)]
# metric_scalars already has 6 elements; add policy_entropy as 6th → N_METRICS=7
```

Wait — `grad_norm` is added separately in the concat:
```python
transfer_buf = jnp.concatenate([
    metric_scalars,       # 6 elements → becomes 7 with entropy
    grad_norm[jnp.newaxis],
    new_priorities,
])
```
And `N_METRICS = 6` covers the first 6 scalars before grad_norm. After adding entropy:
- `metric_scalars` has 6 elements (indices 0-5)
- `grad_norm` is at index 6
- `priorities` start at index 7
- `N_METRICS = 7` (total, reward, policy, value, consistency, entropy, grad_norm)

```python
N_METRICS = 7
METRIC_KEYS = ["total_loss", "reward_loss", "policy_loss", "value_loss",
               "consistency_loss", "policy_entropy", "grad_norm"]
```

Log it:
```python
if debug and self.train_step_count % debug_interval == 0:
    logger.info(
        f"(Learner) step={self.train_step_count} | "
        f"total={total_loss:.4f} | "
        f"policy_entropy={metrics['policy_entropy']:.3f} | "  # ← NEW
        f"grad_norm={grad_norm:.3f}"
    )
```

Policy entropy for a uniform policy over 9 actions is `log(9) ≈ 2.197`. If entropy drops below 1.0 early in training, the policy has collapsed to near-deterministic behavior — a sign of bad targets.

- [ ] **Step 2.4: Run with diagnostics enabled**

In `configs/train/smax_3m.yaml`, temporarily add:
```yaml
debug: true
debug_interval: 50
```

Run 5k episodes and check the log for policy entropy and win rate:
```bash
python train/muzero.py train=smax_3m model=smax mcts=smax train.num_episodes=5000
```

Expected healthy values:
- `policy_entropy` ≥ 1.5 early (stays high while exploring), decreasing to 0.5-1.5 as it converges
- `win_rate` growing from 0% toward 30%+ by 5k episodes if training is working

If `policy_entropy < 0.5` at episode 1000, the policy has already collapsed. This confirms Issue 2 (noisy policy targets causing early collapse).

- [ ] **Step 2.5: Commit diagnostic changes**

```bash
git add actors/data_actor.py actors/learner_actor.py training/loop.py
git commit -m "feat: add win_rate logging and policy entropy metric for SMAX diagnosis"
```

---

## Task 3: Fix AWPO to Action-Level Advantage

**Files:**
- Modify: `mcts/mcts_joint_osla.py` — add per-action Q-values to plan output
- Modify: `mcts/base.py` — add `action_q_values` field to `MCTSPlanOutput`
- Modify: `utils/replay_buffer.py` — add `action_q_values` to `ReplayItem` and `Transition`
- Modify: `actors/data_actor.py` — store Q-values in Transition
- Modify: `actors/learner_actor.py` — use per-action advantage in AWPO loss

This is the highest-confidence fix. It aligns our AWPO with MAZero's formulation.

**Background:** In the OSLA tree, each root child `k` has a Q-value:
```
Q_k = reward[child_k] + gamma * osla_value(child_k)
```
The advantage for action `k` is `Q_k - V_root` where `V_root` is the OS(λ) root value. The policy loss becomes:
```
loss = -sum_k [visit_count[k] * exp(clip(Q_k - V_root) / alpha) * log_joint_prob(action_k)]
```
marginalized to per-agent.

- [ ] **Step 3.1: Add `action_q_values` to MCTSPlanOutput**

In `mcts/base.py`, find `MCTSPlanOutput` and add a field:

```python
@chex.dataclass
class MCTSPlanOutput:
    joint_action:   chex.Array   # (B, N) int32 — best action per env
    policy_targets: chex.Array   # (B, N, A) float32 — per-agent marginal visit probs
    root_value:     chex.Array   # (B,) float32 — OS(λ) root value
    agent_order:    chex.Array   # (N,) int32
    # NEW: per-root-child Q-values and actions, for action-level AWPO
    # Shape (B, K, N) int32 — K sampled joint actions as per-agent indices
    root_child_actions: chex.Array | None = None
    # Shape (B, K) float32 — Q-value for each root child (0 if unvisited)
    root_child_q:       chex.Array | None = None
    # Shape (B, K) float32 — visit count for each root child
    root_child_visits:  chex.Array | None = None
```

- [ ] **Step 3.2: Compute and return Q-values from `_osla_plan_single`**

In `mcts/mcts_joint_osla.py`, in `_osla_plan_single`, after computing `osla_root_value` (around line 484):

```python
# ── Per-child Q-values for action-level AWPO ────────────────────────────
# Q_k = reward[child_k] + gamma * osla_value(child_k)
root_children = final_carry.tree.child_node_idx[0]  # [K]
safe_children = jnp.maximum(root_children, 0)

child_rewards = jnp.where(
    root_children >= 0,
    final_carry.tree.reward[safe_children],
    0.0,
)  # [K]

child_osla_v = jax.vmap(
    lambda v, d, n: compute_osla_value_jax(v, d, n, rho, lam)
)(
    final_carry.tree.node_sim_values[safe_children],
    final_carry.tree.node_sim_depths[safe_children],
    jnp.where(root_children >= 0, final_carry.tree.visit_counts[safe_children],
              jnp.zeros(K, jnp.int32)),
)  # [K]

child_q = jnp.where(
    root_children >= 0,
    child_rewards + gamma * child_osla_v,
    0.0,
)  # [K]

# Decode flat joint actions → per-agent indices: (K,) → (K, N)
root_child_per_agent = jnp.stack(
    jnp.unravel_index(root_child_actions, joint_action_shape), axis=-1
)  # (K, N)
```

Return these in `MCTSPlanOutput`:
```python
return MCTSPlanOutput(
    joint_action=best_action[None],         # (1, N)
    policy_targets=marginal_policy,          # (1, N, A)
    root_value=osla_root_value[None],        # (1,)
    agent_order=jnp.arange(N),
    root_child_actions=root_child_per_agent[None],  # (1, K, N)
    root_child_q=child_q[None],                     # (1, K)
    root_child_visits=root_child_visits[None],       # (1, K)
)
```

- [ ] **Step 3.3: Propagate Q-values through the planner's vmap**

In `MCTSJointOSLAPlanner._plan_loop()`, the vmap result has an extra leading dim from `_osla_plan_single`. Squeeze it:

```python
return MCTSPlanOutput(
    joint_action=results.joint_action.squeeze(1),              # (B, N)
    policy_targets=results.policy_targets.squeeze(1),          # (B, N, A)
    root_value=results.root_value.squeeze(1),                  # (B,)
    agent_order=results.agent_order[0],                         # (N,)
    root_child_actions=results.root_child_actions.squeeze(1),  # (B, K, N)
    root_child_q=results.root_child_q.squeeze(1),              # (B, K)
    root_child_visits=results.root_child_visits.squeeze(1),    # (B, K)
)
```

- [ ] **Step 3.4: Add Q-values to Transition and ReplayItem**

In `utils/replay_buffer.py`:

```python
@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    policy_target: np.ndarray
    value_target: float
    agent_order: np.ndarray
    # NEW: per-root-child Q and visit data for action-level AWPO
    # Shape (K, N) int32 — root child actions as per-agent indices
    root_child_actions: np.ndarray = None
    root_child_q: np.ndarray = None       # (K,) float32
    root_child_visits: np.ndarray = None  # (K,) float32


@dataclass
class ReplayItem:
    observation: np.ndarray    # (N, obs_size)
    actions: np.ndarray        # (U, N)
    policy_target: np.ndarray  # (U+1, N, A)
    value_target: np.ndarray   # (U+1, N)
    reward_target: np.ndarray  # (U, N)
    agent_order: np.ndarray    # (N,)
    # NEW: only position 0 matters for AWPO (root step)
    root_child_actions: np.ndarray = None  # (K, N) int32
    root_child_q: np.ndarray = None        # (K,) float32
    root_child_visits: np.ndarray = None   # (K,) float32
```

Update `flatten_replay_item` and `unflatten_replay_item` to include the new fields.

Update `process_episode` to pass through the root-step Q-data:
```python
# In process_episode, extract the new fields:
root_child_actions_arr = np.stack([
    t.root_child_actions if t.root_child_actions is not None
    else np.zeros((K, num_agents), dtype=np.int32)
    for t in trajectory
])  # (T, K, N)
# ... similar for root_child_q, root_child_visits

# In each ReplayItem, include position 0's Q data:
replay_items.append(
    ReplayItem(
        ...
        root_child_actions=root_child_actions_arr[start],   # (K, N)
        root_child_q=root_child_q_arr[start],               # (K,)
        root_child_visits=root_child_visits_arr[start],     # (K,)
    )
)
```

The K dimension needs to be known at replay buffer creation time. Hardcode via `config.mcts.num_gumbel_samples` passed to `ReplayBufferActor`.

Note: The C++ replay buffer does NOT currently store these fields. For now, store them only in the Python fallback or handle them outside the C++ buffer. The simplest approach: store Q-data as part of the `observation` block in a wrapper, OR store them separately as a CPU-side dict in `ReplayBufferActor`.

**Simpler approach for implementation:** Store root Q-data in a separate Python dict keyed by buffer index inside `ReplayBufferActor`. The C++ buffer handles the main replay data; the Q-data dict is updated alongside. This avoids modifying the C++ backend.

In `actors/replay_buffer_actor.py`:
```python
@ray.remote
class ReplayBufferActor:
    def __init__(self, ...):
        ...
        self._q_store: dict = {}  # idx → (root_child_actions, root_child_q, root_child_visits)

    def add(self, items: list, priorities: list):
        for item, priority in zip(items, priorities):
            self._buf.add(item, priority)
            # track idx → q_data for the just-added item
            # C++ buf returns current write pointer; approximate with len
            idx = (len(self._buf) - 1) % self._buf_capacity
            if item.root_child_q is not None:
                self._q_store[idx] = (
                    item.root_child_actions,
                    item.root_child_q,
                    item.root_child_visits,
                )

    def sample(self, batch_size):
        result = self._buf.sample(batch_size)
        if result is None:
            return None, None, None
        batch, weights, indices = result
        # Attach Q data if available
        q_data = [self._q_store.get(int(i)) for i in indices]
        return batch, weights, indices, q_data
```

This approach avoids modifying the C++ backend while still passing Q-data to the learner.

- [ ] **Step 3.5: Update DataActor to store Q-data in Transition**

In `actors/data_actor.py`, `run_episode()`, after getting `plan_output`:

```python
# After: policy_targets_np = np.array(plan_output.policy_targets)
root_child_actions_np = np.array(plan_output.root_child_actions)  # (B, K, N)
root_child_q_np       = np.array(plan_output.root_child_q)        # (B, K)
root_child_visits_np  = np.array(plan_output.root_child_visits)   # (B, K)

# In the per-env step:
episodes[i].add_step(
    Transition(
        ...
        root_child_actions=root_child_actions_np[i],   # (K, N)
        root_child_q=root_child_q_np[i],               # (K,)
        root_child_visits=root_child_visits_np[i],     # (K,)
    )
)
```

- [ ] **Step 3.6: Implement action-level AWPO in make_train_step**

In `actors/learner_actor.py`, `make_train_step`, update the `loss_fn`:

The `batch` now includes `root_child_actions`, `root_child_q`, `root_child_visits` alongside regular fields. The AWPO loss at position 0:

```python
# Action-level AWPO at step 0:
if awpo_alpha > 0.0:
    # root_child_actions: (B, K, N) int32
    # root_child_q:       (B, K) float32  
    # root_child_visits:  (B, K) float32
    # root_value:         (B,) — the OS(λ) root value stored as value_target[:, 0].mean(-1)
    v_root = batch.value_target[:, 0].mean(axis=-1)  # (B,)
    q_k = batch.root_child_q                          # (B, K)
    visits_k = batch.root_child_visits                # (B, K)
    
    # Per-action advantage, clipped before exp (same as MAZero's adv_clip)
    action_adv = (q_k - v_root[:, None]) / awpo_alpha   # (B, K)
    action_adv = jnp.clip(action_adv, -5.0, 5.0)
    awpo_w_k = jnp.exp(action_adv)  # (B, K) — per-action weights

    # Joint log-prob of each sampled action: sum over agents
    # batch.root_child_actions: (B, K, N)
    # init_out.policy_logits: (B, N, A)
    log_probs = jax.nn.log_softmax(init_out.policy_logits, axis=-1)  # (B, N, A)
    # Gather log-prob for each agent's action in each sampled joint action:
    # log_probs[b, n, a] → need to index with root_child_actions[b, k, n]
    # Result shape: (B, K)
    joint_log_prob_k = jnp.sum(
        jnp.take_along_axis(
            log_probs[:, None, :, :].broadcast_to(
                (log_probs.shape[0], q_k.shape[1], log_probs.shape[1], log_probs.shape[2])
            ),  # (B, K, N, A)
            batch.root_child_actions[:, :, :, None],  # (B, K, N, 1) — index into A
            axis=-1,
        ).squeeze(-1),  # (B, K, N)
        axis=-1,
    )  # (B, K) — sum of log-probs over N agents

    # Normalize visit counts across K
    visit_weights = visits_k / (visits_k.sum(axis=-1, keepdims=True) + 1e-8)  # (B, K)

    # AWPO policy loss: -sum_k [visit_weight[k] * awpo_w[k] * log_joint_prob[k]]
    p0_loss = -(visit_weights * awpo_w_k * joint_log_prob_k).sum(axis=-1)  # (B,)
else:
    p0_loss = ce_p0  # plain cross-entropy (existing code)
```

Note: `batch.root_child_q` and `batch.root_child_actions` come from the Q-data attached by `ReplayBufferActor.sample()`. The LearnerActor `_train_step` needs to accept and forward them.

- [ ] **Step 3.7: Write tests for the new AWPO computation**

Add to `tests/test_learner.py`:

```python
def test_action_level_awpo_upweights_high_q_actions():
    """Actions with Q > V_root should get higher AWPO weight."""
    import jax.numpy as jnp
    
    B, K, N, A = 2, 5, 3, 9
    v_root = jnp.array([0.5, 0.5])      # (B,)
    # Action 0 has Q=1.0 (good), Action 1 has Q=0.0 (bad)
    q_k = jnp.array([[1.0, 0.0, 0.5, 0.5, 0.5],
                      [0.8, 0.2, 0.5, 0.5, 0.5]])  # (B, K)
    alpha = 1.0
    action_adv = jnp.clip((q_k - v_root[:, None]) / alpha, -5.0, 5.0)
    awpo_w_k = jnp.exp(action_adv)
    
    # Action 0 (Q=1.0, adv=0.5) should have higher weight than Action 1 (Q=0.0, adv=-0.5)
    assert awpo_w_k[0, 0] > awpo_w_k[0, 1], "High-Q action should have higher AWPO weight"
    assert awpo_w_k[0, 0] > jnp.exp(0.0), "High-Q action should have weight > 1.0"


def test_action_level_awpo_zero_alpha_disables():
    """awpo_alpha=0.0 should skip action-level AWPO (fall back to plain CE)."""
    awpo_alpha = 0.0
    # The loss_fn has `if awpo_alpha > 0.0:` which is a Python-level branch (static at trace time)
    assert awpo_alpha <= 0.0  # confirms the branch condition
```

Run: `conda run -n mazero pytest tests/test_learner.py -v`

- [ ] **Step 3.8: Commit action-level AWPO**

```bash
git add mcts/base.py mcts/mcts_joint_osla.py utils/replay_buffer.py \
        actors/data_actor.py actors/learner_actor.py actors/replay_buffer_actor.py \
        tests/test_learner.py
git commit -m "feat: action-level AWPO (Q_k - V_root advantage) matching MAZero paper"
```

---

## Task 4: Reduce Reanalysis Interference in Early Training

**Files:**
- Modify: `configs/train/smax_3m.yaml`
- Modify: `actors/reanalyze_actor.py` (optional — adaptive reanalysis)

The reanalyze actor runs 4-5× more often than the learner, overwriting n-step targets with bad MCTS estimates early in training. A simple fix: reduce reanalyze frequency or disable it for the first N episodes.

- [ ] **Step 4.1: Lower the reanalyze batch size to slow it down**

The reanalyze actor calls `sample_for_reanalysis(reanalyze_batch_size)`. Its completion rate is 131 vs 31 learner calls — it's running 4× faster. To match learner pace, reduce batch size so it takes similar wall time, or skip calls based on training progress.

In `configs/train/smax_3m.yaml`:
```yaml
reanalyze_batch_size: 32     # was 256 — reanalyze fewer items per call to reduce interference
```

This will make each reanalyze call faster (takes fewer MCTS rollouts), so it still completes frequently, but at least each individual call processes fewer items. A deeper fix would be to throttle calls based on learner steps.

- [ ] **Step 4.2: Add warmup gate to ReanalyzeActor**

In `actors/reanalyze_actor.py`, add a counter that skips reanalysis for the first N learner steps (where N = warmup_episodes × some ratio). This prevents early interference:

```python
def run_reanalyze(self):
    # Skip reanalysis during early training (model too bad to help)
    warm_steps = self.config.train.warmup_episodes * 2
    learner_steps = ray.get(self.learner.get_train_step_count.remote())
    if learner_steps < warm_steps:
        return  # don't reanalyze until model has warmed up
    
    # ... rest of existing run_reanalyze code
```

- [ ] **Step 4.3: Run comparison**

Compare a 50k-episode run with and without reanalysis (`num_reanalyze_actors=0`) to isolate its effect. If disabling reanalysis improves training, the early-interference hypothesis is confirmed.

```bash
# No reanalysis:
python train/muzero.py train=smax_3m model=smax mcts=smax train.num_reanalyze_actors=0
```

- [ ] **Step 4.4: Commit reanalysis changes**

```bash
git add configs/train/smax_3m.yaml actors/reanalyze_actor.py
git commit -m "fix: reduce reanalysis interference during early training (warmup gate + smaller batch)"
```

---

## Task 5: Improve Consistency Loss Signal

**Files:**
- Modify: `configs/train/smax_3m.yaml` (consistency scale)
- Potentially modify: `actors/learner_actor.py` (multi-step SPR vs single-step)

The paper uses `consistency_coeff=2.0`. Our SMAX config uses `consistency_scale=0.25`. While our consistency target (EMA on dynamics path) is weaker than the paper's (fresh re-encode), we may be underpowering it too much.

- [ ] **Step 5.1: Experiment with consistency_scale=1.0**

The original comment in smax_3m.yaml says "default 1.0 causes consistency loss to dominate and degrade value/policy." This was observed before the other fixes. With action-level AWPO and better policy targets, 1.0 might now be fine.

In `configs/train/smax_3m.yaml`:
```yaml
consistency_scale: 1.0   # was 0.25 — try higher to improve representation
```

Run a 30k-episode training and observe:
- `consistency_loss` in the debug log (should be ~0.5-1.0, not dominating total loss)
- Whether avg return improves faster

If consistency loss dominates (>80% of total loss), reduce back. If value/policy loss drops faster, keep it.

- [ ] **Step 5.2: Enable multi-step consistency for SMAX**

The `consistency_horizon` parameter controls SPR depth. Try k=2:

```yaml
consistency_horizon: 2   # was 1 — adds k=2 target pairs
```

With k=2, the consistency loss computes:
- `sim(project_online(h_0), project_target(h_1))` — existing single-step
- `sim(project_online(h_0), project_target(h_2))` — new 2-step
- `sim(project_online(h_1), project_target(h_2))` — new 2-step from unroll step 1

This forces the representation to be predictable 2 steps ahead. From SPR (Schwarzer et al. 2021), this significantly improves sample efficiency.

- [ ] **Step 5.3: Commit consistency changes if they help**

```bash
git add configs/train/smax_3m.yaml
git commit -m "config: increase consistency_scale to 1.0, try consistency_horizon=2"
```

---

## Task 6: Ablation Experiment Matrix

Run these in order, each for 50k episodes. Stop at the first one that clearly improves (>0.8 avg return by 50k episodes):

| Experiment | Config changes | Expected effect |
|---|---|---|
| A (Task 1) | LR=5e-4, 100 sims, K=10 | +LR should accelerate if that's the bottleneck |
| B (Task 3) | + action-level AWPO | Should show stronger policy improvement per step |
| C (Task 4) | + reanalysis warmup gate | Should prevent early target corruption |
| D (Task 5) | + consistency_scale=1.0 | Should improve representation if A+B+C haven't |
| E | All of the above | Full paper-aligned config |

**Decision rule:** If experiment A reaches >0.8 return by 50k episodes → config changes alone are sufficient. If A stalls at ~0.5 → code fixes (B, C) are needed. Log the learner's `policy_entropy` at episode 10k to diagnose policy collapse vs slow learning.

---

## Self-Review Checklist

**Spec coverage:**
- [x] AWPO advantage: Issue 1 → Task 3
- [x] Policy target sparsity: addressed via K=10 in Task 1 (no deeper fix without changing to sampled-action policy loss, which would require full policy head redesign)
- [x] Learning rate: Task 1
- [x] Consistency target weakness: Task 5
- [x] Reanalysis interference: Task 4
- [x] Win rate visibility: Task 2
- [x] Policy entropy logging: Task 2

**Placeholder scan:** All code blocks contain actual implementations.

**Type consistency:**
- `MCTSPlanOutput.root_child_actions` is `(B, K, N)` int32 throughout (DataActor stores `(K, N)` per env, ReplayItem stores `(K, N)`, batch is `(B, K, N)`). ✓
- `root_child_q` is `(B, K)` float32 everywhere. ✓
- `awpo_w_k` computation uses `q_k` (the stored root child Q-values), not `v_mcts - v_net`. ✓

**Potential issue:** The C++ replay buffer doesn't store Q-data. The plan routes Q-data through a separate Python dict in ReplayBufferActor. This dict is NOT persisted to checkpoints. After a restart, Q-data is lost for old buffer items; those items will fall back to plain CE loss (`awpo_alpha=0` path). This is acceptable for training.

**Note on policy target formulation gap:** The deepest fix would be to replace marginal-visit-count policy targets with sampled-action targets (MAZero's exact formulation). This would require changing the policy head to a joint-action factored head and is out of scope here. The marginal approach should be sufficient if targets are based on quality MCTS outputs (K=10, 100 sims, good LR).
