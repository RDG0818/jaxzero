# MAZero JAX Rewrite — Design Spec
**Date:** 2026-04-28  
**Status:** Approved  
**Reference:** MAZero (ICLR 2024), `/home/ryan/Repos/MAZero/` (PyTorch original), `/home/ryan/Repos/sequential-muzero/` (prior JAX attempt)

---

## 1. Goal

Faithful JAX/Flax rewrite of MAZero that trains successfully on SMAX 3m (JaxMARL), matching the win-rate trajectory reported in the paper. The prior attempt (`sequential-muzero`) failed primarily due to:

1. Wrong MCTS algorithm — used `mctx.gumbel_muzero_policy` instead of Sampled MCTS + OS(λ)
2. Wrong hyperparameters — K=5/N=50 vs paper's K=10/N=100; lr=5e-4 vs 1e-4; stacked_obs=1 vs 4
3. `communication_net` placed outside dynamics instead of inside (misread of paper Eq. 4)
4. No positional encoding in transformer

---

## 2. Constraints

- JAX/Flax throughout; no PyTorch
- Python/numpy tree for MCTS (not mctx); model inference calls are JIT'd
- JaxMARL SMAX environment (not original SMAC); verified to match SMAC obs/action/reward semantics
- Sync serial training loop first; async Ray later
- Reanalyze flag-controlled (`use_reanalyze: bool`)
- Policy loss: AWPO sharp (`PG_type=sharp`)
- Clean repo at `jaxzero/`; reference sequential-muzero for env wrapper + replay buffer only

---

## 3. File Structure

```
jaxzero/
├── config.py                   # frozen dataclasses for all hyperparams
├── main.py                     # entry point (argparse → config → train)
├── model/
│   ├── networks.py             # RepNet, DynNet, PredNet, ProjNet, MAMuZeroNet
│   └── transforms.py           # scalar↔support, h/inv_h, φ/φ_inv
├── mcts/
│   └── sampled_mcts.py         # SampledMCTS: Python tree + OS(λ) + batch search
├── envs/
│   ├── base.py                 # EnvWrapper ABC: reset/step/obs_size/action_size/legal_actions
│   ├── smax_wrapper.py         # SMAX + obs stacking + legal actions
│   └── mpe_wrapper.py          # MPE (from sequential-muzero) — sanity check env
├── game.py                     # GameHistory: trajectory storage
├── replay_buffer.py            # PrioritizedReplayBuffer
├── reanalyze.py                # ReanalyzeWorker (flag-controlled)
├── train.py                    # sync serial loop + update_weights (JAX grad)
├── eval.py                     # evaluation: run N episodes, log win_rate + actions
├── CLAUDE.md                   # project context for future sessions
└── tests/
    ├── test_model.py
    ├── test_mcts.py
    ├── test_env.py
    ├── test_replay_buffer.py
    └── test_train.py
```

---

## 4. Hyperparameters (from paper Table 1)

| Parameter | Value |
|-----------|-------|
| Stacked observations | 4 |
| Discount γ | 0.99 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Adam ε | 1e-5 |
| Weight decay | 0 |
| Max gradient norm | 5 |
| Priority α | 0.6 |
| Priority β | 0.4 → 1.0 |
| Min replay size | 300 |
| Target model update interval | 200 |
| Unroll steps K | 5 |
| TD steps n | 5 |
| MCTS sampled actions K | 10 |
| MCTS simulations N | 100 |
| OS(λ) quantile ρ | 0.75 |
| OS(λ) decay λ | 0.8 |
| AWPO temperature α | 3.0 |
| Hidden state size | 128 |
| Representation layers | [128, 128] |
| Dynamics layers | [128, 128] |
| Reward layers | [32] |
| Value layers | [32] |
| Policy layers | [32] |
| Transformer layers | 3 |
| Transformer dropout | 0.1 |
| Value/reward support | [-5, 5] (11 bins) |
| Evaluation episodes | 32 |

---

## 5. Model Design (`model/networks.py`)

### Hidden state convention
`(B, N, D)` throughout all networks. Reshape to `(B, N*D)` only at centralized heads (value, reward).

### MLP helper
Each linear layer followed by ReLU + LayerNorm. Output layer: zero-initialized weights and bias (matches MAZero `use_value_out=True`).

### RepresentationNetwork
```
LayerNorm(obs)  →  MLP([128, 128], out=128)
Input:  (B*N, obs_dim)   [reshape before, reshape after]
Output: (B, N, 128)
```

### CommunicationNetwork (`eθ`)
Transformer encoder, 3 layers, 128 hidden, 8 heads, dropout=0.1.  
**Positional encoding** added to distinguish agents (learned: `nn.Embed(num_agents, D)` added to input).  
Input: `(B, N, D+A)` — concatenated state-action pairs.  
Output: `(B, N, D)` — per-agent communication features.  
Called **inside DynamicsNetwork** per dynamics step (not pre-search).

### DynamicsNetwork (`gθ`)
```
ha = concat(hidden, action_onehot)          # (B, N, D+A)
attn = CommunicationNet(ha)                  # (B, N, D)
dyn_input = concat(hidden, action_onehot, attn)  # (B, N, 2D+A)
next_hidden = MLP([128,128], 128)(dyn_input.reshape(B*N,-1)).reshape(B,N,D)
next_hidden = next_hidden + hidden           # residual connection
reward_input = concat(next_hidden, action_onehot).reshape(B, -1)  # (B, N*(D+A))
reward_logits = MLP([32], 11)(reward_input)  # (B, 11)
```

### PredictionNetwork (`fθ`)
```
# Value (centralized)
value_logits = MLP([32], 11)(hidden.reshape(B, N*D))      # (B, 11)
# Policy (decentralized, parameter shared)
policy_logits = MLP([32], A)(hidden.reshape(B*N, D)).reshape(B, N, A)
```

### ProjectionNetwork (SimSiam, consistency loss only)
```
proj = LayerNorm(MLP([128], 128)(hidden.reshape(B, N*D)))  # (B, 128)
pred = MLP([64], 128)(proj)                                 # online branch only
```

### MAMuZeroNet (`__call__` = initial_inference)
- `__call__`: obs → RepNet → PredNet. Dummy forward of all subnetworks when `is_initializing()`.
- `recurrent_inference(hidden, actions)`: DynNet → PredNet.
- `project_online(hidden)`: proj + pred head (online branch).
- `project_target(hidden)`: proj only (target branch, called with EMA/target params).

### Scalar ↔ Support transforms (`model/transforms.py`)
```python
h(x)     = sign(x) * (sqrt(|x|+1) - 1) + 0.001*x    # invertible transform
inv_h(x) = inverse of h
φ(x)     = categorical encoding onto support bins       # scalar → (B, 11)
φ_inv(x) = dot(softmax(x), support_range)              # (B, 11) → scalar
```

---

## 6. Sampled MCTS (`mcts/sampled_mcts.py`)

### Algorithm
Faithful Python reimplementation of `cytree.Tree_batch`. The outer simulation loop is Python; model inference (`recurrent_inference`) is JAX-JIT'd.

### Tree data structure
Pre-allocated numpy arrays indexed by `(batch_idx, node_idx)`:
```python
visit_count:        (B, max_nodes, K)     # visits per sampled action
q_value:            (B, max_nodes, K)     # OS(λ) advantage estimate
reward:             (B, max_nodes, K)     # immediate reward per action
sampled_actions:    (B, max_nodes, K, N)  # joint actions (N agents each)
prior_policy:       (B, max_nodes, K)     # π(a) from prediction net
beta:               (B, max_nodes, K)     # sampling dist β(a)
parent:             (B, max_nodes)
parent_action_idx:  (B, max_nodes)
depth:              (B, max_nodes)
hidden_pool:        list of (B, N, D) arrays, one per simulation step
```
`max_nodes = num_simulations + 1`.

### Per-simulation steps

**Prepare root (once before loop):**
```
β = softmax(π)^(1/τ) masked by legal_actions
β_with_noise = (1-ε)·β + ε·Dirichlet(α)   # exploration
Sample K joint actions from β_with_noise
Set root node: prior=π, beta=β, sampled_actions=K_actions, root_value=v
```

**Selection:**
```
from root, traverse:
  a* = argmax_{a ∈ T(s)} A^ρ_λ(s,a) + P(s,a)·(β̂(a)/β(a))·√(ΣN(s,b))/(1+N(s,a))·c(s)
  where c(s) = pb_c_init + log((ΣN(s,b)+pb_c_base+1)/pb_c_base)
until reaching unexpanded leaf
```

**Expansion:**
```
call model.recurrent_inference(hidden[parent], action*)  [JIT'd]
Sample K new joint actions from β = softmax(policy)^(1/τ)
Add new node with: hidden stored in hidden_pool, prior=π, beta=β
```

**Backup (OS(λ)):**
```
Walk path from leaf to root.
For each node s on path:
  Collect U_d(s) for d=1..max_depth:
    U_d(s) = {Σ_{k<d} γ^k r_k + γ^d v(s')} for all s' at depth dep(s)+d
  U^ρ_d(s) = top ceil((1-ρ)|U_d|) values in U_d(s)
  V^ρ_λ(s) = Σ_d λ^d·mean(U^ρ_d) / Σ_d λ^d·|U^ρ_d|
  A^ρ_λ(s, a_i) = r(s,a_i) + γ·V^ρ_λ(child_i) - v(s)
  N(s, a_i) += 1
```

### Outputs (SearchOutput namedtuple)
```python
root_value:          (B,)        # V^ρ_λ of root
sampled_actions:     list[B] of (K_i, N)   # variable K due to deduplication
sampled_visit_counts:list[B] of (K_i,)
sampled_qvalues:     list[B] of (K_i,)     # A^ρ_λ per sampled action
sampled_imp_ratio:   list[B] of (K_i,)     # π(a)/β(a)
```

### Legal action masking
- Apply mask before β sampling: `β *= legal_actions; β += legal_actions*1e-4; β /= β.sum()`
- Apply mask in UCB selection: only consider actions in `T(s) ∩ legal_actions`
- Dead agents (all actions illegal except no-op): force action index 0

---

## 7. Environment (`envs/`)

### Base interface (`envs/base.py`)
```python
class EnvWrapper(ABC):
    obs_size: int
    action_space_size: int
    num_agents: int

    def reset(self, rng_key) -> tuple[np.ndarray, Any]:
        # returns (obs: (N, obs_size), state)

    def step(self, rng_key, state, actions: np.ndarray) -> tuple[np.ndarray, Any, float, bool, bool]:
        # returns (obs, state, reward, done, won)

    def get_legal_actions(self, state) -> np.ndarray:
        # returns (N, A) bool mask
```

### SMAX wrapper (`envs/smax_wrapper.py`)
- `HeuristicEnemySMAX` via JaxMARL
- Obs stacking: maintain rolling window of 4 obs per agent; return `(N, 4*obs_size)`
- Legal actions extracted from SMAX state per step
- Win detection: `done & reward > 0.5` (SMAX win bonus = 1.0)
- Reward: team reward scalar (single ally's reward, not summed)
- Episode termination: `done["__all__"]` only

### MPE wrapper (`envs/mpe_wrapper.py`)
- Adapted from `sequential-muzero/envs/mpe_env_wrapper.py`
- Uniform legal actions (all actions always valid)
- Used for quick sanity checks that learning is happening before running SMAX

---

## 8. GameHistory (`game.py`)

Stores one complete trajectory. Key fields:
```python
obs_history:       list[ndarray(N, obs_dim)]  # length T + stacked_obs init frames
actions:           ndarray(T, N)
rewards:           ndarray(T,)
legal_actions:     list[ndarray(N, A)]         # length T
root_values:       ndarray(T,)                 # MCTS root value
pred_values:       ndarray(T,)                 # model predicted value (for priority)
sampled_actions:   ndarray(T, K, N)
sampled_policies:  ndarray(T, K)               # visit_count / N_sims
sampled_qvalues:   ndarray(T, K)               # A^ρ_λ advantages
sampled_masks:     ndarray(T, K)               # padding validity mask
model_indices:     ndarray(T,)                 # step when model generated this data
```

`obs(t, unroll_steps, padding=False)` method returns stacked obs window for training.

---

## 9. PrioritizedReplayBuffer (`replay_buffer.py`)

- Stores `GameHistory` objects (not individual transitions)
- Sampling: sample trajectory, then sample position within trajectory
- Priority = `|pred_value - target_value| + ε` (updated after each training step)
- α=0.6, β annealed 0.4→1.0 over training
- `prepare_batch_context(batch_size, beta)` → `(games, positions, indices, weights)`
- `can_sample(batch_size)` → bool (checks `size >= max(batch_size, min_replay_size)`)

---

## 10. Reanalyze Worker (`reanalyze.py`)

```python
class ReanalyzeWorker:
    use_reanalyze: bool   # from config

    def make_batch(self, buffer_context, params) -> BatchData:
        # (1) unpack game_lst, positions, indices, weights from buffer_context
        # (2) build obs_batch, action_batch, mask_batch
        # (3) if use_reanalyze:
        #       run MCTS on obs at each position → fresh sampled_actions, policies, qvalues
        #     else:
        #       use stored targets from GameHistory
        # (4) compute advantages: adv = qvalues - pred_values; normalize 0-mean 1-std
        # (5) reshape to (B, K+1, ...) for training
        # returns BatchData namedtuple
```

`revisit_policy_search_rate=0.99`: only `ceil(B * 0.99)` samples get reanalyzed (rest use stored targets).

TD-step value targets: n-step return with optional shorter horizon for off-policy correction (auto_td_steps).

---

## 11. Training Loop (`train.py`)

### `update_weights` (JIT'd JAX function)
Config is captured via closure or `functools.partial`; not passed as traced argument to avoid JIT recompilation.
```python
# update_weights = functools.partial(_update_weights, config=config)
@jax.jit
def update_weights(params, batch):  # config captured in closure
    # unpack batch: obs, actions, masks, target_rewards, target_values,
    #               sampled_actions, sampled_policies, sampled_advantages, sampled_masks

    # initial inference
    out = model.apply(params, obs[:,:,0:stacks])
    value_loss   = categorical_cross_entropy(out.value_logits, target_value_phi[:,0])
    reward_loss  = zeros  # no reward at step 0
    policy_loss  = awpo_sharp_loss(out.policy_logits, sampled_actions[:,0],
                                    sampled_policies[:,0], sampled_advantages[:,0],
                                    sampled_masks[:,0], config.awpo_alpha)

    # unroll K steps
    hidden = out.hidden_state
    for k in range(1, unroll_steps+1):
        # half-gradient: only 0.5 of gradient flows back through hidden into prior step
        hidden = hidden / 2 + jax.lax.stop_gradient(hidden / 2)
        out = model.apply(params, hidden, actions[:,k-1], method='recurrent_inference')
        reward_loss += categorical_cross_entropy(out.reward_logits, target_reward_phi[:,k-1])
        value_loss  += categorical_cross_entropy(out.value_logits,  target_value_phi[:,k])
        policy_loss += awpo_sharp_loss(...)
        if use_consistency:
            consistency_loss += simsiam_loss(...)

    total = reward_loss + 0.25*value_loss + policy_loss + consistency_scale*consistency_loss
    return (weights * total).mean() / unroll_steps
```

### AWPO sharp loss
```python
def awpo_sharp_loss(policy_logits, sampled_actions, visit_counts, advantages, masks, alpha):
    log_probs = log_softmax(policy_logits)  # (B, N, A)
    # gather log prob of each sampled joint action (sum over agents = joint log prob)
    action_log_probs = gather_and_sum(log_probs, sampled_actions)  # (B, K)
    adv_weights = exp(advantages / alpha)   # (B, K)
    loss = -(action_log_probs * visit_counts * adv_weights * masks).sum(dim=-1)
    return loss  # (B,)
```

### Sync serial loop
```python
def train(config):
    params = model.init(...)
    opt_state = optimizer.init(params)
    replay_buffer = PrioritizedReplayBuffer(config)
    reanalyze_worker = ReanalyzeWorker(config)
    env = build_env(config)

    step = 0
    while step < config.training_steps:
        # collect one episode
        game = collect_episode(env, params, config)
        replay_buffer.add(game)

        if not replay_buffer.can_sample(config.batch_size):
            continue

        # prepare batch (with or without reanalyze)
        beta = beta_schedule(step)
        buffer_ctx = replay_buffer.prepare_batch_context(config.batch_size, beta)
        batch = reanalyze_worker.make_batch(buffer_ctx, params)

        # gradient step
        loss, grads = jax.value_and_grad(update_weights)(params, batch, config)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # update priorities
        replay_buffer.update_priorities(batch.indices, batch.new_priorities)

        # update target model every target_model_interval steps
        if step % config.target_model_interval == 0:
            target_params = params  # (EMA or hard copy)

        # eval + logging
        if step % config.eval_interval == 0:
            eval_results = evaluate(env, params, config)
            log(step, loss, eval_results)  # includes action log per episode

        step += 1
```

---

## 12. Config (`config.py`)

```python
@dataclass(frozen=True)
class MAZeroConfig:
    # Environment
    env_name: str = "3m"
    num_agents: int = 3
    obs_size: int = 0           # sentinel; set via dataclasses.replace(config, obs_size=env.obs_size) at startup
    action_space_size: int = 0  # sentinel; set at startup
    stacked_observations: int = 4
    max_episode_steps: int = 100

    # Model
    hidden_state_size: int = 128
    fc_representation_layers: tuple = (128, 128)
    fc_dynamic_layers: tuple = (128, 128)
    fc_reward_layers: tuple = (32,)
    fc_value_layers: tuple = (32,)
    fc_policy_layers: tuple = (32,)
    attention_layers: int = 3
    attention_heads: int = 8
    dropout_rate: float = 0.1
    value_support_size: int = 5   # range [-5, 5] → 11 bins
    reward_support_size: int = 5

    # MCTS
    num_simulations: int = 100
    sampled_action_times: int = 10
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    mcts_rho: float = 0.75
    mcts_lambda: float = 0.8
    tree_value_stat_delta_lb: float = 0.01

    # Training
    training_steps: int = 100_000
    batch_size: int = 256
    unroll_steps: int = 5
    td_steps: int = 5
    discount: float = 0.99
    learning_rate: float = 1e-4
    adam_eps: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 5.0
    awpo_alpha: float = 3.0
    reward_loss_coeff: float = 1.0
    value_loss_coeff: float = 0.25
    policy_loss_coeff: float = 1.0
    consistency_coeff: float = 2.0

    # Replay
    replay_buffer_size: int = 100_000
    min_replay_size: int = 300
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    target_model_interval: int = 200

    # Reanalyze
    use_reanalyze: bool = True
    revisit_policy_search_rate: float = 0.99

    # Logging / eval
    eval_interval: int = 1000
    eval_episodes: int = 32
    log_interval: int = 100
    seed: int = 0
```

---

## 13. Testing Strategy

Each module tested in isolation before integration. Tests use `pytest`.

| Module | What to test |
|--------|-------------|
| `model/transforms.py` | `h`/`inv_h` round-trip, `φ`/`φ_inv` round-trip, edge cases at support boundaries |
| `model/networks.py` | output shapes for all methods, `is_initializing()` includes all params, residual connection in dynamics |
| `mcts/sampled_mcts.py` | UCB selection formula vs hand-computed, OS(λ) backup vs hand-computed example, legal action masking, batch consistency (all B roots give independent results) |
| `envs/smax_wrapper.py` | obs/action/reward shapes over full episode, legal action masks always valid, win detection, obs stack initialization |
| `game.py` | `obs()` method returns correct stacked window, padding at boundaries |
| `replay_buffer.py` | priority sampling distribution, weight calculation, `prepare_batch_context` shapes |
| `reanalyze.py` | batch shapes match expected `(B, K+1, ...)`, `use_reanalyze=False` returns stored targets unchanged |
| `train.py` | loss decreases on overfit (single batch repeated), gradient norms within bounds |

---

## 14. Visualization & Logging

Each eval run logs:
- Win rate, avg return, avg episode length
- Per-step action histogram (which actions agents chose most)
- Sample trajectory: timestep-by-timestep `[agent_0_action, agent_1_action, ..., reward]`
- Loss components: reward_loss, value_loss, policy_loss, consistency_loss

This directly addresses the "can't tell what agents are doing" problem in the prior implementation.

---

## 15. Implementation Order

1. `config.py` + `model/transforms.py` + tests
2. `model/networks.py` + tests
3. `mcts/sampled_mcts.py` + tests  ← hardest, most critical
4. `envs/base.py` + `envs/smax_wrapper.py` + `envs/mpe_wrapper.py` + tests
5. `game.py` + `replay_buffer.py` + tests
6. `reanalyze.py` + tests
7. `train.py` + `eval.py` + integration test (MPE first, then SMAX 3m)
8. `CLAUDE.md` with project context

Each step: implement → test → verify before moving to next.
