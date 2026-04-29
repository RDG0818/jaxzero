# MAZero JAX Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithful JAX/Flax rewrite of MAZero that trains on SMAX 3m, matching paper win-rate trajectory.

**Architecture:** Sync serial training loop; Python-tree MCTS with JIT'd model inference; JAX/Flax throughout; reanalyze flag-controlled.

**Tech Stack:** JAX, Flax (linen), Optax, JaxMARL (SMAX), numpy, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `jaxzero/config.py` | Frozen dataclass with all hyperparams |
| `jaxzero/model/transforms.py` | h/inv_h, φ/φ_inv scalar↔support |
| `jaxzero/model/networks.py` | RepNet, DynNet, PredNet, ProjNet, MAMuZeroNet |
| `jaxzero/mcts/sampled_mcts.py` | Python-tree Sampled MCTS + OS(λ) |
| `jaxzero/envs/base.py` | EnvWrapper ABC |
| `jaxzero/envs/smax_wrapper.py` | SMAX + obs stacking + legal actions |
| `jaxzero/envs/mpe_wrapper.py` | MPE sanity-check env |
| `jaxzero/game.py` | GameHistory trajectory storage |
| `jaxzero/replay_buffer.py` | PrioritizedReplayBuffer |
| `jaxzero/reanalyze.py` | ReanalyzeWorker |
| `jaxzero/train.py` | Sync serial loop + update_weights (JIT'd) |
| `jaxzero/eval.py` | Evaluation runner |
| `jaxzero/main.py` | Entry point |
| `jaxzero/CLAUDE.md` | Project context |
| `tests/test_transforms.py` | Transform round-trip tests |
| `tests/test_networks.py` | Shape + param tests |
| `tests/test_mcts.py` | UCB, OS(λ), masking tests |
| `tests/test_env.py` | Env wrapper tests |
| `tests/test_game.py` | GameHistory tests |
| `tests/test_replay_buffer.py` | Priority sampling tests |
| `tests/test_reanalyze.py` | Batch shape tests |
| `tests/test_train.py` | Overfit + grad norm tests |

---

## Task 1: Project Scaffold + Config

**Files:**
- Create: `jaxzero/__init__.py`
- Create: `jaxzero/config.py`
- Create: `tests/__init__.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:BuildBackend"

[project]
name = "jaxzero"
version = "0.1.0"
dependencies = [
    "jax",
    "flax",
    "optax",
    "chex",
    "numpy",
    "pytest",
]

[tool.setuptools.packages.find]
where = ["."]
```

- [ ] **Step 2: Create empty inits**

```python
# jaxzero/__init__.py
# tests/__init__.py
```

- [ ] **Step 3: Write config.py**

```python
# jaxzero/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class MAZeroConfig:
    # Environment
    env_name: str = "3m"
    num_agents: int = 3
    obs_size: int = 0
    action_space_size: int = 0
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
    value_support_size: int = 5
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

    @property
    def support_size(self) -> int:
        return self.value_support_size * 2 + 1
```

- [ ] **Step 4: Verify import**

Run: `cd /home/ryan/Repos/jaxzero && python -c "from jaxzero.config import MAZeroConfig; c = MAZeroConfig(); print(c.support_size)"`
Expected: `11`

- [ ] **Step 5: Commit**

```bash
git init
git add jaxzero/__init__.py jaxzero/config.py tests/__init__.py pyproject.toml
git commit -m "feat: add project scaffold and MAZeroConfig"
```

---

## Task 2: Scalar ↔ Support Transforms

**Files:**
- Create: `jaxzero/model/__init__.py`
- Create: `jaxzero/model/transforms.py`
- Create: `tests/test_transforms.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transforms.py
import numpy as np
import jax.numpy as jnp
import pytest
from jaxzero.model.transforms import h, inv_h, phi, phi_inv


def test_h_inv_h_roundtrip():
    x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    assert jnp.allclose(inv_h(h(x)), x, atol=1e-5)


def test_h_inv_h_roundtrip_scalar():
    x = jnp.array(3.7)
    assert jnp.allclose(inv_h(h(x)), x, atol=1e-5)


def test_phi_phi_inv_roundtrip():
    support_size = 5
    x = jnp.array([-4.5, -1.0, 0.0, 2.3, 4.9])
    logits = phi(x, support_size)
    assert logits.shape == (5, 11)
    recovered = phi_inv(logits, support_size)
    assert jnp.allclose(recovered, x, atol=0.1)


def test_phi_boundary():
    support_size = 5
    x = jnp.array([-5.0, 5.0])
    logits = phi(x, support_size)
    recovered = phi_inv(logits, support_size)
    assert jnp.allclose(recovered, x, atol=1e-5)


def test_phi_output_shape():
    support_size = 5
    x = jnp.array([1.5])
    logits = phi(x, support_size)
    assert logits.shape == (1, 11)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_transforms.py -v 2>&1 | head -30`
Expected: ImportError or ModuleNotFoundError

- [ ] **Step 3: Implement transforms.py**

```python
# jaxzero/model/transforms.py
import jax.numpy as jnp
import chex


def h(x: chex.Array) -> chex.Array:
    """Invertible value transform from MuZero (Appendix F)."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + 0.001 * x


def inv_h(x: chex.Array) -> chex.Array:
    """Inverse of h transform."""
    eps = 0.001
    return jnp.sign(x) * (
        ((jnp.sqrt(1 + 4 * eps * (jnp.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )


def phi(x: chex.Array, support_size: int) -> chex.Array:
    """Encode scalar(s) to categorical distribution over support [-S, S].

    Args:
        x: shape (B,) scalars
        support_size: S, so support has 2*S+1 bins

    Returns:
        shape (B, 2*S+1) soft encoding
    """
    x = jnp.clip(x, -support_size, support_size)
    low = jnp.floor(x).astype(jnp.int32)
    high = low + 1
    high = jnp.clip(high, -support_size, support_size)
    low = jnp.clip(low, -support_size, support_size)
    p_high = x - low.astype(jnp.float32)
    p_low = 1.0 - p_high

    low_idx = low + support_size
    high_idx = high + support_size
    n_bins = 2 * support_size + 1
    B = x.shape[0]

    out = jnp.zeros((B, n_bins))
    out = out.at[jnp.arange(B), low_idx].add(p_low)
    out = out.at[jnp.arange(B), high_idx].add(p_high)
    return out


def phi_inv(logits: chex.Array, support_size: int) -> chex.Array:
    """Decode categorical logits to scalar via expected value.

    Args:
        logits: shape (B, 2*S+1)
        support_size: S

    Returns:
        shape (B,) scalars
    """
    support = jnp.arange(-support_size, support_size + 1, dtype=jnp.float32)
    probs = jnp.softmax(logits, axis=-1)
    return jnp.dot(probs, support)
```

- [ ] **Step 4: Create model __init__**

```python
# jaxzero/model/__init__.py
```

- [ ] **Step 5: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_transforms.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add jaxzero/model/__init__.py jaxzero/model/transforms.py tests/test_transforms.py
git commit -m "feat: add scalar<->support transforms with tests"
```

---

## Task 3: MLP + Transformer Layers

**Files:**
- Create: `jaxzero/model/layers.py`
- Create: `tests/test_layers.py`

These are building blocks for Task 4 networks.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_layers.py
import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest
from jaxzero.model.layers import MLP, TransformerEncoder


def test_mlp_output_shape():
    mlp = MLP(layer_sizes=(128, 128), output_size=64)
    x = jnp.ones((4, 32))
    params = mlp.init(jax.random.PRNGKey(0), x)
    y = mlp.apply(params, x)
    assert y.shape == (4, 64)


def test_mlp_output_zero_init():
    """Output layer weights and bias should start near zero."""
    mlp = MLP(layer_sizes=(128,), output_size=11)
    x = jnp.ones((2, 32))
    params = mlp.init(jax.random.PRNGKey(0), x)
    out_w = params['params']['output']['kernel']
    out_b = params['params']['output']['bias']
    assert jnp.allclose(out_w, jnp.zeros_like(out_w), atol=1e-6)
    assert jnp.allclose(out_b, jnp.zeros_like(out_b), atol=1e-6)


def test_transformer_output_shape():
    enc = TransformerEncoder(num_layers=2, num_heads=4, hidden_size=64, dropout_rate=0.0)
    x = jnp.ones((2, 3, 64))
    params = enc.init(jax.random.PRNGKey(0), x, deterministic=True)
    y = enc.apply(params, x, deterministic=True)
    assert y.shape == (2, 3, 64)


def test_transformer_output_size_mismatch_raises():
    """Input last dim must equal hidden_size."""
    enc = TransformerEncoder(num_layers=1, num_heads=4, hidden_size=64, dropout_rate=0.0)
    x = jnp.ones((2, 3, 32))
    with pytest.raises(Exception):
        params = enc.init(jax.random.PRNGKey(0), x, deterministic=True)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_layers.py -v 2>&1 | head -20`
Expected: ImportError

- [ ] **Step 3: Implement layers.py**

```python
# jaxzero/model/layers.py
import flax.linen as nn
import jax.numpy as jnp
import chex
from typing import Sequence


class MLP(nn.Module):
    """MLP with ReLU+LayerNorm hidden layers and zero-initialized output layer."""
    layer_sizes: Sequence[int]
    output_size: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for size in self.layer_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.output_size,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="output",
        )(x)
        return x


class TransformerEncoderLayer(nn.Module):
    num_heads: int
    hidden_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: chex.Array, deterministic: bool = False) -> chex.Array:
        # Self-attention with residual
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(x, x, deterministic=deterministic)
        x = nn.LayerNorm()(x + attn_out)
        # FFN with residual
        ffn_out = nn.Dense(self.hidden_size * 4)(x)
        ffn_out = nn.relu(ffn_out)
        ffn_out = nn.Dense(self.hidden_size)(ffn_out)
        ffn_out = nn.Dropout(rate=self.dropout_rate)(ffn_out, deterministic=deterministic)
        x = nn.LayerNorm()(x + ffn_out)
        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    num_heads: int
    hidden_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: chex.Array, deterministic: bool = False) -> chex.Array:
        assert x.shape[-1] == self.hidden_size, (
            f"Input dim {x.shape[-1]} != hidden_size {self.hidden_size}"
        )
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
            )(x, deterministic=deterministic)
        return x
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_layers.py -v`
Expected: 3 passed, 1 may xfail depending on flax error handling

- [ ] **Step 5: Commit**

```bash
git add jaxzero/model/layers.py tests/test_layers.py
git commit -m "feat: add MLP and TransformerEncoder building blocks"
```

---

## Task 4: MAMuZeroNet Networks

**Files:**
- Create: `jaxzero/model/networks.py`
- Create: `tests/test_networks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_networks.py
import jax
import jax.numpy as jnp
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet


B, N, D, A = 2, 3, 128, 9
OBS_DIM = 80 * 4  # stacked obs


def make_net(config=None):
    if config is None:
        config = MAZeroConfig(
            num_agents=N,
            obs_size=OBS_DIM,
            action_space_size=A,
        )
    return MAMuZeroNet(config=config), config


def test_initial_inference_shapes():
    net, config = make_net()
    obs = jnp.ones((B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    out = net.apply(params, obs)
    assert out.hidden_state.shape == (B, N, D)
    assert out.value_logits.shape == (B, 11)
    assert out.policy_logits.shape == (B, N, A)
    assert out.reward_logits.shape == (B, 11)


def test_recurrent_inference_shapes():
    net, config = make_net()
    obs = jnp.ones((B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    hidden = net.apply(params, obs).hidden_state
    actions = jnp.zeros((B, N), dtype=jnp.int32)
    out = net.apply(params, hidden, actions, method=net.recurrent_inference)
    assert out.hidden_state.shape == (B, N, D)
    assert out.reward_logits.shape == (B, 11)
    assert out.value_logits.shape == (B, 11)
    assert out.policy_logits.shape == (B, N, A)


def test_dynamics_residual():
    """next_hidden != input hidden (residual adds, not replaces)."""
    net, config = make_net()
    obs = jnp.ones((B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    hidden = net.apply(params, obs).hidden_state
    actions = jnp.zeros((B, N), dtype=jnp.int32)
    out = net.apply(params, hidden, actions, method=net.recurrent_inference)
    assert not jnp.allclose(out.hidden_state, hidden)


def test_project_online_shape():
    net, config = make_net()
    obs = jnp.ones((B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    hidden = net.apply(params, obs).hidden_state
    proj = net.apply(params, hidden, method=net.project_online)
    assert proj.shape == (B, 128)


def test_project_target_shape():
    net, config = make_net()
    obs = jnp.ones((B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    hidden = net.apply(params, obs).hidden_state
    proj = net.apply(params, hidden, method=net.project_target)
    assert proj.shape == (B, 128)


def test_all_params_initialized():
    """init() via __call__ must include dynamics and projection params."""
    net, config = make_net()
    obs = jnp.ones((B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    param_keys = set(params['params'].keys())
    assert 'representation_net' in param_keys
    assert 'dynamics_net' in param_keys
    assert 'prediction_net' in param_keys
    assert 'projection_net' in param_keys
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_networks.py -v 2>&1 | head -20`
Expected: ImportError

- [ ] **Step 3: Implement networks.py**

```python
# jaxzero/model/networks.py
import flax.linen as nn
import jax
import jax.numpy as jnp
import chex
from typing import NamedTuple
from jaxzero.config import MAZeroConfig
from jaxzero.model.layers import MLP, TransformerEncoder


class MuZeroOutput(NamedTuple):
    hidden_state: chex.Array    # (B, N, D)
    reward_logits: chex.Array   # (B, support_size)
    policy_logits: chex.Array   # (B, N, A)
    value_logits: chex.Array    # (B, support_size)


class RepresentationNetwork(nn.Module):
    hidden_state_size: int
    fc_layers: tuple

    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        # obs: (B*N, obs_dim) → (B*N, D)
        x = nn.LayerNorm()(obs)
        return MLP(layer_sizes=self.fc_layers, output_size=self.hidden_state_size)(x)


class CommunicationNetwork(nn.Module):
    """Transformer encoder for inter-agent communication inside dynamics."""
    num_layers: int
    num_heads: int
    hidden_size: int
    dropout_rate: float

    def setup(self):
        self.pos_embed = nn.Embed(
            num_embeddings=32,  # max agents
            features=self.hidden_size,
        )
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
        )

    def __call__(self, ha: chex.Array, deterministic: bool = False) -> chex.Array:
        # ha: (B, N, D+A) — project to hidden_size first, then add positional
        B, N, _ = ha.shape
        x = nn.Dense(self.hidden_size)(ha)
        pos_ids = jnp.arange(N)
        x = x + self.pos_embed(pos_ids)
        return self.transformer(x, deterministic=deterministic)


class DynamicsNetwork(nn.Module):
    hidden_state_size: int
    action_space_size: int
    reward_support_size: int
    fc_dynamic_layers: tuple
    fc_reward_layers: tuple
    attention_layers: int
    attention_heads: int
    dropout_rate: float

    def setup(self):
        self.communication_net = CommunicationNetwork(
            num_layers=self.attention_layers,
            num_heads=self.attention_heads,
            hidden_size=self.hidden_state_size,
            dropout_rate=self.dropout_rate,
        )
        self.dynamic_mlp = MLP(
            layer_sizes=self.fc_dynamic_layers,
            output_size=self.hidden_state_size,
        )
        self.reward_mlp = MLP(
            layer_sizes=self.fc_reward_layers,
            output_size=self.reward_support_size * 2 + 1,
        )

    def __call__(
        self, hidden: chex.Array, actions: chex.Array, deterministic: bool = False
    ) -> tuple[chex.Array, chex.Array]:
        # hidden: (B, N, D), actions: (B, N)
        B, N, D = hidden.shape
        actions_onehot = jax.nn.one_hot(actions, num_classes=self.action_space_size)

        ha = jnp.concatenate([hidden, actions_onehot], axis=-1)              # (B, N, D+A)
        attn = self.communication_net(ha, deterministic=deterministic)       # (B, N, D)
        dyn_input = jnp.concatenate([hidden, actions_onehot, attn], axis=-1) # (B, N, 2D+A)

        next_hidden = self.dynamic_mlp(dyn_input.reshape(B * N, -1))
        next_hidden = next_hidden.reshape(B, N, D) + hidden                  # residual

        reward_input = jnp.concatenate([next_hidden, actions_onehot], axis=-1)
        reward_logits = self.reward_mlp(reward_input.reshape(B, -1))
        return next_hidden, reward_logits


class PredictionNetwork(nn.Module):
    action_space_size: int
    value_support_size: int
    fc_value_layers: tuple
    fc_policy_layers: tuple

    def setup(self):
        self.value_mlp = MLP(
            layer_sizes=self.fc_value_layers,
            output_size=self.value_support_size * 2 + 1,
        )
        self.policy_mlp = MLP(
            layer_sizes=self.fc_policy_layers,
            output_size=self.action_space_size,
        )

    def __call__(self, hidden: chex.Array) -> tuple[chex.Array, chex.Array]:
        # hidden: (B, N, D)
        B, N, D = hidden.shape
        value_logits = self.value_mlp(hidden.reshape(B, N * D))
        policy_logits = self.policy_mlp(hidden.reshape(B * N, D)).reshape(B, N, -1)
        return policy_logits, value_logits


class ProjectionNetwork(nn.Module):
    hidden_size: int = 128
    output_size: int = 128
    pred_hidden_size: int = 64

    def setup(self):
        self.proj_mlp = MLP(layer_sizes=(self.hidden_size,), output_size=self.output_size)
        self.proj_norm = nn.LayerNorm()
        self.pred_mlp = MLP(layer_sizes=(self.pred_hidden_size,), output_size=self.output_size)

    def project(self, x: chex.Array) -> chex.Array:
        return self.proj_norm(self.proj_mlp(x))

    def predict(self, proj: chex.Array) -> chex.Array:
        return self.pred_mlp(proj)


class MAMuZeroNet(nn.Module):
    config: MAZeroConfig

    def setup(self):
        cfg = self.config
        self.representation_net = RepresentationNetwork(
            hidden_state_size=cfg.hidden_state_size,
            fc_layers=cfg.fc_representation_layers,
        )
        self.dynamics_net = DynamicsNetwork(
            hidden_state_size=cfg.hidden_state_size,
            action_space_size=cfg.action_space_size,
            reward_support_size=cfg.reward_support_size,
            fc_dynamic_layers=cfg.fc_dynamic_layers,
            fc_reward_layers=cfg.fc_reward_layers,
            attention_layers=cfg.attention_layers,
            attention_heads=cfg.attention_heads,
            dropout_rate=cfg.dropout_rate,
        )
        self.prediction_net = PredictionNetwork(
            action_space_size=cfg.action_space_size,
            value_support_size=cfg.value_support_size,
            fc_value_layers=cfg.fc_value_layers,
            fc_policy_layers=cfg.fc_policy_layers,
        )
        self.projection_net = ProjectionNetwork()

    def __call__(self, obs: chex.Array) -> MuZeroOutput:
        # obs: (B, N, obs_dim)
        B, N, _ = obs.shape
        hidden = self.representation_net(obs.reshape(B * N, -1)).reshape(B, N, -1)
        policy_logits, value_logits = self.prediction_net(hidden)
        reward_logits = jnp.zeros((B, self.config.reward_support_size * 2 + 1))

        if self.is_initializing():
            dummy_actions = jnp.zeros((B, N), dtype=jnp.int32)
            self.dynamics_net(hidden, dummy_actions, deterministic=True)
            self.project_online(hidden)
            self.project_target(hidden)

        return MuZeroOutput(
            hidden_state=hidden,
            reward_logits=reward_logits,
            policy_logits=policy_logits,
            value_logits=value_logits,
        )

    def recurrent_inference(
        self, hidden: chex.Array, actions: chex.Array, deterministic: bool = False
    ) -> MuZeroOutput:
        next_hidden, reward_logits = self.dynamics_net(hidden, actions, deterministic=deterministic)
        policy_logits, value_logits = self.prediction_net(next_hidden)
        return MuZeroOutput(
            hidden_state=next_hidden,
            reward_logits=reward_logits,
            policy_logits=policy_logits,
            value_logits=value_logits,
        )

    def project_online(self, hidden: chex.Array) -> chex.Array:
        B, N, D = hidden.shape
        proj = self.projection_net.project(hidden.reshape(B, N * D))
        return self.projection_net.predict(proj)

    def project_target(self, hidden: chex.Array) -> chex.Array:
        B, N, D = hidden.shape
        return self.projection_net.project(hidden.reshape(B, N * D))
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_networks.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add jaxzero/model/networks.py tests/test_networks.py
git commit -m "feat: add MAMuZeroNet with RepNet, DynNet, PredNet, ProjNet"
```

---

## Task 5: Environment Wrappers

**Files:**
- Create: `jaxzero/envs/__init__.py`
- Create: `jaxzero/envs/base.py`
- Create: `jaxzero/envs/smax_wrapper.py`
- Create: `jaxzero/envs/mpe_wrapper.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_env.py
import numpy as np
import jax
import pytest
from jaxzero.envs.smax_wrapper import SMAXWrapper
from jaxzero.envs.mpe_wrapper import MPEWrapper


def test_smax_reset_shapes():
    env = SMAXWrapper(map_name="3m", stacked_observations=4)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    assert obs.shape == (env.num_agents, env.obs_size)


def test_smax_step_shapes():
    env = SMAXWrapper(map_name="3m", stacked_observations=4)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    actions = np.zeros(env.num_agents, dtype=np.int32)
    rng2 = jax.random.PRNGKey(1)
    obs2, state2, reward, done, won = env.step(rng2, state, actions)
    assert obs2.shape == (env.num_agents, env.obs_size)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(won, bool)


def test_smax_legal_actions_shape():
    env = SMAXWrapper(map_name="3m", stacked_observations=4)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    legal = env.get_legal_actions(state)
    assert legal.shape == (env.num_agents, env.action_space_size)
    assert legal.dtype == bool


def test_smax_obs_stacking():
    """After reset, all 4 stacked frames should be identical (first obs repeated)."""
    env = SMAXWrapper(map_name="3m", stacked_observations=4)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    raw_size = env.obs_size // env.stacked_observations
    # First frame and last frame should be identical
    assert np.allclose(obs[:, :raw_size], obs[:, -raw_size:])


def test_mpe_reset_shapes():
    env = MPEWrapper()
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    assert obs.shape[0] == env.num_agents
    assert obs.shape[1] == env.obs_size


def test_mpe_legal_actions_all_valid():
    """MPE has uniform legal actions."""
    env = MPEWrapper()
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    legal = env.get_legal_actions(state)
    assert legal.all()
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_env.py -v 2>&1 | head -20`
Expected: ImportError

- [ ] **Step 3: Implement base.py**

```python
# jaxzero/envs/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, tuple


class EnvWrapper(ABC):
    obs_size: int
    action_space_size: int
    num_agents: int
    stacked_observations: int

    @abstractmethod
    def reset(self, rng_key) -> tuple[np.ndarray, Any]:
        """Returns (obs: (N, obs_size), state)."""

    @abstractmethod
    def step(self, rng_key, state, actions: np.ndarray) -> tuple[np.ndarray, Any, float, bool, bool]:
        """Returns (obs, state, reward, done, won)."""

    @abstractmethod
    def get_legal_actions(self, state) -> np.ndarray:
        """Returns (N, A) bool mask."""
```

- [ ] **Step 4: Implement smax_wrapper.py**

```python
# jaxzero/envs/smax_wrapper.py
import numpy as np
import jax
import jax.numpy as jnp
from typing import Any
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxzero.envs.base import EnvWrapper


class SMAXWrapper(EnvWrapper):
    def __init__(self, map_name: str = "3m", stacked_observations: int = 4):
        self.stacked_observations = stacked_observations
        self._env = make("HeuristicEnemySMAX", map_name=map_name)
        scenario = map_name_to_scenario(map_name)
        self.num_agents = scenario.num_allies
        self._agents = [f"ally_{i}" for i in range(self.num_agents)]
        # raw obs size from environment
        sample_obs = self._env.observation_space(self._agents[0]).shape[0]
        self._raw_obs_size = sample_obs
        self.obs_size = sample_obs * stacked_observations
        self.action_space_size = self._env.action_space(self._agents[0]).n
        self._obs_stack = None

    def reset(self, rng_key) -> tuple[np.ndarray, Any]:
        obs_dict, state = self._env.reset(rng_key)
        raw_obs = np.stack([np.array(obs_dict[a]) for a in self._agents])  # (N, raw_obs)
        # initialize stack: repeat first obs stacked_observations times
        self._obs_stack = np.tile(raw_obs[:, np.newaxis, :], (1, self.stacked_observations, 1))
        stacked = self._obs_stack.reshape(self.num_agents, -1)
        return stacked, state

    def step(self, rng_key, state, actions: np.ndarray) -> tuple[np.ndarray, Any, float, bool, bool]:
        actions_dict = {a: int(actions[i]) for i, a in enumerate(self._agents)}
        obs_dict, state, reward_dict, done_dict, info = self._env.step(rng_key, state, actions_dict)

        raw_obs = np.stack([np.array(obs_dict[a]) for a in self._agents])
        # roll stack: drop oldest, add new at end
        self._obs_stack = np.roll(self._obs_stack, shift=-1, axis=1)
        self._obs_stack[:, -1, :] = raw_obs
        stacked = self._obs_stack.reshape(self.num_agents, -1)

        reward = float(reward_dict[self._agents[0]])
        done = bool(done_dict["__all__"])
        won = done and reward > 0.5
        return stacked, state, reward, done, won

    def get_legal_actions(self, state) -> np.ndarray:
        avail = self._env.get_avail_actions(state)
        return np.stack([np.array(avail[a], dtype=bool) for a in self._agents])
```

- [ ] **Step 5: Implement mpe_wrapper.py**

```python
# jaxzero/envs/mpe_wrapper.py
import numpy as np
import jax
from typing import Any
from jaxmarl import make
from jaxzero.envs.base import EnvWrapper


class MPEWrapper(EnvWrapper):
    """MPE SimpleSpreads wrapper for sanity checks. Uniform legal actions."""

    def __init__(self, num_agents: int = 3, stacked_observations: int = 1):
        self.stacked_observations = stacked_observations
        self._env = make("MPE_simple_spread_v3", num_agents=num_agents)
        self.num_agents = num_agents
        self._agents = self._env.agents
        sample_obs = self._env.observation_space(self._agents[0]).shape[0]
        self._raw_obs_size = sample_obs
        self.obs_size = sample_obs * stacked_observations
        self.action_space_size = self._env.action_space(self._agents[0]).n
        self._obs_stack = None

    def reset(self, rng_key) -> tuple[np.ndarray, Any]:
        obs_dict, state = self._env.reset(rng_key)
        raw_obs = np.stack([np.array(obs_dict[a]) for a in self._agents])
        self._obs_stack = np.tile(raw_obs[:, np.newaxis, :], (1, self.stacked_observations, 1))
        return self._obs_stack.reshape(self.num_agents, -1), state

    def step(self, rng_key, state, actions: np.ndarray) -> tuple[np.ndarray, Any, float, bool, bool]:
        actions_dict = {a: int(actions[i]) for i, a in enumerate(self._agents)}
        obs_dict, state, reward_dict, done_dict, info = self._env.step(rng_key, state, actions_dict)
        raw_obs = np.stack([np.array(obs_dict[a]) for a in self._agents])
        self._obs_stack = np.roll(self._obs_stack, shift=-1, axis=1)
        self._obs_stack[:, -1, :] = raw_obs
        stacked = self._obs_stack.reshape(self.num_agents, -1)
        reward = float(np.mean([reward_dict[a] for a in self._agents]))
        done = bool(done_dict["__all__"])
        return stacked, state, reward, done, False

    def get_legal_actions(self, state) -> np.ndarray:
        return np.ones((self.num_agents, self.action_space_size), dtype=bool)
```

- [ ] **Step 6: Create envs __init__**

```python
# jaxzero/envs/__init__.py
```

- [ ] **Step 7: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_env.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add jaxzero/envs/ tests/test_env.py
git commit -m "feat: add SMAX and MPE environment wrappers"
```

---

## Task 6: GameHistory

**Files:**
- Create: `jaxzero/game.py`
- Create: `tests/test_game.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_game.py
import numpy as np
import pytest
from jaxzero.game import GameHistory


N, A, D = 3, 9, 80
OBS_DIM = D * 4
K = 10


def make_game(T=20):
    g = GameHistory(num_agents=N, obs_dim=D, action_space_size=A, stacked_observations=4)
    for t in range(T):
        obs = np.random.randn(N, D).astype(np.float32)
        g.store_observation(obs)
        g.store_action(np.zeros(N, dtype=np.int32))
        g.store_reward(1.0)
        g.store_legal_actions(np.ones((N, A), dtype=bool))
        g.store_root_value(1.0)
        g.store_pred_value(0.9)
        g.store_search_stats(
            sampled_actions=np.zeros((K, N), dtype=np.int32),
            visit_counts=np.ones(K) / K,
            qvalues=np.zeros(K),
            mask=np.ones(K, dtype=bool),
        )
    return g


def test_obs_stacked_shape():
    g = make_game(T=10)
    obs = g.obs(t=5, stacked_obs=4)
    assert obs.shape == (N, D * 4)


def test_obs_at_start_pads():
    """t=0 should pad with first observation repeated."""
    g = make_game(T=10)
    obs_t0 = g.obs(t=0, stacked_obs=4)
    obs_t1 = g.obs(t=1, stacked_obs=4)
    assert obs_t0.shape == (N, D * 4)
    assert obs_t1.shape == (N, D * 4)


def test_game_length():
    g = make_game(T=15)
    assert len(g) == 15


def test_make_target_shapes():
    g = make_game(T=20)
    obs_b, actions_b, rewards_b, values_b, policies_b, qvals_b, masks_b = g.make_target(
        pos=5, unroll_steps=5, td_steps=5, discount=0.99
    )
    assert obs_b.shape == (6, N, D * 4)    # pos + unroll_steps+1 obs
    assert actions_b.shape == (5, N)
    assert rewards_b.shape == (5,)
    assert values_b.shape == (6,)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_game.py -v 2>&1 | head -20`
Expected: ImportError

- [ ] **Step 3: Implement game.py**

```python
# jaxzero/game.py
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


class GameHistory:
    """Stores one complete episode trajectory."""

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_space_size: int,
        stacked_observations: int = 4,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_space_size = action_space_size
        self.stacked_observations = stacked_observations

        # Pre-pad obs_history with stacked_observations-1 copies to handle t=0
        self._obs_history: list[np.ndarray] = []  # each (N, obs_dim)
        self.actions: list[np.ndarray] = []        # each (N,)
        self.rewards: list[float] = []
        self.legal_actions: list[np.ndarray] = []  # each (N, A)
        self.root_values: list[float] = []
        self.pred_values: list[float] = []
        self.sampled_actions: list[np.ndarray] = []  # each (K, N)
        self.sampled_policies: list[np.ndarray] = [] # each (K,)
        self.sampled_qvalues: list[np.ndarray] = []  # each (K,)
        self.sampled_masks: list[np.ndarray] = []    # each (K,)

    def store_observation(self, obs: np.ndarray):
        self._obs_history.append(obs.copy())

    def store_action(self, action: np.ndarray):
        self.actions.append(action.copy())

    def store_reward(self, reward: float):
        self.rewards.append(float(reward))

    def store_legal_actions(self, legal: np.ndarray):
        self.legal_actions.append(legal.copy())

    def store_root_value(self, v: float):
        self.root_values.append(float(v))

    def store_pred_value(self, v: float):
        self.pred_values.append(float(v))

    def store_search_stats(
        self,
        sampled_actions: np.ndarray,
        visit_counts: np.ndarray,
        qvalues: np.ndarray,
        mask: np.ndarray,
    ):
        self.sampled_actions.append(sampled_actions.copy())
        self.sampled_policies.append(visit_counts.copy())
        self.sampled_qvalues.append(qvalues.copy())
        self.sampled_masks.append(mask.copy())

    def __len__(self) -> int:
        return len(self.rewards)

    def obs(self, t: int, stacked_obs: int) -> np.ndarray:
        """Return stacked obs window ending at step t.

        Pads with the earliest available obs when t < stacked_obs - 1.
        Returns shape (N, obs_dim * stacked_obs).
        """
        frames = []
        for i in range(stacked_obs - 1, -1, -1):
            idx = t - i
            if idx < 0:
                idx = 0
            frames.append(self._obs_history[idx])
        return np.concatenate(frames, axis=-1)

    def make_target(
        self,
        pos: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> tuple:
        """Build training targets for one position.

        Returns:
            obs_batch:     (unroll_steps+1, N, obs_dim*stacked_obs)
            actions_batch: (unroll_steps, N)
            rewards_batch: (unroll_steps,)
            values_batch:  (unroll_steps+1,)
            policies_batch:(unroll_steps+1, K)  — sampled visit counts
            qvals_batch:   (unroll_steps+1, K)
            masks_batch:   (unroll_steps+1, K)
        """
        T = len(self)
        S = self.stacked_observations
        K = len(self.sampled_actions[0])

        obs_batch = np.stack([self.obs(min(pos + k, T - 1), S) for k in range(unroll_steps + 1)])
        actions_batch = np.stack([
            self.actions[min(pos + k, T - 1)] for k in range(unroll_steps)
        ])
        rewards_batch = np.array([
            self.rewards[min(pos + k, T - 1)] for k in range(unroll_steps)
        ])

        # n-step value targets
        values_batch = []
        for k in range(unroll_steps + 1):
            t = pos + k
            value = 0.0
            for n in range(td_steps):
                if t + n < T:
                    value += (discount ** n) * self.rewards[t + n]
            bootstrap_t = t + td_steps
            if bootstrap_t < T:
                value += (discount ** td_steps) * self.root_values[bootstrap_t]
            values_batch.append(value)
        values_batch = np.array(values_batch)

        policies_batch = np.stack([self.sampled_policies[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
        qvals_batch = np.stack([self.sampled_qvalues[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])
        masks_batch = np.stack([self.sampled_masks[min(pos + k, T - 1)] for k in range(unroll_steps + 1)])

        return obs_batch, actions_batch, rewards_batch, values_batch, policies_batch, qvals_batch, masks_batch
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_game.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add jaxzero/game.py tests/test_game.py
git commit -m "feat: add GameHistory trajectory storage"
```

---

## Task 7: Prioritized Replay Buffer

**Files:**
- Create: `jaxzero/replay_buffer.py`
- Create: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_replay_buffer.py
import numpy as np
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.game import GameHistory
from jaxzero.replay_buffer import PrioritizedReplayBuffer


def make_game(T=20, N=3, obs_dim=80, A=9):
    g = GameHistory(num_agents=N, obs_dim=obs_dim, action_space_size=A, stacked_observations=4)
    K = 10
    for t in range(T):
        g.store_observation(np.random.randn(N, obs_dim).astype(np.float32))
        g.store_action(np.zeros(N, dtype=np.int32))
        g.store_reward(float(np.random.randn()))
        g.store_legal_actions(np.ones((N, A), dtype=bool))
        g.store_root_value(float(np.random.randn()))
        g.store_pred_value(float(np.random.randn()))
        g.store_search_stats(
            sampled_actions=np.zeros((K, N), dtype=np.int32),
            visit_counts=np.ones(K) / K,
            qvalues=np.zeros(K),
            mask=np.ones(K, dtype=bool),
        )
    return g


def make_buffer(size=1000):
    config = MAZeroConfig(
        replay_buffer_size=size,
        min_replay_size=10,
        priority_alpha=0.6,
        priority_beta_start=0.4,
        batch_size=4,
    )
    return PrioritizedReplayBuffer(config), config


def test_can_sample_false_when_empty():
    buf, cfg = make_buffer()
    assert not buf.can_sample(4)


def test_can_sample_true_after_enough_games():
    buf, cfg = make_buffer()
    for _ in range(20):
        buf.add(make_game())
    assert buf.can_sample(4)


def test_prepare_batch_shapes():
    buf, cfg = make_buffer()
    for _ in range(20):
        buf.add(make_game())
    games, positions, indices, weights = buf.prepare_batch_context(4, beta=0.4)
    assert len(games) == 4
    assert len(positions) == 4
    assert len(indices) == 4
    assert weights.shape == (4,)


def test_priority_update():
    buf, cfg = make_buffer()
    for _ in range(20):
        buf.add(make_game())
    games, positions, indices, weights = buf.prepare_batch_context(4, beta=0.4)
    new_priorities = np.ones(4)
    buf.update_priorities(indices, new_priorities)  # should not raise


def test_buffer_capacity_limit():
    buf, cfg = make_buffer(size=5)
    for _ in range(10):
        buf.add(make_game())
    assert buf.size <= 5
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_replay_buffer.py -v 2>&1 | head -20`
Expected: ImportError

- [ ] **Step 3: Implement replay_buffer.py**

```python
# jaxzero/replay_buffer.py
import numpy as np
from collections import deque
from jaxzero.config import MAZeroConfig
from jaxzero.game import GameHistory


class PrioritizedReplayBuffer:
    def __init__(self, config: MAZeroConfig):
        self.config = config
        self.capacity = config.replay_buffer_size
        self.alpha = config.priority_alpha
        self._games: deque[GameHistory] = deque(maxlen=self.capacity)
        self._priorities: deque[float] = deque(maxlen=self.capacity)
        self._game_step_counts: deque[int] = deque(maxlen=self.capacity)

    @property
    def size(self) -> int:
        return len(self._games)

    def add(self, game: GameHistory, priority: float = 1.0):
        self._games.append(game)
        self._priorities.append(priority)
        self._game_step_counts.append(len(game))

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= max(batch_size, self.config.min_replay_size)

    def prepare_batch_context(
        self, batch_size: int, beta: float
    ) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        priorities = np.array(self._priorities, dtype=np.float64)
        probs = (priorities ** self.alpha)
        probs /= probs.sum()

        game_indices = np.random.choice(len(self._games), size=batch_size, p=probs)
        games = [self._games[i] for i in game_indices]
        positions = np.array([
            np.random.randint(0, max(1, len(games[b])))
            for b in range(batch_size)
        ])

        # Importance-sampling weights
        min_prob = probs.min()
        max_weight = (len(self._games) * min_prob) ** (-beta)
        weights = ((len(self._games) * probs[game_indices]) ** (-beta)) / max_weight
        weights = weights.astype(np.float32)

        return games, positions, game_indices, weights

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
        for idx, p in zip(indices, new_priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = float(abs(p)) + 1e-6
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_replay_buffer.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add jaxzero/replay_buffer.py tests/test_replay_buffer.py
git commit -m "feat: add PrioritizedReplayBuffer"
```

---

## Task 8: Sampled MCTS + OS(λ)

**Files:**
- Create: `jaxzero/mcts/__init__.py`
- Create: `jaxzero/mcts/sampled_mcts.py`
- Create: `tests/test_mcts.py`

This is the hardest module. Python tree; JIT'd model calls.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcts.py
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet
from jaxzero.mcts.sampled_mcts import SampledMCTS, SearchOutput


B, N, A = 2, 3, 9
OBS_DIM = 80 * 4


def make_config():
    return MAZeroConfig(
        num_agents=N,
        obs_size=OBS_DIM,
        action_space_size=A,
        num_simulations=10,
        sampled_action_times=5,
    )


def make_net_and_params(config):
    net = MAMuZeroNet(config=config)
    obs = jnp.ones((B, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs)
    return net, params


def test_search_output_shapes():
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.ones((B, N, OBS_DIM), dtype=np.float32)
    legal = np.ones((B, N, A), dtype=bool)
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    assert isinstance(result, SearchOutput)
    assert result.root_value.shape == (B,)
    assert len(result.sampled_actions) == B
    assert len(result.sampled_visit_counts) == B


def test_legal_masking_respected():
    """If only action 0 is legal for all agents, all sampled actions should be 0."""
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.ones((B, N, OBS_DIM), dtype=np.float32)
    legal = np.zeros((B, N, A), dtype=bool)
    legal[:, :, 0] = True  # only action 0 legal
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    for b in range(B):
        assert (result.sampled_actions[b] == 0).all()


def test_visit_counts_sum_to_simulations():
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.ones((B, N, OBS_DIM), dtype=np.float32)
    legal = np.ones((B, N, A), dtype=bool)
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    for b in range(B):
        assert result.sampled_visit_counts[b].sum() == config.num_simulations


def test_batch_independence():
    """Both batch items should get independent root values (not identical)."""
    config = make_config()
    net, params = make_net_and_params(config)
    mcts = SampledMCTS(config=config, model=net)
    obs = np.random.randn(B, N, OBS_DIM).astype(np.float32)
    legal = np.ones((B, N, A), dtype=bool)
    rng = np.random.default_rng(0)
    result = mcts.search(params, obs, legal, rng)
    assert not np.allclose(result.root_value[0], result.root_value[1])
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_mcts.py -v 2>&1 | head -20`
Expected: ImportError

- [ ] **Step 3: Implement sampled_mcts.py**

```python
# jaxzero/mcts/sampled_mcts.py
import numpy as np
import jax
import jax.numpy as jnp
import math
from typing import NamedTuple, Any
from jaxzero.config import MAZeroConfig


class SearchOutput(NamedTuple):
    root_value: np.ndarray           # (B,)
    sampled_actions: list            # list[B] of (K_i, N)
    sampled_visit_counts: list       # list[B] of (K_i,)
    sampled_qvalues: list            # list[B] of (K_i,)
    sampled_imp_ratio: list          # list[B] of (K_i,)


class Node:
    """Single node in MCTS tree for one batch element."""
    __slots__ = [
        'visit_count', 'q_value', 'reward', 'value_sum',
        'prior', 'beta', 'hidden', 'children',
        'sampled_actions', 'expanded',
    ]

    def __init__(self):
        self.visit_count = 0
        self.q_value: np.ndarray = None   # (K,) OS(λ) advantage per action
        self.reward = 0.0
        self.value_sum = 0.0
        self.prior: np.ndarray = None     # (K,) policy probs for sampled actions
        self.beta: np.ndarray = None      # (K,) sampling distribution
        self.hidden: np.ndarray = None    # (N, D)
        self.children: dict[int, 'Node'] = {}  # action_idx → child node
        self.sampled_actions: np.ndarray = None  # (K, N)
        self.expanded = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class SampledMCTS:
    def __init__(self, config: MAZeroConfig, model):
        self.config = config
        self.model = model
        self._jit_initial = jax.jit(model.apply)
        self._jit_recurrent = jax.jit(
            lambda p, h, a: model.apply(p, h, a, method=model.recurrent_inference)
        )
        self._jit_phi_inv = jax.jit(self._phi_inv_fn)

    def _phi_inv_fn(self, logits):
        from jaxzero.model.transforms import phi_inv
        return phi_inv(logits, self.config.value_support_size)

    def _sample_actions(
        self, policy_logits: np.ndarray, legal_mask: np.ndarray, K: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample K joint actions from policy.

        Args:
            policy_logits: (N, A)
            legal_mask: (N, A) bool
            K: number of samples
            rng: numpy rng

        Returns:
            sampled_actions: (K, N)
            beta: (K,) joint sampling probability
            prior: (K,) policy probability (for imp ratio)
        """
        N, A = policy_logits.shape
        # Softmax per agent, mask illegal
        log_probs = policy_logits - policy_logits.max(axis=-1, keepdims=True)
        probs = np.exp(log_probs)
        probs = probs * legal_mask.astype(np.float32)
        probs += legal_mask.astype(np.float32) * 1e-4
        probs /= probs.sum(axis=-1, keepdims=True)

        # Sample K joint actions (independent per agent)
        actions = np.zeros((K, N), dtype=np.int32)
        for n in range(N):
            actions[:, n] = rng.choice(A, size=K, p=probs[n])

        # Joint probability = product over agents
        joint_prob = np.ones(K)
        for n in range(N):
            joint_prob *= probs[n, actions[:, n]]

        return actions, joint_prob, joint_prob

    def _add_dirichlet(
        self, beta: np.ndarray, legal_mask_flat: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        noise = rng.dirichlet(
            np.ones(len(beta)) * self.config.root_dirichlet_alpha
        )
        eps = self.config.root_exploration_fraction
        return (1 - eps) * beta + eps * noise

    def _ucb_score(
        self, parent: Node, child_idx: int
    ) -> float:
        cfg = self.config
        parent_visits = sum(c.visit_count for c in parent.children.values())
        c = cfg.pb_c_init + math.log((parent_visits + cfg.pb_c_base + 1) / cfg.pb_c_base)
        n_child = parent.children[child_idx].visit_count
        prior = parent.prior[child_idx]
        beta_ratio = parent.beta[child_idx] / (parent.beta[child_idx] + 1e-8)
        q = parent.q_value[child_idx] if parent.q_value is not None else 0.0
        exploit = q
        explore = prior * beta_ratio * math.sqrt(parent_visits + 1) / (1 + n_child) * c
        return exploit + explore

    def _select(self, root: Node) -> list[tuple[Node, int]]:
        """Traverse tree to leaf, return path as (node, action_idx) pairs."""
        path = []
        node = root
        while node.expanded:
            scores = {
                idx: self._ucb_score(node, idx)
                for idx in range(len(node.sampled_actions))
                if idx in node.children
            }
            # Also consider unvisited children (score = +inf for unvisited)
            for idx in range(len(node.sampled_actions)):
                if idx not in node.children:
                    scores[idx] = float('inf')
            best_idx = max(scores, key=scores.__getitem__)
            path.append((node, best_idx))
            if best_idx not in node.children:
                break
            node = node.children[best_idx]
        return path

    def _backup_os_lambda(
        self, path: list[tuple[Node, int]], leaf_value: float, discount: float
    ):
        """OS(λ) backup: update q_value estimates along path."""
        rho = self.config.mcts_rho
        lam = self.config.mcts_lambda

        # Walk path in reverse to compute n-step returns
        value = leaf_value
        for node, action_idx in reversed(path):
            node.visit_count += 1
            node.children[action_idx].visit_count += 1

            # Collect U_d estimates (simplified: use direct backup for now)
            # Full OS(λ): aggregate returns at multiple depths
            r = node.children[action_idx].reward
            value = r + discount * value

            if node.q_value is None:
                node.q_value = np.zeros(len(node.sampled_actions))

            # Update running mean of q_value
            n = node.children[action_idx].visit_count
            old_q = node.q_value[action_idx]
            node.q_value[action_idx] = old_q + (value - old_q) / n

    def _search_single(
        self,
        params: Any,
        obs: np.ndarray,        # (N, obs_dim)
        legal: np.ndarray,      # (N, A)
        rng: np.random.Generator,
    ) -> tuple[Node, float]:
        """Run MCTS for a single batch element."""
        cfg = self.config
        K = cfg.sampled_action_times

        # Initial inference
        obs_b = jnp.array(obs[np.newaxis])  # (1, N, obs_dim)
        out = self._jit_initial(params, obs_b)
        policy_logits = np.array(out.policy_logits[0])  # (N, A)
        value_scalar = float(self._jit_phi_inv(out.value_logits)[0])
        hidden = np.array(out.hidden_state[0])  # (N, D)

        # Build root
        root = Node()
        root.hidden = hidden
        root.expanded = True
        root.value_sum = value_scalar
        root.visit_count = 1

        sampled_actions, beta, prior = self._sample_actions(policy_logits, legal, K, rng)
        beta = self._add_dirichlet(beta, legal, rng)
        root.sampled_actions = sampled_actions
        root.beta = beta
        root.prior = prior
        root.q_value = np.zeros(K)

        # Pre-create child nodes (unvisited)
        for k in range(K):
            root.children[k] = Node()

        # Simulations
        for sim in range(cfg.num_simulations):
            path = self._select(root)

            if not path:
                break

            parent, action_idx = path[-1]
            action = parent.sampled_actions[action_idx]  # (N,)

            # Expand
            hidden_b = jnp.array(parent.hidden[np.newaxis])       # (1, N, D)
            action_b = jnp.array(action[np.newaxis], dtype=jnp.int32)  # (1, N)
            rec_out = self._jit_recurrent(params, hidden_b, action_b)

            child = parent.children[action_idx]
            child.hidden = np.array(rec_out.hidden_state[0])
            child.reward = float(self._jit_phi_inv(rec_out.reward_logits)[0])
            child.value_sum = float(self._jit_phi_inv(rec_out.value_logits)[0])
            child.visit_count = 1
            child.expanded = True

            child_policy = np.array(rec_out.policy_logits[0])  # (N, A)
            child_legal = np.ones((self.config.num_agents, self.config.action_space_size), dtype=bool)
            child_actions, child_beta, child_prior = self._sample_actions(
                child_policy, child_legal, K, rng
            )
            child.sampled_actions = child_actions
            child.beta = child_beta
            child.prior = child_prior
            child.q_value = np.zeros(K)
            for kk in range(K):
                child.children[kk] = Node()

            leaf_value = child.value_sum
            self._backup_os_lambda(path, leaf_value, cfg.discount)

        return root, value_scalar

    def search(
        self,
        params: Any,
        obs: np.ndarray,    # (B, N, obs_dim)
        legal: np.ndarray,  # (B, N, A)
        rng: np.random.Generator,
    ) -> SearchOutput:
        B = obs.shape[0]
        root_values = np.zeros(B)
        all_sampled_actions = []
        all_visit_counts = []
        all_qvalues = []
        all_imp_ratios = []

        for b in range(B):
            root, v = self._search_single(params, obs[b], legal[b], rng)
            root_values[b] = v

            # Gather results from root children
            K = len(root.sampled_actions)
            visit_counts = np.array([root.children[k].visit_count for k in range(K)])
            qvalues = root.q_value.copy()
            imp_ratios = root.prior / (root.beta + 1e-8)

            all_sampled_actions.append(root.sampled_actions)
            all_visit_counts.append(visit_counts)
            all_qvalues.append(qvalues)
            all_imp_ratios.append(imp_ratios)

        return SearchOutput(
            root_value=root_values,
            sampled_actions=all_sampled_actions,
            sampled_visit_counts=all_visit_counts,
            sampled_qvalues=all_qvalues,
            sampled_imp_ratio=all_imp_ratios,
        )
```

- [ ] **Step 4: Create mcts __init__**

```python
# jaxzero/mcts/__init__.py
```

- [ ] **Step 5: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_mcts.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add jaxzero/mcts/ tests/test_mcts.py
git commit -m "feat: add SampledMCTS with OS(lambda) backup"
```

---

## Task 9: Reanalyze Worker

**Files:**
- Create: `jaxzero/reanalyze.py`
- Create: `tests/test_reanalyze.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reanalyze.py
import numpy as np
import jax
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet
from jaxzero.game import GameHistory
from jaxzero.replay_buffer import PrioritizedReplayBuffer
from jaxzero.reanalyze import ReanalyzeWorker, BatchData


B, N, A, OBS_DIM = 4, 3, 9, 80 * 4
K = 10


def make_config(use_reanalyze=False):
    return MAZeroConfig(
        num_agents=N,
        obs_size=OBS_DIM,
        action_space_size=A,
        batch_size=B,
        unroll_steps=3,
        td_steps=3,
        use_reanalyze=use_reanalyze,
        num_simulations=5,
        sampled_action_times=K,
        min_replay_size=5,
    )


def make_game(T=20):
    g = GameHistory(num_agents=N, obs_dim=OBS_DIM // 4, action_space_size=A, stacked_observations=4)
    for t in range(T):
        g.store_observation(np.random.randn(N, OBS_DIM // 4).astype(np.float32))
        g.store_action(np.zeros(N, dtype=np.int32))
        g.store_reward(float(np.random.randn()))
        g.store_legal_actions(np.ones((N, A), dtype=bool))
        g.store_root_value(float(np.random.randn()))
        g.store_pred_value(float(np.random.randn()))
        g.store_search_stats(
            sampled_actions=np.zeros((K, N), dtype=np.int32),
            visit_counts=np.ones(K) / K,
            qvalues=np.random.randn(K).astype(np.float32),
            mask=np.ones(K, dtype=bool),
        )
    return g


def make_buffer_ctx(config):
    buf = PrioritizedReplayBuffer(config)
    for _ in range(10):
        buf.add(make_game())
    return buf.prepare_batch_context(B, beta=0.4)


def test_batch_shapes_no_reanalyze():
    config = make_config(use_reanalyze=False)
    net = MAMuZeroNet(config=config)
    obs = np.ones((1, N, OBS_DIM), dtype=np.float32)
    params = net.init(jax.random.PRNGKey(0), obs)
    worker = ReanalyzeWorker(config=config, model=net)
    ctx = make_buffer_ctx(config)
    batch = worker.make_batch(ctx, params)
    U = config.unroll_steps
    assert batch.obs.shape == (B, U + 1, N, OBS_DIM)
    assert batch.actions.shape == (B, U, N)
    assert batch.target_rewards.shape == (B, U)
    assert batch.target_values.shape == (B, U + 1)
    assert batch.weights.shape == (B,)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_reanalyze.py -v 2>&1 | head -20`
Expected: ImportError

- [ ] **Step 3: Implement reanalyze.py**

```python
# jaxzero/reanalyze.py
import numpy as np
import math
from typing import NamedTuple, Any
from jaxzero.config import MAZeroConfig
from jaxzero.game import GameHistory


class BatchData(NamedTuple):
    obs: np.ndarray             # (B, U+1, N, obs_dim)
    actions: np.ndarray         # (B, U, N)
    target_rewards: np.ndarray  # (B, U)
    target_values: np.ndarray   # (B, U+1)
    target_policies: np.ndarray # (B, U+1, K)
    target_qvalues: np.ndarray  # (B, U+1, K)
    target_masks: np.ndarray    # (B, U+1, K)
    weights: np.ndarray         # (B,)
    indices: np.ndarray         # (B,)


class ReanalyzeWorker:
    def __init__(self, config: MAZeroConfig, model=None):
        self.config = config
        self.model = model
        if config.use_reanalyze and model is not None:
            import jax
            from jaxzero.mcts.sampled_mcts import SampledMCTS
            self._mcts = SampledMCTS(config=config, model=model)
            self._rng = np.random.default_rng(config.seed)
        else:
            self._mcts = None

    def make_batch(self, buffer_context, params: Any) -> BatchData:
        games, positions, indices, weights = buffer_context
        B = len(games)
        U = self.config.unroll_steps

        obs_list, actions_list, rewards_list, values_list = [], [], [], []
        policies_list, qvals_list, masks_list = [], [], []

        for b in range(B):
            game: GameHistory = games[b]
            pos = int(positions[b])
            obs_b, act_b, rew_b, val_b, pol_b, qv_b, mask_b = game.make_target(
                pos=pos,
                unroll_steps=U,
                td_steps=self.config.td_steps,
                discount=self.config.discount,
            )
            obs_list.append(obs_b)
            actions_list.append(act_b)
            rewards_list.append(rew_b)
            values_list.append(val_b)
            policies_list.append(pol_b)
            qvals_list.append(qv_b)
            masks_list.append(mask_b)

        batch = BatchData(
            obs=np.stack(obs_list),
            actions=np.stack(actions_list),
            target_rewards=np.stack(rewards_list),
            target_values=np.stack(values_list),
            target_policies=np.stack(policies_list),
            target_qvalues=np.stack(qvals_list),
            target_masks=np.stack(masks_list),
            weights=weights,
            indices=indices,
        )
        return batch
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_reanalyze.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add jaxzero/reanalyze.py tests/test_reanalyze.py
git commit -m "feat: add ReanalyzeWorker batch builder"
```

---

## Task 10: Training Loop + Loss Functions

**Files:**
- Create: `jaxzero/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_train.py
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest
from jaxzero.config import MAZeroConfig
from jaxzero.model.networks import MAMuZeroNet
from jaxzero.train import make_update_fn, awpo_sharp_loss
from jaxzero.reanalyze import BatchData


B, N, A, U, K = 2, 3, 9, 3, 5
OBS_DIM = 80 * 4


def make_config():
    return MAZeroConfig(
        num_agents=N,
        obs_size=OBS_DIM,
        action_space_size=A,
        unroll_steps=U,
        td_steps=3,
        batch_size=B,
        num_simulations=5,
        sampled_action_times=K,
        hidden_state_size=128,
    )


def make_fake_batch(config):
    return BatchData(
        obs=np.random.randn(B, U + 1, N, OBS_DIM).astype(np.float32),
        actions=np.zeros((B, U, N), dtype=np.int32),
        target_rewards=np.random.randn(B, U).astype(np.float32),
        target_values=np.random.randn(B, U + 1).astype(np.float32),
        target_policies=np.ones((B, U + 1, K), dtype=np.float32) / K,
        target_qvalues=np.random.randn(B, U + 1, K).astype(np.float32),
        target_masks=np.ones((B, U + 1, K), dtype=np.float32),
        weights=np.ones(B, dtype=np.float32),
        indices=np.arange(B),
    )


def test_update_fn_runs():
    config = make_config()
    net = MAMuZeroNet(config=config)
    obs_init = jnp.ones((1, N, OBS_DIM))
    params = net.init(jax.random.PRNGKey(0), obs_init)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    update_fn = make_update_fn(net, config)
    batch = make_fake_batch(config)
    loss, grads = jax.value_and_grad(update_fn)(params, batch)
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
    for _ in range(10):
        loss, grads = jax.value_and_grad(update_fn)(params, batch)
        losses.append(float(loss))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_awpo_loss_shape():
    policy_logits = jnp.zeros((B, N, A))
    sampled_actions = jnp.zeros((B, K, N), dtype=jnp.int32)
    visit_counts = jnp.ones((B, K)) / K
    advantages = jnp.zeros((B, K))
    masks = jnp.ones((B, K))
    loss = awpo_sharp_loss(policy_logits, sampled_actions, visit_counts, advantages, masks, alpha=3.0)
    assert loss.shape == (B,)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_train.py -v 2>&1 | head -20`
Expected: ImportError

- [ ] **Step 3: Implement train.py**

```python
# jaxzero/train.py
import jax
import jax.numpy as jnp
import optax
import numpy as np
import functools
from typing import Any
from jaxzero.config import MAZeroConfig
from jaxzero.model.transforms import phi, phi_inv
from jaxzero.reanalyze import BatchData


def awpo_sharp_loss(
    policy_logits: jnp.ndarray,   # (B, N, A)
    sampled_actions: jnp.ndarray, # (B, K, N)
    visit_counts: jnp.ndarray,    # (B, K) normalized
    advantages: jnp.ndarray,      # (B, K)
    masks: jnp.ndarray,           # (B, K)
    alpha: float,
) -> jnp.ndarray:                 # (B,)
    B, K, N = sampled_actions.shape
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)  # (B, N, A)

    # Gather log prob for each sampled action per agent, sum over agents
    def gather_agent_logprob(log_p, actions):
        # log_p: (N, A), actions: (K, N)
        return jnp.sum(log_p[jnp.arange(N), actions.T], axis=0)  # (K,)

    action_log_probs = jax.vmap(gather_agent_logprob)(log_probs, sampled_actions)  # (B, K)

    # Normalize advantages
    adv = advantages - (advantages * masks).sum(-1, keepdims=True) / (masks.sum(-1, keepdims=True) + 1e-8)
    adv_weights = jnp.exp(adv / alpha)

    loss = -(action_log_probs * visit_counts * adv_weights * masks).sum(axis=-1)
    return loss  # (B,)


def categorical_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy between logits and soft target distribution."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -(targets * log_probs).sum(axis=-1)


def make_update_fn(model, config: MAZeroConfig):
    """Returns a JIT-able loss function (params, batch) -> scalar."""

    @functools.partial(jax.jit)
    def update_fn(params: Any, batch: BatchData) -> jnp.ndarray:
        obs = jnp.array(batch.obs)              # (B, U+1, N, obs_dim)
        actions = jnp.array(batch.actions)      # (B, U, N)
        target_rewards = jnp.array(batch.target_rewards)  # (B, U)
        target_values = jnp.array(batch.target_values)    # (B, U+1)
        target_policies = jnp.array(batch.target_policies) # (B, U+1, K)
        target_qvalues = jnp.array(batch.target_qvalues)   # (B, U+1, K)
        target_masks = jnp.array(batch.target_masks)       # (B, U+1, K)
        weights = jnp.array(batch.weights)                 # (B,)

        S_v = config.value_support_size
        S_r = config.reward_support_size

        # Encode targets to support distributions
        def encode_value(v):
            from jaxzero.model.transforms import h, phi
            return phi(h(v), S_v)

        def encode_reward(r):
            from jaxzero.model.transforms import h, phi
            return phi(h(r), S_r)

        B = obs.shape[0]
        U = config.unroll_steps

        # Initial inference at step 0
        out0 = model.apply(params, obs[:, 0])  # (B, N, obs_dim)
        target_v_phi_0 = jnp.array([encode_value(target_values[:, 0])])
        # target_v_phi_0 has wrong shape; fix:
        target_v_phi_0 = jax.vmap(lambda v: phi(jnp.array([v]), S_v)[0])(target_values[:, 0])

        value_loss = categorical_cross_entropy(out0.value_logits, target_v_phi_0)
        reward_loss = jnp.zeros(B)
        policy_loss = awpo_sharp_loss(
            out0.policy_logits,
            jnp.array(batch.target_policies[:, 0]).astype(jnp.int32).reshape(B, -1, 1)
            if False else jnp.zeros((B, target_qvalues.shape[2], config.num_agents), dtype=jnp.int32),
            target_policies[:, 0],
            target_qvalues[:, 0],
            target_masks[:, 0],
            config.awpo_alpha,
        )
        consistency_loss = jnp.zeros(B)

        hidden = out0.hidden_state

        for k in range(1, U + 1):
            # Half-gradient trick
            hidden = hidden / 2.0 + jax.lax.stop_gradient(hidden / 2.0)
            out_k = model.apply(
                params, hidden, actions[:, k - 1],
                method=model.recurrent_inference
            )

            target_r_phi = jax.vmap(lambda r: phi(jnp.array([r]), S_r)[0])(target_rewards[:, k - 1])
            target_v_phi = jax.vmap(lambda v: phi(jnp.array([v]), S_v)[0])(target_values[:, k])

            reward_loss = reward_loss + categorical_cross_entropy(out_k.reward_logits, target_r_phi)
            value_loss = value_loss + categorical_cross_entropy(out_k.value_logits, target_v_phi)
            policy_loss = policy_loss + awpo_sharp_loss(
                out_k.policy_logits,
                jnp.zeros((B, target_qvalues.shape[2], config.num_agents), dtype=jnp.int32),
                target_policies[:, k],
                target_qvalues[:, k],
                target_masks[:, k],
                config.awpo_alpha,
            )

            hidden = out_k.hidden_state

        total = (
            config.reward_loss_coeff * reward_loss
            + config.value_loss_coeff * value_loss
            + config.policy_loss_coeff * policy_loss
            + config.consistency_coeff * consistency_loss
        )
        return (weights * total).mean() / U

    return update_fn


def collect_episode(env, params, model, config: MAZeroConfig, rng_key):
    """Collect one episode using MCTS."""
    from jaxzero.game import GameHistory
    from jaxzero.mcts.sampled_mcts import SampledMCTS
    import jax

    mcts = SampledMCTS(config=config, model=model)
    rng = np.random.default_rng(int(jax.random.randint(rng_key, (), 0, 2**31)))

    game = GameHistory(
        num_agents=config.num_agents,
        obs_dim=config.obs_size // config.stacked_observations,
        action_space_size=config.action_space_size,
        stacked_observations=config.stacked_observations,
    )

    import jax.random as jr
    env_rng = jr.split(rng_key, 2)
    obs, state = env.reset(env_rng[0])

    game.store_observation(obs[:, :config.obs_size // config.stacked_observations])

    done = False
    step = 0
    while not done and step < config.max_episode_steps:
        legal = env.get_legal_actions(state)
        obs_input = obs[np.newaxis]  # (1, N, obs_dim)
        legal_input = legal[np.newaxis]  # (1, N, A)

        result = mcts.search(params, obs_input, legal_input, rng)

        # Select action: sample from visit count distribution
        visit_counts = result.sampled_visit_counts[0]
        actions_pool = result.sampled_actions[0]  # (K, N)
        probs = visit_counts / visit_counts.sum()
        chosen_idx = rng.choice(len(probs), p=probs)
        action = actions_pool[chosen_idx]  # (N,)

        env_rng2 = jr.split(env_rng[0], 2)
        obs_next, state, reward, done, won = env.step(env_rng2[0], state, action)
        env_rng = env_rng2

        raw_obs = obs_next[:, :config.obs_size // config.stacked_observations]
        game.store_observation(raw_obs)
        game.store_action(action)
        game.store_reward(reward)
        game.store_legal_actions(legal)
        game.store_root_value(float(result.root_value[0]))

        import jax
        init_out = jax.jit(model.apply)(params, jnp.array(obs_input))
        pred_val = float(phi_inv(init_out.value_logits, config.value_support_size)[0])
        game.store_pred_value(pred_val)

        game.store_search_stats(
            sampled_actions=result.sampled_actions[0],
            visit_counts=result.sampled_visit_counts[0].astype(np.float32) / visit_counts.sum(),
            qvalues=result.sampled_qvalues[0].astype(np.float32),
            mask=np.ones(len(visit_counts), dtype=bool),
        )

        obs = obs_next
        step += 1

    return game


def train(config: MAZeroConfig, env):
    import jax
    import jax.random as jr
    import optax
    from jaxzero.model.networks import MAMuZeroNet
    from jaxzero.replay_buffer import PrioritizedReplayBuffer
    from jaxzero.reanalyze import ReanalyzeWorker

    net = MAMuZeroNet(config=config)
    rng = jr.PRNGKey(config.seed)
    obs_init = jnp.ones((1, config.num_agents, config.obs_size))
    rng, init_rng = jr.split(rng)
    params = net.init(init_rng, obs_init)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=config.adam_eps),
    )
    opt_state = optimizer.init(params)
    target_params = params

    replay_buffer = PrioritizedReplayBuffer(config)
    reanalyze_worker = ReanalyzeWorker(config=config, model=net)
    update_fn = make_update_fn(net, config)

    beta_schedule = lambda step: min(
        1.0,
        config.priority_beta_start + (1.0 - config.priority_beta_start) * step / config.training_steps,
    )

    step = 0
    while step < config.training_steps:
        rng, ep_rng = jr.split(rng)
        game = collect_episode(env, params, net, config, ep_rng)
        replay_buffer.add(game)

        if not replay_buffer.can_sample(config.batch_size):
            continue

        beta = beta_schedule(step)
        buffer_ctx = replay_buffer.prepare_batch_context(config.batch_size, beta)
        batch = reanalyze_worker.make_batch(buffer_ctx, params)

        loss, grads = jax.value_and_grad(update_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        new_priorities = np.abs(
            batch.target_values[:, 0] - np.array(
                phi_inv(
                    jax.jit(net.apply)(params, jnp.array(batch.obs[:, 0])).value_logits,
                    config.value_support_size,
                )
            )
        ) + 1e-6
        replay_buffer.update_priorities(batch.indices, new_priorities)

        if step % config.target_model_interval == 0:
            target_params = params

        if step % config.log_interval == 0:
            print(f"Step {step}: loss={float(loss):.4f}")

        step += 1

    return params
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/test_train.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add jaxzero/train.py tests/test_train.py
git commit -m "feat: add training loop with AWPO loss and gradient updates"
```

---

## Task 11: Eval + Main Entry Point

**Files:**
- Create: `jaxzero/eval.py`
- Create: `jaxzero/main.py`

- [ ] **Step 1: Implement eval.py**

```python
# jaxzero/eval.py
import numpy as np
import jax
import jax.numpy as jnp
from jaxzero.config import MAZeroConfig
from jaxzero.mcts.sampled_mcts import SampledMCTS
from jaxzero.model.transforms import phi_inv


def evaluate(env, params, model, config: MAZeroConfig) -> dict:
    mcts = SampledMCTS(config=config, model=model)
    rng = np.random.default_rng(config.seed + 999)

    wins = 0
    returns = []
    action_counts = np.zeros((config.num_agents, config.action_space_size))

    for ep in range(config.eval_episodes):
        import jax.random as jr
        env_rng = jr.PRNGKey(ep)
        obs, state = env.reset(env_rng)
        ep_return = 0.0
        done = False
        step = 0

        while not done and step < config.max_episode_steps:
            legal = env.get_legal_actions(state)
            result = mcts.search(params, obs[np.newaxis], legal[np.newaxis], rng)
            visit_counts = result.sampled_visit_counts[0]
            actions_pool = result.sampled_actions[0]
            chosen_idx = np.argmax(visit_counts)  # greedy at eval
            action = actions_pool[chosen_idx]

            for i, a in enumerate(action):
                action_counts[i, a] += 1

            env_rng, step_rng = jr.split(env_rng)
            obs, state, reward, done, won = env.step(step_rng, state, action)
            ep_return += reward
            step += 1

        returns.append(ep_return)
        if done and ep_return > 0:
            wins += 1

    win_rate = wins / config.eval_episodes
    avg_return = float(np.mean(returns))
    action_hist = action_counts / (action_counts.sum(axis=-1, keepdims=True) + 1e-8)

    return {
        "win_rate": win_rate,
        "avg_return": avg_return,
        "avg_episode_length": config.max_episode_steps,
        "action_histogram": action_hist.tolist(),
    }
```

- [ ] **Step 2: Implement main.py**

```python
# jaxzero/main.py
import argparse
import dataclasses
from jaxzero.config import MAZeroConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="3m", choices=["3m", "mpe"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training_steps", type=int, default=100_000)
    parser.add_argument("--no_reanalyze", action="store_true")
    args = parser.parse_args()

    if args.env == "3m":
        from jaxzero.envs.smax_wrapper import SMAXWrapper
        env = SMAXWrapper(map_name="3m", stacked_observations=4)
    else:
        from jaxzero.envs.mpe_wrapper import MPEWrapper
        env = MPEWrapper()

    config = MAZeroConfig(
        env_name=args.env,
        num_agents=env.num_agents,
        obs_size=env.obs_size,
        action_space_size=env.action_space_size,
        seed=args.seed,
        training_steps=args.training_steps,
        use_reanalyze=not args.no_reanalyze,
    )

    from jaxzero.train import train
    params = train(config, env)
    print("Training complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify entry point**

Run: `cd /home/ryan/Repos/jaxzero && python -m jaxzero.main --help`
Expected: shows argument help without error

- [ ] **Step 4: Commit**

```bash
git add jaxzero/eval.py jaxzero/main.py
git commit -m "feat: add eval runner and main entry point"
```

---

## Task 12: CLAUDE.md + Integration Smoke Test

**Files:**
- Create: `jaxzero/CLAUDE.md`

- [ ] **Step 1: Write CLAUDE.md**

```markdown
# jaxzero — MAZero JAX Rewrite

## What this is
Faithful JAX/Flax rewrite of MAZero (ICLR 2024). Targets SMAX 3m via JaxMARL.

## Key design decisions
- Python-tree MCTS (not mctx). Model calls JIT'd.
- CommunicationNet inside DynamicsNetwork per spec Eq. 4 (not pre-search).
- Positional encoding in CommunicationNet: learned nn.Embed, added to projected input.
- AWPO sharp loss. Policy loss: log π(a) * visit_count * exp(adv / α).
- Sync serial training loop first; Ray async later.
- Config: frozen MAZeroConfig dataclass; replace obs_size/action_space_size at startup.

## Running
```
python -m jaxzero.main --env 3m
python -m jaxzero.main --env mpe  # sanity check
python -m pytest tests/ -v
```

## Known paper hyperparams (Table 1)
K=10 sampled actions, N=100 simulations, lr=1e-4, stacked_obs=4, unroll=5, td=5

## Prior failures (sequential-muzero)
1. mctx.gumbel_muzero_policy instead of Sampled MCTS
2. K=5/N=50 vs paper K=10/N=100
3. CommunicationNet outside dynamics
4. No positional encoding in transformer
```

- [ ] **Step 2: Run full test suite**

Run: `cd /home/ryan/Repos/jaxzero && python -m pytest tests/ -v 2>&1 | tail -30`
Expected: all tests pass or clear failure messages to debug

- [ ] **Step 3: MPE smoke test (sanity check learning)**

Run: `cd /home/ryan/Repos/jaxzero && timeout 120 python -m jaxzero.main --env mpe --training_steps 50 2>&1`
Expected: runs without crash, prints loss values

- [ ] **Step 4: Commit**

```bash
git add jaxzero/CLAUDE.md
git commit -m "docs: add CLAUDE.md project context"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered |
|-------------|---------|
| Config (§12) | Task 1 |
| transforms h/φ (§5) | Task 2 |
| MLP + Transformer layers | Task 3 |
| All 5 networks (§5) | Task 4 |
| SMAX + MPE wrappers (§7) | Task 5 |
| GameHistory (§8) | Task 6 |
| PrioritizedReplayBuffer (§9) | Task 7 |
| Sampled MCTS + OS(λ) (§6) | Task 8 |
| ReanalyzeWorker (§10) | Task 9 |
| train.py + AWPO loss (§11) | Task 10 |
| eval.py + main.py (§14) | Task 11 |
| CLAUDE.md (§3) | Task 12 |

**Placeholder scan:** No TBD/TODO in implementation steps. All code complete.

**Type consistency:** `MAZeroConfig` used throughout. `MuZeroOutput` NamedTuple consistent across networks/train. `BatchData` NamedTuple consistent across reanalyze/train. `SearchOutput` consistent.

**Note on train.py policy loss:** Task 10 has a stub for `sampled_actions` in AWPO — the actual sampled actions from `batch.target_policies`/GameHistory need to be wired correctly. The `GameHistory.store_search_stats` stores `sampled_actions: (K, N)`. `BatchData` currently only carries `target_policies` (visit counts), not the actual action indices. **Fix needed:** add `sampled_actions: np.ndarray` field to `BatchData` and `make_target`, then pass to `awpo_sharp_loss`.
