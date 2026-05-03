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
        # Standard (non-zero) init for representation so hidden states are non-trivial.
        x = nn.LayerNorm()(obs)
        return MLP(layer_sizes=self.fc_layers, output_size=self.hidden_state_size, zero_init_output=False)(x)


class CommunicationNetwork(nn.Module):
    """Transformer encoder for inter-agent communication inside dynamics."""
    num_layers: int
    num_heads: int
    hidden_size: int
    dropout_rate: float

    def setup(self):
        self.input_proj = nn.Dense(self.hidden_size)
        self.pos_embed = nn.Embed(
            num_embeddings=32,  # supports up to 32 agents; increase for larger maps
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
        x = self.input_proj(ha)
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
        # dynamic_mlp uses standard (non-zero) output init so the residual update is non-trivial.
        self.dynamic_mlp = MLP(
            layer_sizes=self.fc_dynamic_layers,
            output_size=self.hidden_state_size,
            zero_init_output=False,
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
            zero_init_output=False,
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
        self.projection_net = ProjectionNetwork(hidden_size=cfg.hidden_state_size)

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
        self, hidden: chex.Array, actions: chex.Array, deterministic: bool = True
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
