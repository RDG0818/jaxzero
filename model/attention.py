# model/attention.py

import flax.linen as nn
import jax.numpy as jnp
import chex
from model.layers import MLP


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> chex.Array:
    """
    Generates a sinusoidal positional encoding matrix.

    This is a pure function of static inputs (seq_len and d_model are fixed at
    construction time), so XLA constant-folds it after the first JIT trace.

    Args:
        seq_len: Length of the sequence (i.e. number of agents).
        d_model: Feature dimensionality. Must be even.

    Returns:
        Positional encoding of shape (1, seq_len, d_model).
    """
    position = jnp.arange(seq_len)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
    pos_enc = jnp.zeros((seq_len, d_model))
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_enc[jnp.newaxis, ...]  # (1, seq_len, d_model)


class TransformerEncoderLayer(nn.Module):
    """Single Pre-LN Transformer encoder layer (LayerNorm applied before each sub-block)."""
    num_heads: int
    hidden_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: chex.Array, *, deterministic: bool) -> chex.Array:
        """
        Args:
            x: Input. Shape: (batch, num_agents, hidden_size)
            deterministic: If True, disables dropout.

        Returns:
            Output of the same shape as input.
        """
        # Pre-LN self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(y, y)
        x = x + nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        # Pre-LN feed-forward
        y = nn.LayerNorm()(x)
        y = MLP(layer_sizes=(self.hidden_size * 2,), output_size=self.hidden_size)(y)
        x = x + nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        return x


class TransformerAttentionEncoder(nn.Module):
    """Transformer encoder for modeling interactions across agents via self-attention."""
    num_layers: int
    num_heads: int
    hidden_size: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: chex.Array, *, deterministic: bool = False) -> chex.Array:
        """
        Args:
            x: Input. Shape: (batch, num_agents, features)
            deterministic: If True, disables dropout.

        Returns:
            Output. Shape: (batch, num_agents, hidden_size)
        """
        chex.assert_rank(x, 3)  # (batch, num_agents, features)

        # Project to hidden_size and add positional encoding so the model knows agent order.
        x = nn.Dense(features=self.hidden_size)(x)
        x = x + sinusoidal_positional_encoding(seq_len=x.shape[1], d_model=x.shape[2])
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
            )(x, deterministic=deterministic)
        return x
