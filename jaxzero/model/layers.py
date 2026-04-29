import flax.linen as nn
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
    """Single post-LN transformer layer: MHSA + FFN, each wrapped in residual + LayerNorm.
    FFN expands to 4x hidden_size. Dropout applied after FFN projection only."""
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
    """Stack of TransformerEncoderLayers. Input last dim must equal hidden_size."""
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
