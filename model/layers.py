# model/layers.py

import flax.linen as nn
import jax
import chex


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm + ReLU on each hidden layer, linear output."""
    layer_sizes: tuple[int, ...]
    output_size: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        for size in self.layer_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.LayerNorm()(x)
            x = jax.nn.relu(x)
        x = nn.Dense(features=self.output_size)(x)
        return x
