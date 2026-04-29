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
