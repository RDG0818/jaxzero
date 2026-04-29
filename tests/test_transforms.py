import numpy as np
import jax.numpy as jnp
import pytest
from jaxzero.model.transforms import h, inv_h, phi, phi_inv


def test_h_inv_h_roundtrip():
    x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    # float32 accumulates ~1e-4 error through sqrt chains at large magnitudes
    assert jnp.allclose(inv_h(h(x)), x, atol=2e-4)


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
