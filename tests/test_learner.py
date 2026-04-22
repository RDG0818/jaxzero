"""Tests for actors/learner_actor.py module-level utilities.

Run with:
    conda run -n mazero pytest tests/test_learner.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp


def test_scale_grad_half_forward_is_identity():
    """scale_grad_half should be identity in the forward pass."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.learner_actor import scale_grad_half
    x = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(scale_grad_half(x), x)


def test_scale_grad_half_backward_halves_gradient():
    """scale_grad_half backward pass should halve the gradient."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from actors.learner_actor import scale_grad_half
    x = jnp.array([1.0, 2.0, 3.0])
    grad_fn = jax.grad(lambda v: scale_grad_half(v).sum())
    g = grad_fn(x)
    assert jnp.allclose(g, jnp.full_like(x, 0.5)), f"expected 0.5 everywhere, got {g}"
