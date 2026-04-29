import jax
import jax.numpy as jnp
import chex


def h(x: chex.Array) -> chex.Array:
    """Invertible value transform from MuZero (Appendix F)."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + 0.001 * x


def inv_h(x: chex.Array) -> chex.Array:
    """Inverse of h transform."""
    eps = 0.001
    # More numerically stable formulation
    u = jnp.abs(x)
    sqrt_term = jnp.sqrt(1 + 4 * eps * (u + 1 + eps))
    # Use better numerical stability for the division
    v = (sqrt_term - 1) / (2 * eps)
    return jnp.sign(x) * (v ** 2 - 1)


def phi(x: chex.Array, support_size: int) -> chex.Array:
    """Encode scalar(s) to categorical distribution over support [-S, S].

    Args:
        x: shape (B,) scalars — must be 1-D
        support_size: S, so support has 2*S+1 bins

    Returns:
        shape (B, 2*S+1) soft encoding
    """
    x = jnp.atleast_1d(x)
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
        logits: shape (B, 2*S+1) unnormalized logits (e.g. network output)
        support_size: S

    Returns:
        shape (B,) scalars
    """
    support = jnp.arange(-support_size, support_size + 1, dtype=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.dot(probs, support)
