# utils/transforms.py
import jax
import jax.numpy as jnp
from typing import NamedTuple


class DiscreteSupport(NamedTuple):
    """Discrete support for categorical value/reward distributions."""
    min: int
    max: int

    @property
    def size(self) -> int:
        return self.max - self.min + 1


def muzero_scale(x: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """
    MuZero value scaling function — reduces the scale of large rewards/values
    to improve numerical stability during training.

    Reference: Appendix A, https://arxiv.org/pdf/1805.11593.pdf
    """
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1 + epsilon * x)


def muzero_scale_inv(x: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """Inverse of muzero_scale — converts scaled values back to original scale."""
    sign = jnp.sign(x)
    sqrt_term = jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon))
    numerator = sqrt_term - 1
    denominator = 2 * epsilon
    squared_term = (numerator / denominator) ** 2
    return sign * (squared_term - 1)


def scalar_to_support(scalar: jnp.ndarray, support: DiscreteSupport) -> jnp.ndarray:
    """
    Encodes a scalar value into a two-hot categorical distribution over the support.

    Applies muzero_scale first, then distributes probability mass between the
    two nearest support atoms via linear interpolation.

    Args:
        scalar: Scalar values to encode. Any shape.
        support: DiscreteSupport defining the range.

    Returns:
        Categorical distribution. Shape: (*scalar.shape, support.size)
    """
    scaled_scalar = muzero_scale(scalar)
    clipped_scalar = jnp.clip(scaled_scalar, support.min, support.max)

    floor = jnp.floor(clipped_scalar).astype(jnp.int32)
    ceil = jnp.ceil(clipped_scalar).astype(jnp.int32)
    prob = clipped_scalar - floor

    floor_indices = (floor - support.min).astype(jnp.int32)
    ceil_indices = (ceil - support.min).astype(jnp.int32)

    floor_one_hot = jax.nn.one_hot(floor_indices, num_classes=support.size)
    ceil_one_hot = jax.nn.one_hot(ceil_indices, num_classes=support.size)

    return floor_one_hot * (1 - prob)[..., None] + ceil_one_hot * prob[..., None]


def support_to_scalar(distribution: jnp.ndarray, support: DiscreteSupport) -> jnp.ndarray:
    """
    Decodes a categorical distribution (or logits) back to a scalar value.

    Applies softmax to convert logits to probabilities, computes the expected
    value over the support atoms, then inverts the muzero_scale transform.

    Args:
        distribution: Logits or probabilities over the support.
                      Shape: (*batch_shape, support.size)
        support: DiscreteSupport defining the range.

    Returns:
        Scalar values. Shape: (*batch_shape,)
    """
    probs = jax.nn.softmax(distribution, axis=-1)
    support_range = jnp.arange(support.min, support.max + 1, dtype=jnp.float32)
    scalar = jnp.sum(probs * jnp.broadcast_to(support_range, probs.shape), axis=-1)
    return muzero_scale_inv(scalar)
