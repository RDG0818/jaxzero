# utils/obs_norm.py
"""
Online EMA normalizer for observations.

Maintains a running mean and variance over the per-feature observation
statistics using exponential moving averages.  Updated on each training batch
(over all (B * N, obs_size) observations); read-only during MCTS data
collection.

The normalizer state is serializable (plain numpy dicts) so it can be synced
to DataActors and ReanalyzeActors alongside model parameters via get_params().

Usage in LearnerActor._train_step():
    self.obs_norm.update(batch.observation)       # update running stats
    batch = replace(batch, observation=self.obs_norm.normalize(batch.observation))
    # then device_put and train_step as usual

Usage in DataActor.run_episode():
    obs_to_plan = self.obs_norm.normalize(np.array(observations))
    plan_output = self.plan_fn(self.params, plan_key, jnp.array(obs_to_plan))
"""
import numpy as np


class ObsRunningNorm:
    """
    Per-feature EMA normalizer for observations.

    Statistics are computed over the flattened (batch * agents, obs_size)
    tensor so that all agents' observations contribute equally regardless of
    the batch dimension layout.

    Args:
        obs_size:  Length of a single agent observation vector.
        momentum:  EMA decay rate.  0.99 = slow adaptation (stable),
                   0.9 = faster adaptation (tracks non-stationarity).
        epsilon:   Small constant added to variance for numerical stability.
    """

    def __init__(self, obs_size: int, momentum: float = 0.99, epsilon: float = 1e-5):
        self.obs_size = obs_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.mean = np.zeros(obs_size, dtype=np.float32)
        self.var = np.ones(obs_size, dtype=np.float32)
        self._initialized = False

    # ------------------------------------------------------------------
    # Update / normalize
    # ------------------------------------------------------------------

    def update(self, obs: np.ndarray) -> None:
        """
        Update running statistics from a batch of observations.

        Args:
            obs: Any shape ending in obs_size, e.g. (B, N, obs_size).
        """
        flat = obs.reshape(-1, self.obs_size).astype(np.float32)
        batch_mean = flat.mean(axis=0)
        batch_var = flat.var(axis=0)

        if not self._initialized:
            # Cold start: use batch stats directly so the first normalization
            # is already sensible instead of dividing by 1.0 everywhere.
            self.mean = batch_mean
            self.var = np.maximum(batch_var, self.epsilon)
            self._initialized = True
        else:
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1.0 - self.momentum) * np.maximum(batch_var, self.epsilon)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations to approximately zero mean, unit variance.

        Args:
            obs: Any shape ending in obs_size, e.g. (B, N, obs_size).

        Returns:
            Normalized array of the same shape and dtype=float32.
        """
        obs = obs.astype(np.float32)
        return (obs - self.mean) / np.sqrt(self.var + self.epsilon)

    # ------------------------------------------------------------------
    # Serialization (for syncing to DataActors alongside model params)
    # ------------------------------------------------------------------

    def state(self) -> dict:
        """Returns a serializable snapshot of the running statistics."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "initialized": self._initialized,
        }

    @classmethod
    def from_state(
        cls,
        state: dict,
        obs_size: int,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
    ) -> "ObsRunningNorm":
        """Reconstruct a normalizer from a previously serialized state dict."""
        norm = cls(obs_size, momentum, epsilon)
        norm.mean = state["mean"].copy()
        norm.var = state["var"].copy()
        norm._initialized = bool(state.get("initialized", True))
        return norm
