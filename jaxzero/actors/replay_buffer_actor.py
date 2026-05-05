import ray
from jaxzero.config import MAZeroConfig


@ray.remote
class ReplayBufferActor:
    """PrioritizedReplayBuffer in an isolated Ray process. No JAX required."""

    def __init__(self, config: MAZeroConfig):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
        from jaxzero.replay_buffer import PrioritizedReplayBuffer
        self._buf = PrioritizedReplayBuffer(config)

    def add(self, game) -> None:
        self._buf.add(game)

    def can_sample(self, batch_size: int) -> bool:
        return self._buf.can_sample(batch_size)

    def prepare_batch_context(self, batch_size: int, beta: float):
        """Returns (games, positions, indices, weights) or None if buffer not ready."""
        if not self._buf.can_sample(batch_size):
            return None
        return self._buf.prepare_batch_context(batch_size, beta)

    def update_priorities(self, indices, new_priorities) -> None:
        self._buf.update_priorities(indices, new_priorities)

    def get_size(self) -> int:
        return self._buf.size
