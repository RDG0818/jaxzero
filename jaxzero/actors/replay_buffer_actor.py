import ray
from jaxzero.config import MAZeroConfig


@ray.remote
class ReplayBufferActor:
    """PrioritizedReplayBuffer in an isolated Ray process. No JAX required."""

    def __init__(self, config: MAZeroConfig):
        import os
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
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

    def sample_for_reanalyze(self, batch_size: int):
        """Sample (game_idx, pos, obs, legal) for reanalysis."""
        if self._buf.size == 0:
            return None
        
        # Uniform sampling for reanalysis
        game_indices = np.random.randint(0, self._buf.size, size=batch_size)
        results = []
        for g_idx in game_indices:
            game = self._buf._games[g_idx]
            pos = np.random.randint(0, len(game))
            obs = game.obs(pos, self._buf.config.stacked_observations)
            # We need legal actions for the re-search
            legal = game.legal_actions[pos]
            results.append((g_idx, pos, obs, legal))
        
        return results

    def update_reanalyzed_stats(self, results: list):
        """Apply fresh MCTS results back to buffer."""
        for res in results:
            self._buf.update_reanalyzed_stats(
                res["game_idx"],
                res["pos"],
                res["policy"],
                res["qvalues"],
                res["actions"],
                res["mask"],
                res["root_value"]
            )

    def get_size(self) -> int:
        return self._buf.size
