import numpy as np
from collections import deque
from jaxzero.config import MAZeroConfig
from jaxzero.game import GameHistory


class PrioritizedReplayBuffer:
    def __init__(self, config: MAZeroConfig):
        self.config = config
        self.capacity = config.replay_buffer_size
        self.alpha = config.priority_alpha
        self._games: deque[GameHistory] = deque(maxlen=self.capacity)
        self._priorities: deque[float] = deque(maxlen=self.capacity)

    @property
    def size(self) -> int:
        return len(self._games)

    def add(self, game: GameHistory, priority: float = 1.0):
        self._games.append(game)
        self._priorities.append(priority)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= max(batch_size, self.config.min_replay_size)

    def prepare_batch_context(
        self, batch_size: int, beta: float
    ) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        priorities = np.array(self._priorities, dtype=np.float64)
        probs = (priorities ** self.alpha)
        probs /= probs.sum()

        game_indices = np.random.choice(len(self._games), size=batch_size, p=probs)
        games = [self._games[i] for i in game_indices]
        positions = np.array([
            np.random.randint(0, max(1, len(games[b])))
            for b in range(batch_size)
        ])

        min_prob = probs.min()
        max_weight = (len(self._games) * min_prob) ** (-beta)
        weights = ((len(self._games) * probs[game_indices]) ** (-beta)) / max_weight
        weights = weights.astype(np.float32)

        return games, positions, game_indices, weights

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
        for idx, p in zip(indices, new_priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = float(abs(p)) + 1e-6
