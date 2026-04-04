import os
import numpy as np
from typing import List

import ray

from config import ExperimentConfig


@ray.remote
class ReplayBufferActor:
    """
    Manages prioritized experience replay using Flashbax.

    Flashbax provides a JAX-native segment tree for O(log n) priority updates
    and sampling, replacing the previous O(n) numpy recomputation.

    New items receive the current maximum recorded priority, ensuring they are
    sampled at least once before their TD error is known (standard PER behavior).
    """

    def __init__(self, obs_size: int, action_size: int, config: ExperimentConfig):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["JAX_PLATFORMS"] = "cpu"

        import jax
        import jax.numpy as jnp
        import flashbax as fbx

        self._jax = jax
        self._jnp = jnp

        U = config.train.unroll_steps
        N = config.train.num_agents
        self.capacity = config.train.replay_buffer_size
        self.alpha = config.train.replay_buffer_alpha
        self.beta_start = config.train.replay_buffer_beta_start
        self.beta_frames = config.train.replay_buffer_beta_frames
        self.frame_count = 0

        self.buffer = fbx.make_prioritised_item_buffer(
            max_length=self.capacity,
            min_length=config.train.batch_size,
            sample_batch_size=config.train.batch_size,
            add_batches=False,
            priority_exponent=self.alpha,
            device="cpu",
        )

        # Init buffer state from a fake item that defines all array shapes.
        fake_item = {
            "observation":   jnp.zeros((N, obs_size),    dtype=jnp.float32),
            "actions":       jnp.zeros((U, N),            dtype=jnp.int32),
            "policy_target": jnp.zeros((U + 1, N, action_size), dtype=jnp.float32),
            "value_target":  jnp.zeros((U + 1, N),        dtype=jnp.float32),
            "reward_target": jnp.zeros((U, N),             dtype=jnp.float32),
            "agent_order":   jnp.zeros((N,),               dtype=jnp.int32),
        }
        self.state = self.buffer.init(fake_item)
        self.rng_key = jax.random.PRNGKey(42)

    def add(self, items: list, priorities: List[float]):
        """
        Adds ReplayItems to the buffer. Explicit priorities are ignored —
        Flashbax assigns max_recorded_priority to new items automatically.
        """
        jnp = self._jnp
        for item in items:
            jax_item = {
                "observation":   jnp.asarray(item.observation,   dtype=jnp.float32),
                "actions":       jnp.asarray(item.actions,       dtype=jnp.int32),
                "policy_target": jnp.asarray(item.policy_target, dtype=jnp.float32),
                "value_target":  jnp.asarray(item.value_target,  dtype=jnp.float32),
                "reward_target": jnp.asarray(item.reward_target, dtype=jnp.float32),
                "agent_order":   jnp.asarray(item.agent_order,   dtype=jnp.int32),
            }
            self.state = self.buffer.add(self.state, jax_item)

    def sample(self, batch_size: int):
        """
        Samples a batch with prioritized experience replay.
        batch_size is fixed at construction; this argument is accepted for
        interface compatibility but ignored.

        Returns (batch, importance_weights, indices) or (None, None, None)
        if the buffer doesn't have enough items yet.
        """
        jax = self._jax
        jnp = self._jnp

        if not self.buffer.can_sample(self.state):
            return None, None, None

        self.rng_key, sample_key = jax.random.split(self.rng_key)
        s = self.buffer.sample(self.state, sample_key)

        n = int(jnp.where(self.state.is_full, self.capacity, self.state.current_index))
        beta = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        self.frame_count += 1

        # s.probabilities = p_i^alpha / sum(p_j^alpha) — ready for IS weight computation.
        is_weights = (n * s.probabilities) ** (-beta)
        is_weights = is_weights / is_weights.max()

        from utils.replay_buffer import ReplayItem
        exp = s.experience
        batch = ReplayItem(
            observation=np.array(exp["observation"]),
            actions=np.array(exp["actions"]),
            policy_target=np.array(exp["policy_target"]),
            value_target=np.array(exp["value_target"]),
            reward_target=np.array(exp["reward_target"]),
            agent_order=np.array(exp["agent_order"]),
        )
        return batch, np.array(is_weights), np.array(s.indices)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates segment tree priorities. Raw TD errors are passed; alpha is applied internally."""
        jnp = self._jnp
        self.state = self.buffer.set_priorities(
            self.state,
            jnp.asarray(indices),
            jnp.asarray(priorities, dtype=jnp.float32),
            priority_exponent=self.alpha,
        )

    def get_size(self) -> int:
        jnp = self._jnp
        return int(jnp.where(self.state.is_full, self.capacity, self.state.current_index))

    def get_stats(self) -> dict:
        jnp = self._jnp
        size = self.get_size()
        if size == 0:
            return {"size": 0, "capacity": self.capacity, "fill_pct": 0.0}
        beta = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        return {
            "size": size,
            "capacity": self.capacity,
            "fill_pct": 100.0 * size / self.capacity,
            "max_recorded_priority": float(self.state.sum_tree_state.max_recorded_priority),
            "beta": beta,
        }
