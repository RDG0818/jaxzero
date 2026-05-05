import os
import numpy as np
import ray
from jaxzero.config import MAZeroConfig


def _make_envs(config: MAZeroConfig):
    """Reconstruct envs from config.env_name inside the actor process."""
    if config.env_name == "3m":
        from jaxzero.envs.smax_wrapper import SMAXWrapper
        return [
            SMAXWrapper(map_name="3m", stacked_observations=config.stacked_observations)
            for _ in range(config.num_envs_parallel)
        ]
    elif config.env_name == "mpe":
        from jaxzero.envs.mpe_wrapper import MPEWrapper
        return [MPEWrapper() for _ in range(config.num_envs_parallel)]
    else:
        raise ValueError(f"Unknown env_name: {config.env_name!r}")


@ray.remote(num_cpus=1)
class DataActor:
    """Runs ctree MCTS collection on CPU. Ships GameHistory objects to ReplayBufferActor."""

    def __init__(self, actor_id: int, config: MAZeroConfig, learner_actor, replay_buffer_actor):
        # Must set BEFORE any JAX import
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")

        import jax  # noqa: F401 — triggers JAX init after env vars are set
        from jaxzero.model.networks import MAMuZeroNet
        from jaxzero.mcts.sampled_mcts import SampledMCTS

        self.actor_id = actor_id
        self.config = config
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor
        self.episodes_since_update = 0
        self._param_future = None
        self.np_rng = np.random.default_rng(actor_id * 1337 + 7)

        self.envs = _make_envs(config)

        net = MAMuZeroNet(config=config)
        self.mcts = SampledMCTS(config=config, model=net)

        # Fetch initial params synchronously
        self.params = ray.get(learner_actor.get_params.remote())

        # Warmup: JIT compile initial + recurrent inference at collection batch size
        _obs = np.ones(
            (config.num_envs_parallel, config.num_agents, config.obs_size), dtype=np.float32
        )
        _legal = np.ones(
            (config.num_envs_parallel, config.num_agents, config.action_space_size), dtype=bool
        )
        self.mcts.search(self.params, _obs, _legal, np.random.default_rng(0))

    def run_episode(self) -> float:
        """Collect one batch of parallel episodes. Returns mean episode return."""
        import jax.random as jr
        from jaxzero.train import collect_episodes_parallel

        rng_key = jr.PRNGKey(int(self.np_rng.integers(0, 2**31 - 1)))
        games = collect_episodes_parallel(
            self.envs, self.mcts, self.params, self.config, rng_key
        )

        for game in games:
            self.replay_buffer.add.remote(game)

        self.episodes_since_update += len(games)

        # Non-blocking check: resolve in-flight param fetch if ready
        if self._param_future is not None:
            ready, _ = ray.wait([self._param_future], timeout=0)
            if ready:
                self.params = ray.get(self._param_future)
                self._param_future = None
                self.episodes_since_update = 0

        # Fire new async param fetch if due
        if (self._param_future is None
                and self.episodes_since_update >= self.config.param_update_interval):
            self._param_future = self.learner.get_params.remote()

        return float(np.mean([sum(g.rewards) for g in games]))
