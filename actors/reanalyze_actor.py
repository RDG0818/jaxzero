import os
import numpy as np

import ray

from config import ExperimentConfig
from utils.logging_utils import logger
from utils.profiler import Profiler


@ray.remote(num_cpus=2)
class ReanalyzeActor:
    """
    Re-runs MCTS on stored observations using the latest model params to
    produce fresher policy/value targets.

    Only position 0 (the root observation) of each stored ReplayItem sequence
    can be updated — intermediate unroll states are not stored in the buffer.
    Params are synced from LearnerActor on every call to run_reanalyze().
    """

    def __init__(
        self,
        actor_id: int,
        obs_size: int,
        action_size: int,
        learner_actor,
        replay_buffer_actor,
        config: ExperimentConfig,
    ):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["JAX_PLATFORMS"] = "cpu"

        # Match DataActor: limit XLA/BLAS threads to avoid thrashing with other actors.
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("XLA_FLAGS",
            (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
        )

        import jax
        from model import FlaxMAMuZeroNet
        from mcts import MCTSIndependentPlanner, MCTSJointPlanner, MCTSJointOSLAPlanner

        self.actor_id = actor_id
        self.config = config
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor

        self.rng_key = jax.random.PRNGKey(actor_id * 2718 + 42)

        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {
            "independent": MCTSIndependentPlanner,
            "joint": MCTSJointOSLAPlanner,
            "joint_legacy": MCTSJointPlanner,
        }
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map.keys())}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)
        self.plan_fn = jax.jit(planner.plan)
        result = ray.get(learner_actor.get_params.remote())
        self.params = result["params"]
        self.norm_state = result["norm_state"]
        self._param_future = None
        self.profiler = Profiler(f"reanalyze[{actor_id}]", log_interval=config.train.debug_interval)
        logger.info(f"(ReanalyzeActor {actor_id} pid={os.getpid()}) Setup complete.")

    def run_reanalyze(self):
        """
        Samples a batch of stored observations, re-runs MCTS with the latest
        params, and writes back fresh policy/value targets at position 0.
        """
        import jax
        import jax.numpy as jnp

        with self.profiler.time("sample_wait"):
            indices, observations, _ = ray.get(
                self.replay_buffer.sample_for_reanalysis.remote(
                    self.config.train.reanalyze_batch_size
                )
            )
        if indices is None:
            return

        # Resolve previous async param fetch (if ready), then fire the next one.
        with self.profiler.time("param_sync"):
            if self._param_future is not None:
                ready, _ = ray.wait([self._param_future], timeout=0)
                if ready:
                    result = ray.get(self._param_future)
                    self.params = result["params"]
                    self.norm_state = result["norm_state"]
                    self._param_future = None
            if self._param_future is None:
                self._param_future = self.learner.get_params.remote()

        with self.profiler.time("mcts_plan"):
            self.rng_key, plan_key = jax.random.split(self.rng_key)
            obs_arr = np.array(observations)
            if self.norm_state is not None:
                obs_arr = (obs_arr - self.norm_state["mean"]) / np.sqrt(
                    self.norm_state["var"] + 1e-5
                )
            plan_output = self.plan_fn(self.params, plan_key, jnp.array(obs_arr))
            jax.block_until_ready(plan_output.policy_targets)

        with self.profiler.time("buffer_update"):
            self.replay_buffer.update_targets.remote(
                indices,
                np.array(plan_output.policy_targets),
                np.array(plan_output.root_value),
            )

        self.profiler.step()
