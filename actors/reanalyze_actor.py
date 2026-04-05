import os
import numpy as np

import ray

from config import ExperimentConfig
from utils.logging_utils import logger
from utils.profiler import Profiler


@ray.remote(num_cpus=1)
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

        import jax
        from model import FlaxMAMuZeroNet
        from mcts import MCTSIndependentPlanner, MCTSJointPlanner

        self.actor_id = actor_id
        self.config = config
        self.learner = learner_actor
        self.replay_buffer = replay_buffer_actor

        self.rng_key = jax.random.PRNGKey(actor_id * 2718 + 42)

        model = FlaxMAMuZeroNet(config.model, action_size)
        planner_map = {"independent": MCTSIndependentPlanner, "joint": MCTSJointPlanner}
        if config.mcts.planner_mode not in planner_map:
            raise ValueError(
                f"Unknown planner_mode '{config.mcts.planner_mode}'. "
                f"Choose from: {list(planner_map)}"
            )
        planner = planner_map[config.mcts.planner_mode](model=model, config=config)
        self.plan_fn = jax.jit(planner.plan)
        self.params = ray.get(learner_actor.get_params.remote())
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

        with self.profiler.time("param_sync"):
            self.params = ray.get(self.learner.get_params.remote())

        with self.profiler.time("mcts_plan"):
            self.rng_key, plan_key = jax.random.split(self.rng_key)
            plan_output = self.plan_fn(self.params, plan_key, jnp.array(observations))
            jax.block_until_ready(plan_output.policy_targets)

        with self.profiler.time("buffer_update"):
            self.replay_buffer.update_targets.remote(
                indices,
                np.array(plan_output.policy_targets),
                np.array(plan_output.root_value),
            )

        self.profiler.step()
