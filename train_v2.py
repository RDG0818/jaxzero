"""
Asynchronous actor-learner training for Multi-Agent MuZero.

Architecture
------------
  LearnerActor  (1×, GPU)  trains model parameters, serves them to actors
  DataActor     (N×, CPU)  runs MCTS episodes, ships replay items to buffer
  ReplayBufferActor (1×)   stores and samples prioritized experience

See actors/ for the actor implementations and training/ for the loop logic.
"""

import os
from typing import Tuple

import ray

from config import CONFIG
from utils.logging_utils import logger
from actors.replay_buffer_actor import ReplayBufferActor
from actors.learner_actor import LearnerActor
from actors.data_actor import DataActor
from training.loop import run_warmup, run_training_loop


@ray.remote
def _fetch_env_metadata() -> Tuple[int, int]:
    """
    Returns (obs_size, action_size) by instantiating a temporary env.

    Runs as a Ray task so JAX (imported transitively by MPEEnvWrapper) is
    never initialized in the main process, preserving GPU memory for LearnerActor.
    """
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from utils.mpe_env_wrapper import MPEEnvWrapper

    env = MPEEnvWrapper(
        CONFIG.train.env_name,
        CONFIG.train.num_agents,
        CONFIG.train.max_episode_steps,
    )
    return env.observation_size, env.action_space_size


def initialize_actors(obs_size: int, action_size: int):
    replay_buffer = ReplayBufferActor.options(num_cpus=4).remote(obs_size, action_size)
    learner = LearnerActor.remote(obs_size, action_size, replay_buffer)
    data_actors = [
        DataActor.remote(i, obs_size, action_size, learner, replay_buffer)
        for i in range(CONFIG.train.num_actors)
    ]
    return replay_buffer, learner, data_actors


def main():
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(ignore_reinit_error=True)
    logger.info(f"Ray resources: {ray.available_resources()}")

    if CONFIG.train.wandb_mode != "disabled":
        import wandb
        wandb.init(project=CONFIG.train.project_name, config=CONFIG)

    obs_size, action_size = ray.get(_fetch_env_metadata.remote())
    logger.info(f"Env: obs_size={obs_size}, action_size={action_size}")

    replay_buffer, learner, data_actors = initialize_actors(obs_size, action_size)

    logger.info("Compiling plan_fn (one warm-up episode per actor)...")
    ray.get([actor.run_episode.remote() for actor in data_actors])

    actor_tasks = run_warmup(data_actors, replay_buffer)
    run_training_loop(learner, data_actors, replay_buffer, actor_tasks)

    if CONFIG.train.wandb_mode != "disabled":
        import wandb
        wandb.finish()

    ray.shutdown()


if __name__ == "__main__":
    main()
