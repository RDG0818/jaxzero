"""
Asynchronous actor-learner training for Multi-Agent MuZero.

Architecture
------------
  LearnerActor      (1×, GPU)  trains model parameters, serves them to actors
  DataActor         (N×, CPU)  runs MCTS episodes, ships replay items to buffer
  ReplayBufferActor (1×)       stores and samples prioritized experience

See actors/ for the actor implementations and training/ for the loop logic.

Usage
-----
  python train.py                                  # default config
  python train.py mcts=joint                       # switch to joint planner
  python train.py train.num_episodes=50000         # override single value
  python train.py train.batch_size=128 mcts.num_simulations=50
"""

import os
from typing import Tuple

import ray
import hydra
from omegaconf import DictConfig, OmegaConf

from config import ExperimentConfig, ModelConfig, MCTSConfig, TrainConfig
from utils.logging_utils import logger
from actors import ReplayBufferActor, LearnerActor, DataActor
from training import run_warmup, run_training_loop


@ray.remote
def _fetch_env_metadata(config: ExperimentConfig) -> Tuple[int, int]:
    """
    Returns (obs_size, action_size) by instantiating a temporary env.

    Runs as a Ray task so JAX (imported transitively by MPEEnvWrapper) is
    never initialized in the main process, preserving GPU memory for LearnerActor.
    """
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    from envs import MPEEnvWrapper

    env = MPEEnvWrapper(
        config.train.env_name,
        config.train.num_agents,
        config.train.max_episode_steps,
    )
    return env.observation_size, env.action_space_size


def initialize_actors(obs_size: int, action_size: int, config: ExperimentConfig):
    replay_buffer = ReplayBufferActor.options(num_cpus=4).remote(obs_size, action_size, config)
    learner = LearnerActor.remote(obs_size, action_size, replay_buffer, config)
    data_actors = [
        DataActor.remote(i, obs_size, action_size, learner, replay_buffer, config)
        for i in range(config.train.num_actors)
    ]
    return replay_buffer, learner, data_actors


def _build_config(cfg: DictConfig) -> ExperimentConfig:
    """Converts a Hydra DictConfig into a typed ExperimentConfig dataclass."""
    return ExperimentConfig(
        model=ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True)),
        mcts=MCTSConfig(**OmegaConf.to_container(cfg.mcts, resolve=True)),
        train=TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True)),
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    config = _build_config(cfg)

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(ignore_reinit_error=True)
    logger.info(f"Ray resources: {ray.available_resources()}")

    if config.train.wandb_mode != "disabled":
        import wandb
        wandb.init(project=config.train.project_name, config=OmegaConf.to_container(cfg))

    obs_size, action_size = ray.get(_fetch_env_metadata.remote(config))
    logger.info(f"Env: obs_size={obs_size}, action_size={action_size}")

    replay_buffer, learner, data_actors = initialize_actors(obs_size, action_size, config)

    logger.info("Compiling plan_fn (one warm-up episode per actor)...")
    ray.get([actor.run_episode.remote() for actor in data_actors])

    actor_tasks = run_warmup(data_actors, replay_buffer, config)
    run_training_loop(learner, data_actors, replay_buffer, actor_tasks, config)

    if config.train.wandb_mode != "disabled":
        import wandb
        wandb.finish()

    ray.shutdown()


if __name__ == "__main__":
    main()

