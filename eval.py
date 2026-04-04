"""
Standalone evaluation script for a trained MuZero checkpoint.

Loads a checkpoint, runs N episodes with MCTS (no gradient updates), and
reports mean ± std episode return. Useful for comparing checkpoints or
runs without re-training.

Usage:
  python eval.py                                    # default config + latest checkpoint
  python eval.py train.checkpoint_dir=runs/myrun   # specific run directory
  python eval.py eval_episodes=200                  # more episodes for tighter estimate
  python eval.py train.num_simulations=100          # more MCTS sims for eval
"""

import os
import time
import numpy as np

import ray
import hydra
from omegaconf import DictConfig, OmegaConf

from config import ExperimentConfig, ModelConfig, MCTSConfig, TrainConfig
from utils.logging_utils import logger


@ray.remote
def _run_eval(obs_size: int, action_size: int, config: ExperimentConfig, num_episodes: int) -> list:
    """
    Runs `num_episodes` evaluation episodes in a Ray task (keeps JAX off the
    main process, same pattern as DataActor and _fetch_env_metadata).

    Returns a list of undiscounted episode returns.
    """
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from pathlib import Path
    from model import FlaxMAMuZeroNet
    from mcts import MCTSIndependentPlanner, MCTSJointPlanner
    from envs import VecMPEEnvWrapper

    # Load checkpoint.
    ckpt_dir = Path(config.train.checkpoint_dir).absolute()
    ckpt_manager = ocp.CheckpointManager(ckpt_dir)
    latest = ckpt_manager.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    model = FlaxMAMuZeroNet(config.model, action_size)
    dummy_obs = jnp.ones((1, config.train.num_agents, obs_size))
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_obs)["params"]

    target = {"params": params, "step": np.array(0)}
    restored = ckpt_manager.restore(latest, args=ocp.args.StandardRestore(target))
    params = restored["params"]
    step = int(restored["step"])

    planner_map = {"independent": MCTSIndependentPlanner, "joint": MCTSJointPlanner}
    planner = planner_map[config.mcts.planner_mode](model=model, config=config)
    plan_fn = jax.jit(planner.plan)

    B = config.train.num_envs_per_actor
    env = VecMPEEnvWrapper(
        config.train.env_name,
        config.train.num_agents,
        config.train.max_episode_steps,
        B,
    )

    returns = []
    episodes_done = 0

    while episodes_done < num_episodes:
        rng, *reset_keys_list = jax.random.split(rng, B + 1)
        reset_keys = jnp.stack(reset_keys_list)
        observations, states = env.reset(reset_keys)
        episode_returns = np.zeros(B)
        active = np.ones(B, dtype=bool)

        for _ in range(config.train.max_episode_steps):
            rng, plan_key, step_key = jax.random.split(rng, 3)
            plan_output = plan_fn(params, plan_key, observations)
            actions_np = np.array(plan_output.joint_action)

            step_keys = jax.random.split(step_key, B)
            next_obs, next_states, rewards, dones = env.step(step_keys, states, actions_np)
            rewards_np = np.array(rewards)
            dones_np = np.array(dones)

            episode_returns += rewards_np * active
            active &= ~dones_np
            observations = next_obs
            states = next_states

            if not active.any():
                break

        returns.extend(episode_returns.tolist())
        episodes_done += B

    return returns, step


def _build_config(cfg: DictConfig) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True)),
        mcts=MCTSConfig(**OmegaConf.to_container(cfg.mcts, resolve=True)),
        train=TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True)),
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    num_episodes = cfg.get("eval_episodes", 100)
    config = _build_config(cfg)

    ray.init(ignore_reinit_error=True)

    # Fetch env metadata (same pattern as train.py).
    from train import _fetch_env_metadata
    obs_size, action_size = ray.get(_fetch_env_metadata.remote(config))

    logger.info(
        f"Evaluating checkpoint from '{config.train.checkpoint_dir}' "
        f"over {num_episodes} episodes..."
    )
    t0 = time.monotonic()
    returns, ckpt_step = ray.get(
        _run_eval.remote(obs_size, action_size, config, num_episodes)
    )
    elapsed = time.monotonic() - t0

    returns = np.array(returns)
    logger.info(
        f"Checkpoint step:  {ckpt_step}\n"
        f"Episodes:         {len(returns)}\n"
        f"Mean return:      {returns.mean():.3f}\n"
        f"Std  return:      {returns.std():.3f}\n"
        f"Min  return:      {returns.min():.3f}\n"
        f"Max  return:      {returns.max():.3f}\n"
        f"Elapsed:          {elapsed:.1f}s"
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
