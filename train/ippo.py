"""
IPPO baseline training entry point.

Usage:
  python train/ippo.py
  python train/ippo.py NUM_ENVS=32 TOTAL_TIMESTEPS=5000000
  python train/ippo.py ENV_NAME=MPE_simple_tag_v3 NUM_SEEDS=3
"""

import os
import sys

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import hydra
from omegaconf import OmegaConf

from baselines.ippo import make_train


@hydra.main(version_base=None, config_path="../configs/baseline", config_name="ippo")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    if config.get("WANDB_MODE", "disabled") != "disabled":
        import wandb
        wandb.init(
            entity=config.get("ENTITY"),
            project=config["PROJECT"],
            tags=["IPPO", "FF"],
            config=config,
            mode=config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    # Print final mean return across seeds
    returns = out["metrics"]["returned_episode_returns"]
    print(f"Final mean return: {returns.mean(axis=0)[-1]:.3f}")

    if config.get("WANDB_MODE", "disabled") != "disabled":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
