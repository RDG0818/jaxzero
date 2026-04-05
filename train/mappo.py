"""
MAPPO baseline training entry point.

Usage:
  python train/mappo.py
  python train/mappo.py NUM_ENVS=32 TOTAL_TIMESTEPS=5000000
  python train/mappo.py ENV_NAME=MPE_simple_tag_v3
"""

import os
import sys

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import hydra
from omegaconf import OmegaConf

from baselines.mappo import make_train


@hydra.main(version_base=None, config_path="../configs/baseline", config_name="mappo")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    if config.get("WANDB_MODE", "disabled") != "disabled":
        import wandb
        wandb.init(
            entity=config.get("ENTITY"),
            project=config["PROJECT"],
            tags=["MAPPO", "RNN"],
            config=config,
            mode=config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    if config.get("WANDB_MODE", "disabled") != "disabled":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
