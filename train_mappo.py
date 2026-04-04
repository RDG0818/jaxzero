"""
MAPPO baseline training entry point.

Usage:
  python train_mappo.py
  python train_mappo.py num_envs=64 total_timesteps=5000000
  python train_mappo.py env_name=MPE_simple_tag_v3 num_agents=4
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from baselines.mappo import MAPPOConfig, run_training_loop


@hydra.main(version_base=None, config_path="configs/baseline", config_name="mappo")
def main(cfg: DictConfig):
    raw = OmegaConf.to_container(cfg, resolve=True)
    if "hidden_sizes" in raw:
        raw["hidden_sizes"] = tuple(raw["hidden_sizes"])
    config = MAPPOConfig(**raw)
    run_training_loop(config)


if __name__ == "__main__":
    main()
