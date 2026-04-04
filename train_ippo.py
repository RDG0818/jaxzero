"""
IPPO baseline training entry point.

Usage:
  python train_ippo.py
  python train_ippo.py num_envs=64 total_timesteps=5000000
  python train_ippo.py env_name=MPE_simple_tag_v3 num_agents=4
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from baselines.ippo import IPPOConfig, run_training_loop


@hydra.main(version_base=None, config_path="configs/baseline", config_name="ippo")
def main(cfg: DictConfig):
    raw = OmegaConf.to_container(cfg, resolve=True)
    # hidden_sizes comes in as a list from YAML; convert to tuple for dataclass
    if "hidden_sizes" in raw:
        raw["hidden_sizes"] = tuple(raw["hidden_sizes"])
    config = IPPOConfig(**raw)
    run_training_loop(config)


if __name__ == "__main__":
    main()
