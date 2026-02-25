#!/usr/bin/env python3
"""
Test script for sweep manager - prints Hydra config parameters.
"""
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Print all parameters passed via Hydra config.
    """
    print("=" * 80)
    print("Hydra Configuration Parameters:")
    print("=" * 80)
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    for key, value in sorted(config_dict.items()):
        print(f"{key}: {value}")
    
    print("=" * 80)
    print("\nFull config as YAML:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)


if __name__ == "__main__":
    main()
