from hydra import compose, initialize
from omegaconf import DictConfig


def load_config(config_name: str, overrides: list[str]) -> DictConfig:
    with initialize(config_path="../cofig", version_base=None):
        return compose(config_name, overrides=overrides)
