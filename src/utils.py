from typing import Any

from dacite import from_dict
from omegaconf import DictConfig, OmegaConf


def parse_dict_config(dataclass: Any, dict_config: DictConfig) -> Any:
    config_dict = OmegaConf.to_container(dict_config, resolve=True)
    return from_dict(data_class=dataclass, data=config_dict)  # type: ignore
