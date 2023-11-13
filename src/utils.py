from functools import wraps
from typing import Any, Callable

import pandas as pd
from dacite import from_dict
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def parse_dict_config(dataclass: Any, dict_config: DictConfig) -> Any:
    config_dict = OmegaConf.to_container(dict_config, resolve=True)
    return from_dict(data_class=dataclass, data=config_dict)  # type: ignore


def log_io_length(func: Callable) -> Any:
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Input length: {len(args[0])}")
        result = func(*args, **kwargs)
        logger.info(f"Output length: {len(result)}")
        return result

    return wrapper
