from pathlib import Path
from typing import Any, Union

import yaml


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    return config
