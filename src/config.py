"""Configuration loader for the Speaker Identification project."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Config loaded from {path}")
        return cfg
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Failed to parse config: {exc}")
        raise
