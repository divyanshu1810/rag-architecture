"""
Helpers Module

Helper functions and common utilities.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/app.log",
) -> None:
    """Configure logging for the application."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logging.info("Logging initialized (level=%s, file=%s)", level, log_file)


def get_env_var(key: str, default: str = "") -> str:
    """Get an environment variable with an optional default."""
    value = os.getenv(key, default)
    if not value and not default:
        logging.warning(f"Environment variable '{key}' is not set")
    return value
