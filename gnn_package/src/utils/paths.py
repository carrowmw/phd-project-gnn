# Defines all of the paths used in the project

from pathlib import Path
from typing import Dict
import os

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
assert ROOT_DIR.exists(), f"Invalid ROOT_DIR: {ROOT_DIR}"
assert (
    ROOT_DIR.name == "phd-project-gnn"
), f"Invalid ROOT_DIR - Check Parents: {ROOT_DIR}"

# Package directory

PACKAGE_DIR = ROOT_DIR / "gnn_package"

# Source directory
SRC_DIR = PACKAGE_DIR / "src"

# Data directory
DATA_DIR = PACKAGE_DIR / "data"

# Urban Observatory data directory
SENSORS_DATA_DIR = DATA_DIR / "sensors"

# Preprocessed data directories
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"
PREPROCESSED_GRAPH_DIR = PREPROCESSED_DATA_DIR / "graphs"

# Model directories


# Config directories


# Create directories if they don't exist
DIRS: Dict[str, Path] = {
    "data": DATA_DIR,
    "sensors": SENSORS_DATA_DIR,
    "preprocessed": PREPROCESSED_DATA_DIR,
    # "interim_data": INTERIM_DATA_DIR,
    # "models": MODELS_DIR,
    # "checkpoints": CHECKPOINTS_DIR,
    # "artifacts": ARTIFACTS_DIR,
    # "config": CONFIG_DIR,
    # "model_config": MODEL_CONFIG_DIR,
    # "data_config": DATA_CONFIG_DIR,
    # "results": RESULTS_DIR,
    # "figures": FIGURES_DIR,
    # "logs": LOGS_DIR
}

for dir_path in DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)


def get_path(name: str) -> Path:
    """Get path by name from DIRS dictionary."""
    if name not in DIRS:
        raise KeyError(f"Path '{name}' not found in DIRS dictionary.")
    return DIRS[name]


def add_path(name: str, path: Path) -> None:
    """Add new path to DIRS dictionary."""
    DIRS[name] = path
    path.mkdir(parents=True, exist_ok=True)
