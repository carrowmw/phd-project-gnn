# gnn_package/__init__.py (UPDATED)
from .config import paths
from .src.utils import data_utils, sensor_utils
from .src.utils.exceptions import GNNException
from .src.utils.logging_utils import configure_logging, get_logger
from .config.paths import *
from .src import preprocessing
from .src import dataloaders
from .src import models
from .src import training

# Configure default logging
logger = configure_logging()

__all__ = [
    # Core modules
    "paths",
    "data_utils",
    "preprocessing",
    "dataloaders",
    "models",
    "training",
    # Utilities
    "sensor_utils",
    "GNNException",
    "configure_logging",
    "get_logger",
]
