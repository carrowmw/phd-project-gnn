# gnn_package/__init__.py
from .config import paths
from .src.utils import data_utils, sensor_utils
from .src.utils.exceptions import GNNException
from .src.utils.logging_utils import configure_logging, get_logger
from .src.utils.model_io import load_model, save_model
from .src.utils.metrics import calculate_error_metrics, format_prediction_results
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
    "load_model",
    "save_model",
    "calculate_error_metrics",
    "format_prediction_results",
]
