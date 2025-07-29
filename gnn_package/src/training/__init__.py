# src/training/__init__.py
# Import the generalized training components
from .base_trainer import BaseTrainer
from .trainers import TqdmTrainer
from .cross_validation import run_cross_validation
from .experiment_manager import run_experiment
from .prediction import (
    predict_with_model,
    format_predictions,
    predict_and_evaluate,
    fetch_data_for_prediction,
)

from gnn_package.src.utils.model_io import load_model, save_model

# Expose key functions and classes
__all__ = [
    # Base training classes
    "BaseTrainer",
    "TqdmTrainer",

    # High-level training functions
    "run_experiment",
    "run_cross_validation",

    # Prediction functions
    "predict_with_model",
    "format_predictions",
    "predict_and_evaluate",
    "fetch_data_for_prediction",

    # Model I/O utilities
    "load_model",
    "save_model",
]