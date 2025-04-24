"""
Configuration utilities for the GNN package.

This module provides higher-level utilities for working with configuration
in various parts of the package, such as model loading/saving and prediction.
"""

import os
import copy
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, Tuple

import yaml
import torch

from gnn_package.config import ExperimentConfig, get_config, ConfigurationManager
from gnn_package.src.models.stgnn import create_stgnn_model


def load_model_for_prediction(
    model_path: Union[str, Path],
    config: Optional[ExperimentConfig] = None,
    override_params: Optional[Dict[str, Any]] = None,
    model_creator_func: Optional[Callable] = None,
) -> Tuple[torch.nn.Module, ExperimentConfig]:
    """
    Load a model with the appropriate configuration for prediction.

    This function handles all the configuration setup needed for loading a model
    for prediction, including finding and loading the right configuration file.

    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model file
    config : ExperimentConfig, optional
        Configuration to use. If None, attempts to find config in model directory
        or falls back to global config.
    override_params : Dict[str, Any], optional
        Parameters to override in the loaded configuration
    model_creator_func : Callable, optional
        Function to create model from config. If not provided, uses default creator

    Returns:
    --------
    tuple(torch.nn.Module, ExperimentConfig)
        The loaded model and its configuration
    """
    model_path = Path(model_path)

    # Configuration handling
    if config is None:
        # Try to find config in the model directory
        potential_config_path = model_path.parent / "config.yml"
        if potential_config_path.exists():
            config = ExperimentConfig(str(potential_config_path))
        else:
            # Fall back to global config
            config = get_config()

    # Convert to prediction config
    prediction_config = create_prediction_config_from_training(config, override_params)

    # Create model with correct architecture
    if model_creator_func is None:
        model_creator_func = create_stgnn_model

    model = model_creator_func(prediction_config)

    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode

    return model, prediction_config


def create_prediction_config_from_training(
    training_config: ExperimentConfig, override_params: Optional[Dict[str, Any]] = None
) -> ExperimentConfig:
    """
    Create a prediction-focused configuration from a training configuration.

    This function preserves model architecture and general data parameters
    but applies prediction-specific settings.

    Parameters:
    -----------
    training_config : ExperimentConfig
        The configuration used for training
    override_params : Dict[str, Any], optional
        Additional parameters to override in the configuration

    Returns:
    --------
    ExperimentConfig
        A new configuration optimized for prediction
    """
    # Prediction-specific parameter overrides
    prediction_overrides = {
        "data.training.use_cross_validation": False,
        "data.training.cv_split_index": 0,
    }

    # Combine with any provided overrides
    if override_params:
        for key, value in override_params.items():
            prediction_overrides[key] = value

    # Create prediction config using the ConfigurationManager
    return ConfigurationManager.create_prediction_config(
        base_config=training_config, override_params=prediction_overrides
    )


def save_model_with_config(
    model: torch.nn.Module, config: ExperimentConfig, path: Union[str, Path]
) -> None:
    """
    Save model and its configuration together.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to save
    config : ExperimentConfig
        The configuration used to create the model
    path : str or Path
        Directory path where to save the model and config
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), path / "model.pth")

    # Save configuration
    config.save(path / "config.yml")


def get_device_from_config(config: ExperimentConfig) -> torch.device:
    """
    Determine the appropriate device based on configuration and availability.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration object that may contain device information

    Returns:
    --------
    torch.device
        The device to use for model operations
    """
    # If device is specified in config, use it
    device_name = getattr(config.training, "device", None)

    if device_name:
        return torch.device(device_name)

    # Auto-detect best available device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def apply_environment_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """
    Apply configuration overrides from environment variables.

    This function looks for environment variables with the prefix "GNN_" and
    updates the configuration accordingly. For example, GNN_EPOCHS would update
    training.num_epochs.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration to update

    Returns:
    --------
    ExperimentConfig
        Updated configuration
    """
    # Create a temporary file with the current config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp:
        config.save(temp.name)
        temp_path = temp.name

    # Environment variable mappings
    env_mappings = {
        "GNN_EPOCHS": "training.num_epochs",
        "GNN_LEARNING_RATE": "training.learning_rate",
        "GNN_WEIGHT_DECAY": "training.weight_decay",
        "GNN_HIDDEN_DIM": "model.hidden_dim",
        "GNN_LAYERS": "model.num_layers",
        "GNN_GC_LAYERS": "model.num_gc_layers",
        "GNN_DROPOUT": "model.dropout",
        "GNN_WINDOW_SIZE": "data.general.window_size",
        "GNN_HORIZON": "data.general.horizon",
        "GNN_BATCH_SIZE": "data.general.batch_size",
        "GNN_DEVICE": "training.device",
    }

    # Collect overrides from environment
    overrides = {}
    for env_var, config_key in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]

            # Convert to appropriate type based on the key
            if config_key.endswith(
                (
                    "epochs",
                    "hidden_dim",
                    "layers",
                    "gc_layers",
                    "window_size",
                    "horizon",
                    "batch_size",
                )
            ):
                value = int(value)
            elif config_key.endswith(("learning_rate", "weight_decay", "dropout")):
                value = float(value)

            overrides[config_key] = value

    # Apply overrides if any were found
    if overrides:
        try:
            # Create updated config
            updated_config = ExperimentConfig(
                config_path=temp_path, override_params=overrides
            )

            # Clean up temp file
            os.unlink(temp_path)

            return updated_config
        except Exception as e:
            # Clean up and re-raise
            os.unlink(temp_path)
            raise ValueError(f"Error applying environment overrides: {e}") from e
    else:
        # No overrides, clean up and return original
        os.unlink(temp_path)
        return config


def extract_config_for_component(
    config: ExperimentConfig, component: str
) -> Dict[str, Any]:
    """
    Extract a subset of configuration specific to a component.

    This is useful for passing only relevant configuration to specific
    components, reducing coupling and potential errors.

    Parameters:
    -----------
    config : ExperimentConfig
        Full configuration object
    component : str
        Component name ("data", "model", "training", etc.)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing only the relevant configuration
    """
    if component == "data":
        return {
            "general": config._dataclass_to_dict(config.data.general),
            "training": config._dataclass_to_dict(config.data.training),
            "prediction": config._dataclass_to_dict(config.data.prediction),
        }
    elif component == "model":
        return config._dataclass_to_dict(config.model)
    elif component == "training":
        return config._dataclass_to_dict(config.training)
    elif component == "paths":
        return config._dataclass_to_dict(config.paths)
    elif component == "visualization":
        return config._dataclass_to_dict(config.visualization)
    else:
        raise ValueError(f"Unknown component: {component}")
