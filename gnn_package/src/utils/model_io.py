# src/utils/model_io.py

import os
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Callable

from gnn_package.src.models.factory import create_model
from gnn_package.config import ExperimentConfig, get_config
from gnn_package.src.models.registry import ModelRegistry
from gnn_package.src.utils.device_utils import get_device
from gnn_package.src.utils.device_utils import get_device_from_config
from gnn_package.src.utils.exceptions import ModelLoadError, ModelCreationError

logger = logging.getLogger(__name__)


def load_model(
    model_path: Union[str, Path],
    model_type: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    model_creator: Optional[Callable] = None,
    is_prediction_mode: bool = False
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a model with comprehensive error handling and device management.

    Parameters:
    -----------
    model_path : str or Path
        Path to the model file
    model_type : str, optional
        Type of model to load (e.g., "stgnn") - inferred from metadata if not provided
    config : ExperimentConfig, optional
        Configuration object. If None, attempts to find config in model directory
    device : str or torch.device, optional
        Device to load the model on - if None, determined from config or auto-detected
    strict : bool
        Whether to strictly enforce all keys matching when loading state dict
    model_creator : callable, optional
        Custom function to create the model instance
    is_prediction_mode : bool
        Whether the model is being loaded for prediction (vs. training/evaluation)

    Returns:
    --------
    tuple
        (loaded_model, metadata_dict)

    Raises:
    -------
    ModelLoadError
        If there's an error loading the model
    FileNotFoundError
        If model file or required configuration is not found
    """
    model_path = Path(model_path)

    # Check if model exists
    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")

    # Determine model directory and try to load config if not provided
    model_dir = model_path.parent
    metadata = {}

    # Try to load metadata if it exists
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                logger.info(f"Loaded metadata from: {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {str(e)}")

    # If model_type not specified, try to get from metadata
    if model_type is None and "model_type" in metadata:
        model_type = metadata["model_type"]
        logger.info(f"Using model type from metadata: {model_type}")

    # If no config provided and in prediction mode, strictly require a config file
    if config is None:
        config_path = model_dir / "config.yml"

        if config_path.exists():
            try:
                # Create a new configuration with prediction mode if needed
                config = ExperimentConfig(str(config_path), is_prediction_mode=is_prediction_mode)
                logger.info(f"Loaded configuration from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
                if is_prediction_mode:
                    raise FileNotFoundError(f"Failed to load valid configuration from {config_path} for prediction")
                # Otherwise, create a new default config as fallback
                config = ExperimentConfig("config.yml")
        else:
            # In prediction mode, we require a valid config
            if is_prediction_mode:
                raise FileNotFoundError(
                    f"No configuration file found at {config_path}. "
                    f"Please provide a configuration file for prediction."
                )
            # Otherwise, use default
            config = ExperimentConfig("config.yml")
            logger.info("Using default configuration")

    # Create model
    try:
        if model_type is not None:
            logger.debug(f"Creating model of type: {model_type}")
            model = ModelRegistry.create_model(model_type, config=config)
        else:
            # Use factory function
            model = create_model(config)
            logger.debug(f"Created model using factory function with architecture: {config.model.architecture}")
    except Exception as e:
        raise ModelCreationError(f"Failed to create model: {str(e)}") from e

    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Model weights loaded from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model weights: {str(e)}")
        raise ModelLoadError(f"Failed to load model weights: {str(e)}") from e

    # Set device
    if device is None:
        device = get_device_from_config(config)
    else:
        device = get_device(device) if isinstance(device, str) else device

    model = model.to(device)
    model.eval()  # Set to evaluation mode by default

    return model, metadata


def save_model(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    config: Optional[ExperimentConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
    save_config: bool = True,
) -> Dict[str, str]:
    """
    Save a model with comprehensive metadata.

    This is the central model saving function that should be used throughout the codebase.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to save
    output_path : str or Path
        Path where to save the model
    config : ExperimentConfig, optional
        Configuration to save alongside the model
    metadata : Dict[str, Any], optional
        Additional metadata to save
    save_config : bool
        Whether to save the configuration

    Returns:
    --------
    Dict[str, str]
        Dictionary of saved file paths
    """
    output_path = Path(output_path)

    # Determine if output_path is a directory or file
    if output_path.suffix == ".pth":
        model_path = output_path
        output_dir = output_path.parent
    else:
        # Assume it's a directory
        output_dir = output_path
        model_path = output_dir / "model.pth"

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare result dictionary
    saved_files = {"model": str(model_path)}

    # Save model state dictionary
    try:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

    # Save metadata if provided
    if metadata:
        metadata_path = output_dir / "metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            saved_files["metadata"] = str(metadata_path)
            logger.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")

    # Save configuration if requested
    if save_config:
        if config is None:
            config = get_config()

        config_path = output_dir / "config.yml"
        try:
            config.save(config_path)
            saved_files["config"] = str(config_path)
            logger.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")

    return saved_files