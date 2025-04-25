# src/utils/model_io.py

import os
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Callable

from gnn_package.config import ExperimentConfig, get_config
from gnn_package.src.models.registry import ModelRegistry
from gnn_package.src.utils.device_utils import get_device
from gnn_package.src.utils.exceptions import ModelLoadError, ModelCreationError

logger = logging.getLogger(__name__)


def load_model(
    model_path: Union[str, Path],
    model_type: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    model_creator: Optional[Callable] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a model with comprehensive error handling and device management.

    This is the central model loading function that should be used throughout the codebase.

    Parameters:
    -----------
    model_path : str or Path
        Path to the model file (.pth)
    model_type : str, optional
        Type of model to load (used with ModelRegistry)
    config : ExperimentConfig, optional
        Configuration to use. If None, attempts to find config in model directory.
    device : str or torch.device, optional
        Device to load the model onto. If None, determined from config.
    strict : bool
        Whether to strictly enforce that the keys in state_dict match model
    model_creator : Callable, optional
        Function to create the model. If provided, overrides model_type.

    Returns:
    --------
    Tuple[torch.nn.Module, Dict[str, Any]]
        The loaded model and metadata dictionary

    Raises:
    -------
    ModelLoadError
        If the model cannot be loaded
    """
    from gnn_package.src.utils.exceptions import safe_execute

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

    # Try to load config if not provided
    if config is None:
        config_path = model_dir / "config.yml"
        if config_path.exists():
            try:
                config = ExperimentConfig(config_path)
                logger.info(f"Loaded configuration from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
                config = get_config()
        else:
            config = get_config()
            logger.info("Using global configuration")

    # Create model
    try:
        if model_creator is not None:
            logger.info("Creating model with provided creator function")
            model = model_creator(config=config)
        elif model_type is not None:
            logger.info(f"Creating model of type: {model_type}")
            model = ModelRegistry.create_model(model_type, config=config)
        else:
            # Default to STGNN if nothing else specified
            from gnn_package.src.models.stgnn import create_stgnn_model

            logger.info("Creating default STGNN model")
            model = create_stgnn_model(config=config)
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
        from gnn_package.src.utils.device_utils import get_device_from_config

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


def load_model_partial(
    model_path: Union[str, Path], model: torch.nn.Module, device
) -> torch.nn.Module:
    """
    Attempt to load a model with partial state dict (for recovery).

    Parameters:
    -----------
    model_path : str or Path
        Path to the model file
    model : torch.nn.Module
        Model instance to load partial weights into
    device : torch.device
        Device to move model to after loading

    Returns:
    --------
    torch.nn.Module
        Model with partially loaded weights
    """
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        # Filter state_dict to only include keys that exist in the model
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

        # Log differences
        missing_keys = [k for k in model.state_dict().keys() if k not in state_dict]
        extra_keys = [k for k in state_dict.keys() if k not in model_keys]

        if missing_keys:
            logger.warning(f"Missing keys in loaded model: {missing_keys}")
        if extra_keys:
            logger.warning(f"Extra keys in loaded model (ignored): {extra_keys}")

        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(
            f"Partial model loading successful with {len(filtered_state_dict)}/{len(model_keys)} keys"
        )

        return model.to(device)
    except Exception as e:
        logger.error(f"Failed even with partial loading: {str(e)}")
        raise RuntimeError("Could not perform partial model loading") from e


def get_model_config(model_path: Union[str, Path]) -> ExperimentConfig:
    """
    Extract configuration for a model from its directory or create a default one.

    Parameters:
    -----------
    model_path : str or Path
        Path to the model file

    Returns:
    --------
    ExperimentConfig
        Configuration for the model
    """
    model_path = Path(model_path)
    model_dir = model_path.parent

    # Look for config in standard locations
    config_paths = [
        model_dir / "config.yml",
        model_dir / "model_config.yml",
        model_dir.parent / "config.yml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"Found model configuration at: {config_path}")
            return ExperimentConfig(config_path)

    # Fall back to global config
    logger.warning(
        f"No configuration found for model {model_path}, using global config"
    )
    return get_config()


def create_model_by_type(model_type: str, config: ExperimentConfig) -> torch.nn.Module:
    """
    Create a model instance by type name.

    Parameters:
    -----------
    model_type : str
        Type of model to create
    config : ExperimentConfig
        Configuration to use for model creation

    Returns:
    --------
    torch.nn.Module
        Created model instance
    """
    try:
        # Try to use model registry first
        if model_type in ModelRegistry._models or model_type in ModelRegistry._creators:
            return ModelRegistry.create_model(model_type, config=config)

        # Fall back to direct import for backward compatibility
        if model_type == "stgnn":
            from gnn_package.src.models.stgnn import create_stgnn_model

            return create_stgnn_model(config=config)

        # Add more model types as needed

        raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        logger.error(f"Error creating model of type {model_type}: {str(e)}")
        raise RuntimeError(f"Failed to create model of type {model_type}") from e
