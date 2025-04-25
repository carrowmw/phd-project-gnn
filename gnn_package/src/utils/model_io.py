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
from gnn_package.src.utils.device_utils import get_device, get_device_from_config

logger = logging.getLogger(__name__)


def save_model(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    save_config: bool = True,
    config: Optional[ExperimentConfig] = None,
) -> Dict[str, str]:
    """
    Save a model with metadata and configuration.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to save
    output_path : str or Path
        Directory or file path for saving the model
    metadata : Dict[str, Any], optional
        Additional metadata to save with the model
    save_config : bool
        Whether to save the configuration alongside the model
    config : ExperimentConfig, optional
        Configuration to save. If None, uses the global config.

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
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")

    # Save metadata if provided
    if metadata:
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        saved_files["metadata"] = str(metadata_path)
        logger.info(f"Metadata saved to: {metadata_path}")

    # Save configuration if requested
    if save_config:
        if config is None:
            config = get_config()

        config_path = output_dir / "config.yml"
        config.save(config_path)
        saved_files["config"] = str(config_path)
        logger.info(f"Configuration saved to: {config_path}")

    return saved_files


def load_model(
    model_path: Union[str, Path],
    model_type: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    model_creator: Optional[Callable] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a model with standard error handling and device management.

    Parameters:
    -----------
    model_path : str or Path
        Path to the model file (.pth)
    model_type : str, optional
        Type of model to load (used with ModelRegistry)
    config : ExperimentConfig, optional
        Configuration for model creation. If None, tries to load config from model directory.
    model_creator : Callable, optional
        Function to create the model. If provided, overrides model_type.
    device : str or torch.device, optional
        Device to load the model onto. If None, determined from config.
    strict : bool
        Whether to strictly enforce that the keys in state_dict match model

    Returns:
    --------
    Tuple[torch.nn.Module, Dict[str, Any]]
        The loaded model and metadata dictionary
    """
    model_path = Path(model_path)

    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Determine model directory and try to load config if not provided
    model_dir = model_path.parent
    metadata = {}

    # Try to load metadata if it exists
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            logger.info(f"Loaded metadata from: {metadata_path}")

    # If model_type not specified, try to get from metadata
    if model_type is None and "model_type" in metadata:
        model_type = metadata["model_type"]
        logger.info(f"Using model type from metadata: {model_type}")

    # Try to load config if not provided
    if config is None:
        config_path = model_dir / "config.yml"
        if config_path.exists():
            config = ExperimentConfig(config_path)
            logger.info(f"Loaded configuration from: {config_path}")
        else:
            config = get_config()
            logger.info("Using global configuration")

    # Create model
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

    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Model weights loaded from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model weights: {str(e)}")
        raise RuntimeError(f"Failed to load model weights: {str(e)}") from e

    # Move model to device
    if device is None:
        device = get_device_from_config(config)
    else:
        device = get_device(device) if isinstance(device, str) else device

    model = model.to(device)
    model.eval()  # Set to evaluation mode by default

    return model, metadata
