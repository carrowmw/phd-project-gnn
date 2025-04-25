# src/utils/device_utils.py
import logging
from typing import Optional, Any
import torch
from gnn_package.config import ExperimentConfig

logger = logging.getLogger(__name__)


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Determine the appropriate device for computations.

    Parameters:
    -----------
    device_name : str, optional
        Name of the device to use ("cpu", "cuda", "mps").
        If None, will auto-detect the best available device.

    Returns:
    --------
    torch.device
        Device to use for computations

    Notes:
    ------
    If the requested device is not available, falls back to CPU.
    """
    if device_name is None:
        # Auto-detect best available device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {device_name}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        # Use specified device
        device_name = device_name.lower()

        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        elif device_name == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(device_name)
            logger.info(f"Using specified device: {device_name}")

    return device


def get_device_from_config(config: ExperimentConfig) -> torch.device:
    """
    Get the computation device based on configuration.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration object containing device specifications

    Returns:
    --------
    torch.device
        Device to use for computations
    """
    # Check if device is specified in config
    device_name = getattr(config.training, "device", None)
    return get_device(device_name)


def to_device(data: Any, device: torch.device) -> Any:
    """
    Move data to the specified device.

    Parameters:
    -----------
    data : Any
        Data to move to device (tensor, module, or collection of tensors)
    device : torch.device
        Target device

    Returns:
    --------
    Any
        Data on the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif hasattr(data, "to") and callable(data.to):
        # For nn.Module and other objects with to() method
        return data.to(device)
    else:
        # Return unchanged for other types
        return data
