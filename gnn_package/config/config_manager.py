# gnn_package/config/config_manager.py

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from .config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    PathsConfig,
    VisualizationConfig,
)

# Singleton pattern for global configuration
_CONFIG_INSTANCE = None


def get_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """
    Get or create the global configuration instance.

    Parameters:
    -----------
    config_path : str, optional
        Path to the configuration file. If not provided, will use the existing
        instance or look for a default config.yml in the current directory.

    Returns:
    --------
    ExperimentConfig
        The global configuration instance
    """
    global _CONFIG_INSTANCE

    # Return existing instance if available and no new path provided
    if _CONFIG_INSTANCE is not None and config_path is None:
        return _CONFIG_INSTANCE

    # Create new instance if path provided or no instance exists
    if config_path is not None or _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = ExperimentConfig(config_path)

    return _CONFIG_INSTANCE


def reset_config():
    """Reset the global configuration instance."""
    global _CONFIG_INSTANCE
    _CONFIG_INSTANCE = None


def create_default_config(output_path: str = "config.yml") -> ExperimentConfig:
    """
    Create a default configuration file and return its instance.

    Parameters:
    -----------
    output_path : str
        Path where to save the default configuration file

    Returns:
    --------
    ExperimentConfig
        The created configuration instance
    """
    # Default experiment metadata
    experiment = {
        "name": "Default Traffic Prediction Experiment",
        "description": "Traffic prediction using spatial-temporal GNN",
        "version": "1.0.0",
        "tags": ["traffic", "gnn", "prediction"],
    }

    # Default data configuration
    data = {
        "start_date": "2024-02-18 00:00:00",
        "end_date": "2024-02-25 00:00:00",
        "graph_prefix": "25022025_test",
        "window_size": 24,
        "horizon": 6,
        "batch_size": 32,
        "days_back": 14,
        "stride": 1,
        "gap_threshold_minutes": 15,
        "standardize": True,
        "sigma_squared": 0.1,
        "epsilon": 0.5,
        "normalization_factor": 10000,
        "max_distance": 100.0,  # For connected components
        "tolerance_decimal_places": 6,  # For coordinate comparison
        "resampling_frequency": "15min",
        "missing_value": -1.0,
        "sensor_id_prefix": "1",  # Added for sensor ID formatting
        "bbox_coords": [
            [-1.65327, 54.93188],
            [-1.54993, 54.93188],
            [-1.54993, 55.02084],
            [-1.65327, 55.02084],
        ],
        "bbox_crs": "EPSG:4326",
        "road_network_crs": "EPSG:27700",
        "network_type": "walk",
        "custom_filter": '["highway"~"footway|path|pedestrian|steps|corridor|'
        'track|service|living_street|residential|unclassified"]'
        '["area"!~"yes"]["access"!~"private"]',
    }

    # Default model configuration
    model = {
        "input_dim": 1,
        "hidden_dim": 64,
        "output_dim": 1,
        "num_layers": 2,
        "dropout": 0.2,
        "num_gc_layers": 2,
        "decode_layers": 2,
    }

    # Default training configuration
    training = {
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "num_epochs": 50,
        "patience": 10,
        "train_val_split": 0.8,
    }

    # Default paths
    paths = {
        "model_save_path": "models",
        "data_cache": "data/cache",
        "results_dir": "results",
    }

    # Default visualization configuration
    visualization = {
        "dashboard_template": "dashboard.html",
        "default_sensors_to_plot": 6,
        "max_sensors_in_heatmap": 50,
    }

    # Create the configuration dictionary
    config_dict = {
        "experiment": experiment,
        "data": data,
        "model": model,
        "training": training,
        "paths": paths,
        "visualization": visualization,
    }

    # Save to file
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Create and return instance
    config = ExperimentConfig(output_path)
    global _CONFIG_INSTANCE
    _CONFIG_INSTANCE = config

    return config


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file

    Returns:
    --------
    Dict[str, Any]
        The loaded configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_dict
