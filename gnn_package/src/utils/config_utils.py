import copy
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import torch

from gnn_package.src.models.stgnn import create_stgnn_model
from gnn_package.config import ExperimentConfig, get_config


def create_prediction_config_from_training(
    training_config: ExperimentConfig,
    override_params: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """
    Create a prediction-focused configuration from a training configuration.
    Preserves model architecture and general data parameters but applies prediction-specific settings.

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
    # Create a deep copy of the configuration dictionary
    config_dict = copy.deepcopy(training_config._config_dict)

    # Update training settings to be more suitable for prediction
    if "data" in config_dict and "training" in config_dict["data"]:
        config_dict["data"]["training"]["use_cross_validation"] = False
        config_dict["data"]["training"]["cv_split_index"] = 0

    # Apply any override parameters
    if override_params:
        for key, value in override_params.items():
            parts = key.split(".")
            if (
                len(parts) == 3 and parts[0] == "data"
            ):  # e.g. "data.prediction.days_back"
                _, section, param = parts
                config_dict["data"][section][param] = value
            elif len(parts) == 2:  # e.g. "model.dropout"
                section, param = parts
                config_dict[section][param] = value

    # Create a temporary config file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp:
        yaml.dump(config_dict, temp, default_flow_style=False)
        temp_path = temp.name

    # Load the new configuration
    new_config = ExperimentConfig(temp_path)

    # Clean up temporary file
    Path(temp_path).unlink()

    return new_config


def save_model_with_config(model, config, path):
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


def load_model_for_prediction(
    model_path, config=None, override_params=None, model_creator_func=None
):
    """
    Load a model with the appropriate configuration for prediction.

    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model file
    config : ExperimentConfig, optional
        Configuration object. If None, attempts to find config in model directory
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
            config = ExperimentConfig(potential_config_path)
        else:
            # Fall back to global config
            config = get_config()

    # Convert training config to prediction config if needed
    config = create_prediction_config_from_training(config, override_params)

    # Create model with correct architecture
    if model_creator_func is None:
        model_creator_func = create_stgnn_model

    model = model_creator_func(config)

    # Load saved weights
    model.load_state_dict(torch.load(model_path))

    return model, config


def create_prediction_config(
    training_config_path: Optional[Path] = None,
) -> ExperimentConfig:
    """
    Create a prediction-specific configuration.

    Parameters:
    -----------
    training_config_path : Path, optional
        Path to training configuration. If None, uses default config.

    Returns:
    --------
    ExperimentConfig
        Configuration optimized for prediction
    """
    from gnn_package.config import ExperimentConfig, get_config

    if training_config_path is None:
        # Use default config
        config = get_config()
    else:
        # Load from provided path
        config = ExperimentConfig(str(training_config_path))

    # Create new config with prediction flag set
    prediction_config = ExperimentConfig(
        config_path=str(config.config_path), is_prediction_mode=True
    )

    return prediction_config
