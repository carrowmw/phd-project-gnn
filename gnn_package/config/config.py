# gnn_package/src/utils/config.py

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import yaml
from datetime import timedelta
import pandas as pd


@dataclass
class ExperimentMetadata:
    """Metadata for the experiment."""

    name: str
    description: str
    version: str
    tags: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Configuration for data processing."""

    start_date: str
    end_date: str
    graph_prefix: str
    window_size: int
    horizon: int
    batch_size: int
    days_back: int = 14
    stride: int = 1
    gap_threshold_minutes: int = 15

    @property
    def gap_threshold(self) -> pd.Timedelta:
        """Get the gap threshold as a pandas Timedelta."""
        return pd.Timedelta(minutes=self.gap_threshold_minutes)


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float
    weight_decay: float
    num_epochs: int
    patience: int
    train_val_split: float = 0.8


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    model_save_path: str
    data_cache: str
    results_dir: str

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.model_save_path = Path(self.model_save_path)
        self.data_cache = Path(self.data_cache)
        self.results_dir = Path(self.results_dir)


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""

    dashboard_template: str
    default_sensors_to_plot: int
    max_sensors_in_heatmap: int


class ExperimentConfig:
    """Main configuration class for experiments."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from a YAML file.

        Parameters:
        -----------
        config_path : str, optional
            Path to the YAML configuration file. If not provided,
            looks for 'config.yml' in the current directory.
        """
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config.yml")

        self.config_path = Path(config_path)
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Initialize sub-configs
        self.experiment = ExperimentMetadata(**config_dict.get("experiment", {}))
        self.data = DataConfig(**config_dict.get("data", {}))
        self.model = ModelConfig(**config_dict.get("model", {}))
        self.training = TrainingConfig(**config_dict.get("training", {}))
        self.paths = PathsConfig(**config_dict.get("paths", {}))
        self.visualization = VisualizationConfig(**config_dict.get("visualization", {}))

        # Store the raw dict for any additional access
        self._config_dict = config_dict

    def save(self, path: Optional[str] = None):
        """
        Save the current configuration to a YAML file.

        Parameters:
        -----------
        path : str, optional
            Path to save the configuration. If not provided, uses the path
            from which the configuration was loaded.
        """
        save_path = Path(path) if path else self.config_path

        # Create nested dictionary from dataclasses
        config_dict = {
            "experiment": self._dataclass_to_dict(self.experiment),
            "data": self._dataclass_to_dict(self.data),
            "model": self._dataclass_to_dict(self.model),
            "training": self._dataclass_to_dict(self.training),
            "paths": self._dataclass_to_dict(self.paths),
            "visualization": self._dataclass_to_dict(self.visualization),
        }

        # Ensure the directory exists
        os.makedirs(save_path.parent, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @staticmethod
    def _dataclass_to_dict(obj):
        """Convert a dataclass instance to a dictionary."""
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            # Handle Path objects
            if isinstance(value, Path):
                value = str(value)
            result[field_name] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by its key path.

        Parameters:
        -----------
        key : str
            Dot-separated path to the configuration value (e.g., 'model.hidden_dim')
        default : Any
            Default value to return if the key is not found

        Returns:
        --------
        Any
            The configuration value or the default
        """
        keys = key.split(".")
        value = self._config_dict

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __str__(self):
        """String representation of the configuration."""
        return (
            f"ExperimentConfig(\n"
            f"  experiment: {self.experiment.name} (v{self.experiment.version})\n"
            f"  data: window_size={self.data.window_size}, horizon={self.data.horizon}\n"
            f"  model: hidden_dim={self.model.hidden_dim}, layers={self.model.num_layers}\n"
            f"  training: epochs={self.training.num_epochs}, lr={self.training.learning_rate}\n"
            f")"
        )
