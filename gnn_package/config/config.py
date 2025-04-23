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
class GeneralDataConfig:
    """Configuration for data processing shared across training and prediction."""

    # Time series-related parameters
    window_size: int = 24
    horizon: int = 6
    stride: int = 1
    gap_threshold_minutes: int = 15
    missing_value: float = -1.0
    resampling_frequency: str = "15min"
    standardize: bool = True
    batch_size: int = 32
    buffer_factor: float = 1.0

    # Graph-related parameters
    graph_prefix: str = "25022025_test"  # TO DEPRECATE
    sigma_squared: float = 0.1
    epsilon: float = 0.5
    normalization_factor: int = 10000
    max_distance: float = 100.0  # For connected components
    tolerance_decimal_places: int = 6  # For coordinate comparison
    resampling_frequency: str = "15min"
    missing_value: float = -999.0
    sensor_id_prefix: str = "1"  # Added for sensor ID formatting
    bbox_coords: List[float] = field(
        default_factory=lambda: [
            [-1.65327, 54.93188],
            [-1.54993, 54.93188],
            [-1.54993, 55.02084],
            [-1.65327, 55.02084],
        ]
    )
    place_name: str = "Newcastle upon Tyne, UK"  # For osmnx graph creation
    bbox_crs: str = "EPSG:4326"
    road_network_crs: str = "EPSG:27700"
    network_type: str = "walk"
    custom_filter: str = """["highway"~"footway|path|pedestrian|steps|corridor|'
        'track|service|living_street|residential|unclassified"]'
        '["area"!~"yes"]["access"!~"private"]"""

    @property
    def gap_threshold(self) -> pd.Timedelta:
        """Get the gap threshold as a pandas Timedelta."""
        return pd.Timedelta(minutes=self.gap_threshold_minutes)


@dataclass
class TrainingDataConfig:
    """Configuration specific to training data processing."""

    start_date: str = "2024-02-18 00:00:00"
    end_date: str = "2024-02-25 00:00:00"
    n_splits: int = 3  # For rolling window splits
    use_cross_validation: bool = True
    split_method: str = "rolling_window"  # Options: "rolling_window", "time_based"
    train_ratio: float = 0.8  # For rolling window splits
    cutoff_date: str = None  # Instead of train_ratio for time-based splits
    cv_split_index: int = -1  # For cross-validation


@dataclass
class PredictionDataConfig:
    """Configuration specific to prediction/testing."""

    days_back: int  # How much historical data to use


@dataclass
class DataConfig:
    """Complete data configuration."""

    general: GeneralDataConfig
    training: TrainingDataConfig
    prediction: PredictionDataConfig


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    input_dim: int = 1
    hidden_dim: int = 64
    output_dim: int = 1
    num_layers: int = 2
    dropout: float = 0.2
    num_gc_layers: int = 2
    use_self_loops: bool = True
    gcn_normalization: str = "symmetric"  # "symmetric", "random_walk", or "none"
    decoder_layers: int = 2
    attention_heads: int = 1
    layer_norm: bool = False


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_epochs: int = 50
    patience: int = 10
    device: Optional[str] = None


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

    def __init__(
        self, config_path: Optional[str] = None, is_prediction_mode: bool = False
    ):
        """
        Initialize configuration from a YAML file.

        Parameters:
        -----------
        config_path : str, optional
            Path to the YAML configuration file
        is_prediction_mode : bool
            Whether this configuration is for prediction (vs training)
        """
        self._initializing = True
        self.is_prediction_mode = is_prediction_mode

        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config.yml")

        self.config_path = Path(config_path)
        self._load_config()

        # Mark initialization as complete and freeze the config
        self._initialzing = False
        self._frozen = True

        # Validate the configuration
        self.validate()

        # Log the configuration
        self.log()

    def __setattr__(self, name, value):
        if hasattr(self, "_frozen") and self._frozen:
            raise AttributeError(
                f"Cannot modify configuration after initialization: {name}"
            )
        super().__setattr__(name, value)

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Initialize sub-configs with proper handling of nested structures
        self.experiment = ExperimentMetadata(**config_dict.get("experiment", {}))

        # Properly handle nested data configuration
        data_dict = config_dict.get("data", {})
        self.data = DataConfig(
            general=GeneralDataConfig(**data_dict.get("general", {})),
            training=TrainingDataConfig(**data_dict.get("training", {})),
            prediction=PredictionDataConfig(**data_dict.get("prediction", {})),
        )

        self.model = ModelConfig(**config_dict.get("model", {}))
        self.training = TrainingConfig(**config_dict.get("training", {}))
        self.paths = PathsConfig(**config_dict.get("paths", {}))
        self.visualization = VisualizationConfig(**config_dict.get("visualization", {}))

        # Store the raw dict for any additional access
        self._config_dict = config_dict

    def validate(self) -> bool:
        """
        Validate that all required configuration values are present and valid.

        Returns:
        --------
        bool
            True if validation passes (raises exceptions otherwise)
        """
        # Check for required model parameters
        required_model_params = [
            "input_dim",
            "hidden_dim",
            "output_dim",
            "num_layers",
            "dropout",
            "num_gc_layers",
            "use_self_loops",
            "gcn_normalization",
            "decoder_layers",
        ]

        for param in required_model_params:
            if not hasattr(self.model, param):
                raise ValueError(f"Missing required config value: model.{param}")

        # Check for required data parameters
        required_data_params = [
            "window_size",
            "horizon",
            "missing_value",
        ]

        for param in required_data_params:
            if not hasattr(self.data.general, param):
                # Add detailed debugging info
                print(f"DEBUG: Validation failed for {param}")
                print(
                    f"DEBUG: self.data.general has attributes: {dir(self.data.general)}"
                )
                print(
                    f"DEBUG: Raw config contains: {self._config_dict.get('data', {}).get('general', {})}"
                )
                raise ValueError(f"Missing required config value: data.general.{param}")

        # Check for required training parameters
        required_training_params = [
            "learning_rate",
            "weight_decay",
            "num_epochs",
            "patience",
            # Add any other required training parameters
        ]

        for param in required_training_params:
            if not hasattr(self.training, param):
                raise ValueError(f"Missing required config value: training.{param}")

        # Type checking and value validation
        if self.data.general.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.data.general.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.model.num_gc_layers <= 0:
            raise ValueError("num_gc_layers must be positive")

        return True

    def log(self, logger=None):
        """
        Log the configuration details.

        Parameters:
        -----------
        logger : logging.Logger, optional
            Logger to use. If None, creates or gets a default logger.
        """
        if logger is None:
            import logging

            logger = logging.getLogger(__name__)

        logger.info("Configuration loaded from: %s", self.config_path)

        logger.info(f"Experiment: {self.experiment.name} (v{self.experiment.version})")
        logger.info(f"Description: {self.experiment.description}")
        logger.info(
            f"Data config: window_size={self.data.general.window_size}, horizon={self.data.general.horizon}"
        )
        logger.info(
            f"Model config: hidden_dim={self.model.hidden_dim}, layers={self.model.num_layers}"
        )
        logger.info(
            f"Training config: epochs={self.training.num_epochs}, lr={self.training.learning_rate}"
        )

        # Log paths
        logger.info(f"Model save path: {self.paths.model_save_path}")
        logger.info(f"Results directory: {self.paths.results_dir}")

    def save(self, path: Optional[str] = None):
        """
        Save the current configuration to a YAML file.
        """
        save_path = Path(path) if path else self.config_path

        # Create nested dictionary from dataclasses
        config_dict = {
            "experiment": self._dataclass_to_dict(self.experiment),
            "model": self._dataclass_to_dict(self.model),
            "training": self._dataclass_to_dict(self.training),
            "paths": self._dataclass_to_dict(self.paths),
            "visualization": self._dataclass_to_dict(self.visualization),
        }

        # Handle the nested data section specially
        data_dict = {}
        data = self.data
        if hasattr(data, "general"):
            data_dict["general"] = self._dataclass_to_dict(data.general)
        if hasattr(data, "training"):
            data_dict["training"] = self._dataclass_to_dict(data.training)
        if hasattr(data, "prediction"):
            data_dict["prediction"] = self._dataclass_to_dict(data.prediction)

        config_dict["data"] = data_dict

        # Ensure the directory exists
        os.makedirs(save_path.parent, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @staticmethod
    def _dataclass_to_dict(obj):
        """Convert a dataclass instance to a dictionary, including nested dataclasses."""
        if obj is None:
            return None

        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)

            # Handle Path objects
            if isinstance(value, Path):
                value = str(value)
            # Handle nested dataclasses
            elif hasattr(value, "__dataclass_fields__"):
                value = ExperimentConfig._dataclass_to_dict(value)
            # Handle lists that might contain dataclasses
            elif isinstance(value, list):
                value = [
                    (
                        ExperimentConfig._dataclass_to_dict(item)
                        if hasattr(item, "__dataclass_fields__")
                        else item
                    )
                    for item in value
                ]

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
            f"  data: window_size={self.data.general.window_size}, horizon={self.data.general.horizon}\n"
            f"  model: hidden_dim={self.model.hidden_dim}, layers={self.model.num_layers}\n"
            f"  training: epochs={self.training.num_epochs}, lr={self.training.learning_rate}\n"
            f")"
        )
