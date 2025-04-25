"""
Centralized configuration system for the GNN package.

This module defines the configuration classes and validation logic
for all package components.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
import tempfile
from typing import Dict, List, Any, Optional, Union
import yaml
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """
    Metadata for the experiment.

    Attributes:
        name (str): Name of the experiment
        description (str): Description of the experiment
        version (str): Version of the experiment
        tags (List[str]): Tags for categorizing the experiment
    """

    name: str
    description: str
    version: str
    tags: List[str] = field(default_factory=list)


@dataclass
class GeneralDataConfig:
    """
    Configuration for data processing shared across training and prediction.

    Attributes:
        window_size (int): Size of the sliding window for time series
        horizon (int): Number of future time steps to predict
        stride (int): Number of steps to move the window
        batch_size (int): Batch size for data loaders
        gap_threshold_minutes (int): Maximum time gap to consider continuous
        missing_value (float): Value to use for missing data points
        resampling_frequency (str): Frequency string for resampling (e.g., "15min")
        standardize (bool): Whether to standardize the data
        buffer_factor (float): Buffer factor for data loading

        # Graph-related parameters
        graph_prefix (str): Prefix for graph data files
        sigma_squared (float): Parameter for Gaussian kernel
        epsilon (float): Threshold for edge weights
        normalization_factor (int): Factor for normalizing distances
        max_distance (float): Maximum distance for connected components
        tolerance_decimal_places (int): Rounding tolerance for coordinates
        sensor_id_prefix (str): Prefix for sensor IDs
        bbox_coords (List[float]): Bounding box coordinates
        place_name (str): Name of the place for network extraction
        bbox_crs (str): CRS for bounding box
        road_network_crs (str): CRS for road network
        network_type (str): Type of network to extract
        custom_filter (str): Filter for network extraction
    """

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
    graph_prefix: str = "default_graph"
    sigma_squared: float = 0.1
    epsilon: float = 0.5
    normalization_factor: int = 10000
    max_distance: float = 100.0
    tolerance_decimal_places: int = 6
    sensor_id_prefix: str = "1"
    bbox_coords: List[List[float]] = field(
        default_factory=lambda: [
            [-1.65327, 54.93188],
            [-1.54993, 54.93188],
            [-1.54993, 55.02084],
            [-1.65327, 55.02084],
        ]
    )
    place_name: str = "Newcastle upon Tyne, UK"
    bbox_crs: str = "EPSG:4326"
    road_network_crs: str = "EPSG:27700"
    network_type: str = "walk"
    custom_filter: str = (
        """["highway"~"footway|path|pedestrian|steps|corridor|track|service|living_street|residential|unclassified"]["area"!~"yes"]["access"!~"private"]"""
    )

    @property
    def gap_threshold(self) -> pd.Timedelta:
        """Get the gap threshold as a pandas Timedelta."""
        return pd.Timedelta(minutes=self.gap_threshold_minutes)

    def validate(self) -> List[str]:
        """
        Validate general data configuration parameters.

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        # Validate numeric parameters
        if self.window_size <= 0:
            errors.append("window_size must be positive")
        if self.horizon <= 0:
            errors.append("horizon must be positive")
        if self.stride <= 0:
            errors.append("stride must be positive")
        if self.gap_threshold_minutes < 0:
            errors.append("gap_threshold_minutes must be non-negative")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.buffer_factor <= 0:
            errors.append("buffer_factor must be positive")

        # Validate graph parameters
        if self.sigma_squared <= 0:
            errors.append("sigma_squared must be positive")
        if self.epsilon < 0 or self.epsilon > 1:
            errors.append("epsilon must be between 0 and 1")
        if self.normalization_factor <= 0:
            errors.append("normalization_factor must be positive")
        if self.max_distance <= 0:
            errors.append("max_distance must be positive")

        # Validate string parameters
        if not self.resampling_frequency:
            errors.append("resampling_frequency must not be empty")
        if not self.sensor_id_prefix:
            errors.append("sensor_id_prefix must not be empty")
        if not self.place_name:
            errors.append("place_name must not be empty")

        return errors


@dataclass
class TrainingDataConfig:
    """
    Configuration specific to training data processing.

    Attributes:
        start_date (str): Start date for training data
        end_date (str): End date for training data
        n_splits (int): Number of splits for cross-validation
        use_cross_validation (bool): Whether to use cross-validation
        split_method (str): Method for splitting data
        train_ratio (float): Ratio of data to use for training
        cutoff_date (str): Specific date to cut off training data
        cv_split_index (int): Index of the split to use for cross-validation
    """

    start_date: str = "2024-02-18 00:00:00"
    end_date: str = "2024-02-25 00:00:00"
    n_splits: int = 3
    use_cross_validation: bool = True
    split_method: str = "rolling_window"  # Options: "rolling_window", "time_based"
    train_ratio: float = 0.8
    cutoff_date: Optional[str] = None
    cv_split_index: int = -1

    def validate(self) -> List[str]:
        """
        Validate training data configuration parameters.

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        # Validate dates
        try:
            if self.start_date:
                pd.to_datetime(self.start_date)
        except ValueError:
            errors.append(f"Invalid start_date format: {self.start_date}")

        try:
            if self.end_date:
                pd.to_datetime(self.end_date)
        except ValueError:
            errors.append(f"Invalid end_date format: {self.end_date}")

        if self.cutoff_date:
            try:
                pd.to_datetime(self.cutoff_date)
            except ValueError:
                errors.append(f"Invalid cutoff_date format: {self.cutoff_date}")

        # Validate numeric parameters
        if self.n_splits <= 0:
            errors.append("n_splits must be positive")
        if self.train_ratio <= 0 or self.train_ratio >= 1:
            errors.append("train_ratio must be between 0 and 1")

        # Validate categorical parameters
        valid_split_methods = ["rolling_window", "time_based"]
        if self.split_method not in valid_split_methods:
            errors.append(f"split_method must be one of {valid_split_methods}")

        return errors


@dataclass
class PredictionDataConfig:
    """
    Configuration specific to prediction/testing.

    Attributes:
        days_back (int): How many days of historical data to use for prediction
    """

    days_back: int = 14

    def validate(self) -> List[str]:
        """
        Validate prediction data configuration parameters.

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        if self.days_back <= 0:
            errors.append("days_back must be positive")

        return errors


@dataclass
class DataConfig:
    """
    Complete data configuration.

    Attributes:
        general (GeneralDataConfig): General data configuration
        training (TrainingDataConfig): Training-specific configuration
        prediction (PredictionDataConfig): Prediction-specific configuration
    """

    general: GeneralDataConfig = field(default_factory=GeneralDataConfig)
    training: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    prediction: PredictionDataConfig = field(default_factory=PredictionDataConfig)

    def validate(self) -> List[str]:
        """
        Validate all data configuration parameters.

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        # Validate each sub-config
        errors.extend(self.general.validate())
        errors.extend(self.training.validate())
        errors.extend(self.prediction.validate())

        # Validate relationships between configs
        if pd.to_datetime(self.training.start_date) >= pd.to_datetime(
            self.training.end_date
        ):
            errors.append("training.start_date must be before training.end_date")

        return errors


@dataclass
class ModelConfig:
    """
    Configuration for the model architecture.

    Attributes:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of output features
        num_layers (int): Number of layers in the model
        dropout (float): Dropout probability
        num_gc_layers (int): Number of graph convolution layers
        use_self_loops (bool): Whether to use self-loops in graph convolution
        gcn_normalization (str): Normalization method for graph convolution
        decoder_layers (int): Number of decoder layers
        attention_heads (int): Number of attention heads
        layer_norm (bool): Whether to use layer normalization
    """

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

    def validate(self) -> List[str]:
        """
        Validate model configuration parameters.

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        # Validate numeric parameters
        if self.input_dim <= 0:
            errors.append("input_dim must be positive")
        if self.hidden_dim <= 0:
            errors.append("hidden_dim must be positive")
        if self.output_dim <= 0:
            errors.append("output_dim must be positive")
        if self.num_layers <= 0:
            errors.append("num_layers must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            errors.append("dropout must be between 0 and 1")
        if self.num_gc_layers <= 0:
            errors.append("num_gc_layers must be positive")
        if self.decoder_layers <= 0:
            errors.append("decoder_layers must be positive")
        if self.attention_heads <= 0:
            errors.append("attention_heads must be positive")

        # Validate categorical parameters
        valid_normalizations = ["symmetric", "random_walk", "none"]
        if self.gcn_normalization not in valid_normalizations:
            errors.append(f"gcn_normalization must be one of {valid_normalizations}")

        return errors


@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Attributes:
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay for regularization
        num_epochs (int): Number of training epochs
        patience (int): Patience for early stopping
        device (str): Device to use for training (e.g., "cpu", "cuda", "mps")
    """

    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_epochs: int = 50
    patience: int = 10
    device: Optional[str] = None

    def validate(self) -> List[str]:
        """
        Validate training configuration parameters.

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        # Validate numeric parameters
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.weight_decay < 0:
            errors.append("weight_decay must be non-negative")
        if self.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        if self.patience <= 0:
            errors.append("patience must be positive")

        # Validate device if provided
        if self.device is not None and self.device not in ["cpu", "cuda", "mps"]:
            errors.append("device must be one of 'cpu', 'cuda', 'mps', or None")

        return errors


@dataclass
class PathsConfig:
    """
    Configuration for file paths.

    Attributes:
        model_save_path (str): Path to save trained models
        data_cache (str): Path to cache data
        results_dir (str): Path to save results
    """

    model_save_path: str = "models"
    data_cache: str = "data/cache"
    results_dir: str = "results"

    def __post_init__(self):
        """Convert string paths to Path objects and ensure directories exist."""
        self.model_save_path = Path(self.model_save_path)
        self.data_cache = Path(self.data_cache)
        self.results_dir = Path(self.results_dir)

    def validate(self) -> List[str]:
        """
        Validate paths configuration.

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        # All paths should be valid
        if not isinstance(self.model_save_path, (str, Path)):
            errors.append("model_save_path must be a string or Path")
        if not isinstance(self.data_cache, (str, Path)):
            errors.append("data_cache must be a string or Path")
        if not isinstance(self.results_dir, (str, Path)):
            errors.append("results_dir must be a string or Path")

        return errors

    def ensure_dirs_exist(self):
        """Ensure all directories exist, creating them if necessary."""
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.data_cache, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization components.

    Attributes:
        dashboard_template (str): Template for dashboard HTML
        default_sensors_to_plot (int): Number of sensors to plot by default
        max_sensors_in_heatmap (int): Maximum number of sensors in heatmap
    """

    dashboard_template: str = "dashboard.html"
    default_sensors_to_plot: int = 6
    max_sensors_in_heatmap: int = 50

    def validate(self) -> List[str]:
        """
        Validate visualization configuration parameters.

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        # Validate numeric parameters
        if self.default_sensors_to_plot <= 0:
            errors.append("default_sensors_to_plot must be positive")
        if self.max_sensors_in_heatmap <= 0:
            errors.append("max_sensors_in_heatmap must be positive")

        # Validate string parameters
        if not self.dashboard_template:
            errors.append("dashboard_template must not be empty")

        return errors


class ExperimentConfig:
    """
    Main configuration class for experiments.

    This class manages all configuration parameters for the GNN package
    and provides methods for loading, validating, and accessing configuration.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        is_prediction_mode: bool = False,
        override_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration from a YAML file with optional overrides.

        Parameters:
        -----------
        config_path : str, optional
            Path to the YAML configuration file. If None, uses 'config.yml' in current directory.
        is_prediction_mode : bool
            Whether this configuration is for prediction (vs training).
        override_params : Dict[str, Any], optional
            Dictionary of parameters to override in the loaded configuration.
        """
        self._initializing = True
        self.is_prediction_mode = is_prediction_mode

        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config.yml")

        self.config_path = Path(config_path)
        self._load_config()

        # Apply any overrides
        if override_params:
            self._apply_overrides(override_params)

        # Mark initialization as complete and freeze the config
        self._initializing = False
        self._frozen = True

        # Validate the configuration
        self.validate()

        # Log the configuration
        self.log()

    def __setattr__(self, name, value):
        """Control attribute setting to enforce immutability after initialization."""
        if hasattr(self, "_frozen") and self._frozen and not name.startswith("_"):
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
        general_dict = data_dict.get("general", {})
        training_dict = data_dict.get("training", {})
        prediction_dict = data_dict.get("prediction", {})

        # Create data configuration
        self.data = DataConfig(
            general=GeneralDataConfig(**general_dict),
            training=TrainingDataConfig(**training_dict),
            prediction=PredictionDataConfig(**prediction_dict),
        )

        self.model = ModelConfig(**config_dict.get("model", {}))
        self.training = TrainingConfig(**config_dict.get("training", {}))
        self.paths = PathsConfig(**config_dict.get("paths", {}))
        self.visualization = VisualizationConfig(**config_dict.get("visualization", {}))

        # Store the raw dict for any additional access
        self._config_dict = config_dict

    def _apply_overrides(self, override_params: Dict[str, Any]):
        """
        Apply parameter overrides to the configuration.

        Parameters:
        -----------
        override_params : Dict[str, Any]
            Dictionary of parameters to override in the loaded configuration.
            Keys should be dot-separated paths (e.g., "model.hidden_dim").
        """
        for key, value in override_params.items():
            parts = key.split(".")

            if len(parts) == 2:
                # Handle two-level paths (e.g., "model.hidden_dim")
                section, attribute = parts
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, attribute):
                        setattr(section_obj, attribute, value)
                    else:
                        logger.warning(f"Unknown attribute in override: {key}")
                else:
                    logger.warning(f"Unknown section in override: {key}")
            elif len(parts) == 3:
                # Handle three-level paths (e.g., "data.general.window_size")
                section, subsection, attribute = parts
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, subsection):
                        subsection_obj = getattr(section_obj, subsection)
                        if hasattr(subsection_obj, attribute):
                            setattr(subsection_obj, attribute, value)
                        else:
                            logger.warning(f"Unknown attribute in override: {key}")
                    else:
                        logger.warning(f"Unknown subsection in override: {key}")
                else:
                    logger.warning(f"Unknown section in override: {key}")
            else:
                logger.warning(f"Invalid override key format: {key}")

    def validate(self, raise_exceptions: bool = True) -> Union[bool, List[str]]:
        """
        Validate that all required configuration values are present and valid.

        Parameters:
        -----------
        raise_exceptions : bool
            If True, raises ValueError with all validation errors.
            If False, returns a list of validation errors.

        Returns:
        --------
        Union[bool, List[str]]
            If raise_exceptions is True, returns True if validation passes.
            If raise_exceptions is False, returns a list of validation errors.
        """
        errors = []

        # Collect errors from all configuration sections
        errors.extend(self.data.validate())
        errors.extend(self.model.validate())
        errors.extend(self.training.validate())
        errors.extend(self.paths.validate())
        errors.extend(self.visualization.validate())

        # If any errors were found and raise_exceptions is True, raise ValueError
        if errors and raise_exceptions:
            raise ValueError("\n".join(errors))

        # Return either True or the list of errors
        return True if not errors else errors

    def log(self, logger=None):
        """
        Log the configuration details.

        Parameters:
        -----------
        logger : logging.Logger, optional
            Logger to use. If None, creates or gets a default logger.
        """
        if logger is None:
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

        Parameters:
        -----------
        path : str, optional
            Path where to save the configuration file. If None, uses the original config path.
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
        config_dict["data"] = {
            "general": self._dataclass_to_dict(self.data.general),
            "training": self._dataclass_to_dict(self.data.training),
            "prediction": self._dataclass_to_dict(self.data.prediction),
        }

        # Ensure the directory exists
        os.makedirs(save_path.parent, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @staticmethod
    def _dataclass_to_dict(obj):
        """Convert a dataclass instance to a dictionary, including nested dataclasses."""
        if obj is None:
            return None

        if hasattr(obj, "__dataclass_fields__"):
            # If it's a dataclass, convert to dict
            return {
                field_name: ExperimentConfig._dataclass_to_dict(
                    getattr(obj, field_name)
                )
                for field_name in obj.__dataclass_fields__
            }
        elif isinstance(obj, Path):
            # Convert Path objects to strings
            return str(obj)
        elif isinstance(obj, list):
            # Handle lists that might contain dataclasses
            return [ExperimentConfig._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            # Handle dictionaries that might contain dataclasses
            return {k: ExperimentConfig._dataclass_to_dict(v) for k, v in obj.items()}
        else:
            # Return other types unchanged
            return obj

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
        current = self

        try:
            for k in keys:
                if hasattr(current, k):
                    current = getattr(current, k)
                else:
                    return default
            return current
        except Exception:
            return default

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the entire configuration to a dictionary.

        Returns:
        --------
        Dict[str, Any]
            Dictionary representation of the configuration
        """
        return {
            "experiment": self._dataclass_to_dict(self.experiment),
            "data": {
                "general": self._dataclass_to_dict(self.data.general),
                "training": self._dataclass_to_dict(self.data.training),
                "prediction": self._dataclass_to_dict(self.data.prediction),
            },
            "model": self._dataclass_to_dict(self.model),
            "training": self._dataclass_to_dict(self.training),
            "paths": self._dataclass_to_dict(self.paths),
            "visualization": self._dataclass_to_dict(self.visualization),
        }

    def create_prediction_config(self) -> "ExperimentConfig":
        """
        Create a prediction-specific configuration based on this configuration.

        Returns:
        --------
        ExperimentConfig
            A new configuration instance optimized for prediction
        """
        # Save current config to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp:
            self.save(temp.name)
            temp_path = temp.name

        try:
            # Create a new config with prediction mode enabled
            prediction_config = ExperimentConfig(
                config_path=temp_path,
                is_prediction_mode=True,
                override_params={
                    # Add any prediction-specific overrides here
                    "data.training.use_cross_validation": False,
                },
            )
            return prediction_config
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

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
