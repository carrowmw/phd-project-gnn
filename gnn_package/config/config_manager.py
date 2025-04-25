"""
Configuration management module for the GNN package.

This module provides functions and classes for managing configuration
instances, including loading, creating, and resetting the global configuration.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union

import yaml

from .config import ExperimentConfig

# Set up logging
logger = logging.getLogger(__name__)

# Singleton pattern for global configuration
_CONFIG_INSTANCE = None


class ConfigurationManager:
    """
    Central manager for all configuration operations.

    This class handles loading, updating, validating, and persisting
    configuration in a consistent way. It also manages the global
    configuration singleton.
    """

    @staticmethod
    def load_config(
        config_path: Optional[Union[str, Path]] = None,
        create_if_missing: bool = True,
        is_prediction_mode: bool = False,
        override_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> ExperimentConfig:
        """
        Load a configuration from a file with comprehensive error handling.

        Parameters:
        -----------
        config_path : str or Path, optional
            Path to the configuration file. If None, will look for config.yml in current directory.
        create_if_missing : bool
            Whether to create a default configuration if the specified file doesn't exist.
        is_prediction_mode : bool
            Whether this configuration is for prediction (vs training).
        override_params : Dict[str, Any], optional
            Dictionary of parameters to override in the loaded configuration.
        verbose : bool
            Whether to print information about the configuration being loaded.

        Returns:
        --------
        ExperimentConfig
            The loaded configuration instance.

        Raises:
        -------
        FileNotFoundError
            If the configuration file doesn't exist and create_if_missing is False.
        ValueError
            If the configuration fails validation.
        """
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config.yml")

        config_path = Path(config_path)

        # Check if the config file exists
        if not config_path.exists():
            if create_if_missing:
                if verbose:
                    logger.info(
                        f"Configuration file not found at {config_path}. Creating default."
                    )
                return ConfigurationManager.create_default_config(
                    config_path, verbose=verbose
                )
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            # Load the configuration
            if verbose:
                logger.info(f"Loading configuration from: {config_path}")

            config = ExperimentConfig(
                config_path=str(config_path),
                is_prediction_mode=is_prediction_mode,
                override_params=override_params,
            )

            if verbose:
                logger.info(f"Configuration loaded successfully from: {config_path}")

            return config

        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise

    @staticmethod
    def create_default_config(
        output_path: Union[str, Path], verbose: bool = False
    ) -> ExperimentConfig:
        """
        Create a default configuration file and return the configuration instance.

        Parameters:
        -----------
        output_path : str or Path
            Path where to save the default configuration file.
        verbose : bool
            Whether to print information about the configuration creation.

        Returns:
        --------
        ExperimentConfig
            The created configuration instance.

        Raises:
        -------
        OSError
            If the configuration file cannot be created due to file system issues.
        """
        output_path = Path(output_path)

        # Ensure the parent directory exists
        os.makedirs(output_path.parent, exist_ok=True)

        # Default experiment metadata
        experiment = {
            "name": "Default Traffic Prediction Experiment",
            "description": "Traffic prediction using spatial-temporal GNN",
            "version": "1.0.0",
            "tags": ["traffic", "gnn", "prediction"],
        }

        # Default data configuration
        data = {
            "general": {
                "window_size": 24,
                "horizon": 6,
                "stride": 1,
                "batch_size": 32,
                "gap_threshold_minutes": 15,
                "standardize": True,
                "missing_value": -999.0,
                "resampling_frequency": "15min",
                "buffer_factor": 1.0,
                "graph_prefix": "default_graph",
                "sensor_id_prefix": "1",
                "sigma_squared": 0.1,
                "epsilon": 0.5,
                "normalization_factor": 10000,
                "max_distance": 100.0,
                "tolerance_decimal_places": 6,
                # Network parameters
                "bbox_coords": [
                    [-1.65327, 54.93188],
                    [-1.54993, 54.93188],
                    [-1.54993, 55.02084],
                    [-1.65327, 55.02084],
                ],
                "place_name": "Newcastle upon Tyne, UK",
                "bbox_crs": "EPSG:4326",
                "road_network_crs": "EPSG:27700",
                "network_type": "walk",
                "custom_filter": '["highway"~"footway|path|pedestrian|steps|corridor|'
                'track|service|living_street|residential|unclassified"]'
                '["area"!~"yes"]["access"!~"private"]',
            },
            "training": {
                "start_date": "2024-02-18 00:00:00",
                "end_date": "2024-02-25 00:00:00",
                "n_splits": 3,
                "use_cross_validation": True,
                "split_method": "rolling_window",
                "train_ratio": 0.8,
                "cutoff_date": None,
                "cv_split_index": -1,
            },
            "prediction": {
                "days_back": 14,
            },
        }

        # Default model configuration
        model = {
            "input_dim": 1,
            "hidden_dim": 64,
            "output_dim": 1,
            "num_layers": 2,
            "dropout": 0.2,
            "num_gc_layers": 2,
            "decoder_layers": 2,
            "use_self_loops": True,
            "gcn_normalization": "symmetric",
            "attention_heads": 1,
            "layer_norm": False,
        }

        # Default training configuration
        training = {
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "num_epochs": 50,
            "patience": 10,
            "device": None,
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

        if verbose:
            logger.info(f"Default configuration created at: {output_path}")

        # Create and return instance
        return ExperimentConfig(output_path)

    @staticmethod
    def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a YAML configuration file as a dictionary.

        Parameters:
        -----------
        config_path : str or Path
            Path to the YAML configuration file.

        Returns:
        --------
        Dict[str, Any]
            The loaded configuration dictionary.

        Raises:
        -------
        FileNotFoundError
            If the configuration file doesn't exist.
        yaml.YAMLError
            If the YAML file is malformed.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            try:
                config_dict = yaml.safe_load(f)
                return config_dict
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {config_path}: {str(e)}")
                raise

    @staticmethod
    def save_config(
        config: ExperimentConfig, output_path: Union[str, Path], verbose: bool = False
    ) -> None:
        """
        Save a configuration to a YAML file.

        Parameters:
        -----------
        config : ExperimentConfig
            The configuration instance to save.
        output_path : str or Path
            Path where to save the configuration file.
        verbose : bool
            Whether to print information about the configuration saving.

        Raises:
        -------
        OSError
            If the configuration file cannot be created due to file system issues.
        """
        output_path = Path(output_path)

        # Ensure the parent directory exists
        os.makedirs(output_path.parent, exist_ok=True)

        # Save the configuration
        config.save(str(output_path))

        if verbose:
            logger.info(f"Configuration saved to: {output_path}")

    @staticmethod
    def create_prediction_config(
        base_config: Optional[ExperimentConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        override_params: Optional[Dict[str, Any]] = None,
    ) -> ExperimentConfig:
        """
        Create a configuration optimized for prediction.

        Parameters:
        -----------
        base_config : ExperimentConfig, optional
            Base configuration to use as a starting point. If None, will load from config_path.
        config_path : str or Path, optional
            Path to the configuration file to use as a base. Used only if base_config is None.
        override_params : Dict[str, Any], optional
            Dictionary of parameters to override in the loaded configuration.

        Returns:
        --------
        ExperimentConfig
            A new configuration instance optimized for prediction.
        """
        # Load base configuration if not provided
        if base_config is None:
            if config_path is not None:
                base_config = ConfigurationManager.load_config(
                    config_path=config_path, create_if_missing=True, verbose=False
                )
            else:
                # Use global config or create default
                base_config = get_config()

        # Combine prediction-specific overrides with any provided overrides
        prediction_overrides = {
            "data.training.use_cross_validation": False,
            "data.training.cv_split_index": 0,
        }

        if override_params:
            prediction_overrides.update(override_params)

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp:
            base_config.save(temp.name)
            temp_path = temp.name

        try:
            # Create prediction config
            prediction_config = ExperimentConfig(
                config_path=temp_path,
                is_prediction_mode=True,
                override_params=prediction_overrides,
            )

            return prediction_config
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def merge_configs(
        base_config: ExperimentConfig, override_config: ExperimentConfig
    ) -> ExperimentConfig:
        """
        Merge two configurations, with override_config taking precedence.

        Parameters:
        -----------
        base_config : ExperimentConfig
            Base configuration to use as a starting point.
        override_config : ExperimentConfig
            Configuration with values that should override the base.

        Returns:
        --------
        ExperimentConfig
            A new configuration instance with merged values.
        """
        # Convert both configs to dictionaries
        base_dict = base_config.as_dict()
        override_dict = override_config.as_dict()

        # Merge dictionaries (this is a recursive deep merge)
        merged_dict = ConfigurationManager._deep_merge_dicts(base_dict, override_dict)

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp:
            yaml.dump(merged_dict, temp, default_flow_style=False)
            temp_path = temp.name

        try:
            # Create merged config
            merged_config = ExperimentConfig(config_path=temp_path)

            return merged_config
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def _deep_merge_dicts(
        base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries, with override taking precedence.

        Parameters:
        -----------
        base : Dict[str, Any]
            Base dictionary to use as a starting point.
        override : Dict[str, Any]
            Dictionary with values that should override the base.

        Returns:
        --------
        Dict[str, Any]
            A new dictionary with merged values.
        """
        merged = base.copy()

        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = ConfigurationManager._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value

        return merged


def get_config(
    config_path: Optional[Union[str, Path]] = None,
    create_if_missing: bool = True,
    is_prediction_mode: bool = False,
    override_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> ExperimentConfig:
    """
    Get or create the global configuration instance.

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to the configuration file. If not provided, will use the existing
        instance or look for a default config.yml in the current directory.
    create_if_missing : bool
        Whether to create a default configuration if the specified file doesn't exist.
    is_prediction_mode : bool
        Whether this configuration is for prediction (vs training).
    override_params : Dict[str, Any], optional
        Dictionary of parameters to override in the loaded configuration.
    verbose : bool
        Whether to print information about the configuration being used.

    Returns:
    --------
    ExperimentConfig
        The global configuration instance.
    """
    global _CONFIG_INSTANCE

    # Return existing instance if available and no new path provided
    if _CONFIG_INSTANCE is not None and config_path is None and not override_params:
        if verbose:
            logger.info(
                f"Using existing configuration from: {_CONFIG_INSTANCE.config_path}"
            )
        return _CONFIG_INSTANCE

    # Create new instance if path provided or no instance exists
    try:
        _CONFIG_INSTANCE = ConfigurationManager.load_config(
            config_path=config_path,
            create_if_missing=create_if_missing,
            is_prediction_mode=is_prediction_mode,
            override_params=override_params,
            verbose=verbose,
        )
        return _CONFIG_INSTANCE
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise


def reset_config() -> None:
    """
    Reset the global configuration instance.

    This is primarily used for testing and situations where you need
    to ensure a fresh configuration state.
    """
    global _CONFIG_INSTANCE
    logger.debug("Resetting global config instance.")
    _CONFIG_INSTANCE = None


def create_default_config(
    output_path: Union[str, Path] = "config.yml", verbose: bool = True
) -> ExperimentConfig:
    """
    Create a default configuration file and set it as the global instance.

    Parameters:
    -----------
    output_path : str or Path
        Path where to save the default configuration file.
    verbose : bool
        Whether to print information about the configuration creation.

    Returns:
    --------
    ExperimentConfig
        The created configuration instance.
    """
    global _CONFIG_INSTANCE

    _CONFIG_INSTANCE = ConfigurationManager.create_default_config(
        output_path=output_path, verbose=verbose
    )

    return _CONFIG_INSTANCE


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file as a dictionary.

    This is a convenience function that delegates to ConfigurationManager.

    Parameters:
    -----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns:
    --------
    Dict[str, Any]
        The loaded configuration dictionary.
    """
    return ConfigurationManager.load_yaml_config(config_path)


def create_prediction_config(
    base_config: Optional[ExperimentConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> ExperimentConfig:
    """
    Create a configuration optimized for prediction.

    This is a convenience function that delegates to ConfigurationManager.

    Parameters:
    -----------
    base_config : ExperimentConfig, optional
        Base configuration to use as a starting point. If None, will use the global instance.
    config_path : str or Path, optional
        Path to the configuration file to use as a base. Used only if base_config is None
        and there is no global instance.

    Returns:
    --------
    ExperimentConfig
        A new configuration instance optimized for prediction.
    """
    if base_config is None:
        base_config = get_config()

    return ConfigurationManager.create_prediction_config(
        base_config=base_config, config_path=config_path
    )
