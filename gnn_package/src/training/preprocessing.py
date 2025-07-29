# src/training/preprocessing.py
import os
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

from gnn_package.config import ExperimentConfig, get_config
from gnn_package.src.data.processors import DataProcessorFactory, ProcessorMode
from gnn_package.src.data.data_sources import FileDataSource, APIDataSource
from gnn_package.src.utils.exceptions import DataProcessingError

logger = logging.getLogger(__name__)

async def fetch_data(
    data_file: Optional[Union[str, Path]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    mode: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fetch and preprocess data for model training or prediction.

    This function creates the appropriate data processor based on the mode
    and configuration, then processes the data and returns a structured data package.

    Parameters:
    -----------
    data_file : str or Path, optional
        Path to data file for file-based data sources
    start_date : str, optional
        Start date for API-based data sources (overrides config)
    end_date : str, optional
        End date for API-based data sources (overrides config)
    config : ExperimentConfig, optional
        Configuration object
    mode : str, optional
        Processing mode: "training" or "prediction"
        (detected automatically if not provided)
    verbose : bool
        Whether to print progress information

    Returns:
    --------
    Dict[str, Any]
        Processed data package containing:
        - data_loaders: Dict with train_loader and val_loader
        - graph_data: Dict with adj_matrix and node_ids
        - time_series: Dict with validation data
        - metadata: Dict with preprocessing stats and mode

    Raises:
    -------
    DataProcessingError: If data processing fails
    """
    # Get configuration
    if config is None:
        config = get_config()

    # Create a copy of config to avoid modifying the original
    config_copy = config

    # Override date range if provided
    if start_date is not None:
        config_copy.data.general.start_date = start_date
    if end_date is not None:
        config_copy.data.general.end_date = end_date

    # Determine the processing mode
    if mode is None:
        # Auto-detect mode based on config and parameters
        if hasattr(config, "is_prediction_mode") and config.is_prediction_mode:
            processor_mode = ProcessorMode.PREDICTION
        elif data_file is None:
            # If no data file is provided, assume we're fetching recent data for prediction
            processor_mode = ProcessorMode.PREDICTION
        else:
            processor_mode = ProcessorMode.TRAINING
    else:
        # Use the specified mode
        processor_mode = ProcessorMode(mode)

    if verbose:
        logger.info(f"Processing data in {processor_mode.name} mode")

    # Create appropriate data source
    if data_file is not None:
        if verbose:
            logger.info(f"Using file data source: {data_file}")
        data_source = FileDataSource(data_file)
    else:
        if verbose:
            logger.info(f"Using API data source with date range: "
                      f"{config_copy.data.general.start_date} to {config_copy.data.general.end_date}")
        data_source = APIDataSource()

    # Create processor using factory
    if verbose:
        logger.info(f"Creating data processor for {processor_mode.name} mode")

    processor = DataProcessorFactory.create_processor(
        mode=processor_mode,
        config=config_copy,
        data_source=data_source,
    )

    # Process the data
    try:
        if verbose:
            logger.info("Processing data...")

        data_package = await processor.process_data()

        if data_package is None:
            raise DataProcessingError("Data processing returned None result")

        if verbose:
            _log_data_package_summary(data_package)

        return data_package

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise DataProcessingError(f"Failed to process data: {str(e)}") from e

async def create_cross_validation_splits(
    data_package: Dict[str, Any],
    config: Optional[ExperimentConfig] = None,
    n_splits: Optional[int] = None,
    stratify: bool = False,
) -> Dict[str, Any]:
    """
    Create cross-validation splits from a data package.

    This function takes a processed data package and creates multiple
    train/validation splits for cross-validation.

    Parameters:
    -----------
    data_package : Dict[str, Any]
        Processed data package from fetch_data()
    config : ExperimentConfig, optional
        Configuration object
    n_splits : int, optional
        Number of CV splits (overrides config)
    stratify : bool
        Whether to stratify splits (ensure similar class distribution)

    Returns:
    --------
    Dict[str, Any]
        Data package with added CV splits
    """
    if config is None:
        config = get_config()

    # Get number of splits from config if not provided
    if n_splits is None:
        n_splits = config.data.training.n_splits

    logger.info(f"Creating {n_splits} cross-validation splits")

    # Extract time series data
    if "time_series" not in data_package:
        raise ValueError("Data package must contain time series data")

    time_series_dict = data_package.get("time_series", {}).get("validation", {})
    if not time_series_dict:
        raise ValueError("Data package missing validation time series data")

    # Create preprocessor
    from gnn_package.src.preprocessing import TimeSeriesPreprocessor
    processor = TimeSeriesPreprocessor(config=config)

    # Create splits based on config strategy
    split_method = config.data.training.split_method

    if split_method == "time_based":
        # Time-based splitting
        logger.info("Using time-based splitting")
        splits_data = processor.create_time_based_split(time_series_dict, config=config)
    elif split_method == "rolling_window":
        # Rolling window splitting
        logger.info("Using rolling window splitting")
        splits_data = processor.create_rolling_window_splits(time_series_dict, config=config)
    else:
        raise ValueError(f"Unknown split method: {split_method}")

    # Process each split into data loaders
    cv_splits = []

    for i, split in enumerate(splits_data):
        logger.info(f"Processing split {i+1}/{len(splits_data)}")

        # Create windows for training data
        X_train, masks_train, _ = processor.create_windows_from_grid(
            split["train"], config=config
        )

        # Create windows for validation data
        X_val, masks_val, _ = processor.create_windows_from_grid(
            split["val"], config=config
        )

        # Get adjacency matrix and node IDs from original data package
        adj_matrix = data_package["graph_data"]["adj_matrix"]
        node_ids = data_package["graph_data"]["node_ids"]

        # Create data loaders
        from gnn_package.src.dataloaders import create_dataloader

        train_loader = create_dataloader(
            X_train,
            masks_train,
            adj_matrix,
            node_ids,
            config.data.general.window_size,
            config.data.general.horizon,
            config.data.general.batch_size,
            shuffle=True,
        )

        val_loader = create_dataloader(
            X_val,
            masks_val,
            adj_matrix,
            node_ids,
            config.data.general.window_size,
            config.data.general.horizon,
            config.data.general.batch_size,
            shuffle=False,
        )

        # Create split data package
        split_package = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "split_index": i,
            "train_size": sum(len(X_train.get(node_id, [])) for node_id in X_train),
            "val_size": sum(len(X_val.get(node_id, [])) for node_id in X_val),
        }

        cv_splits.append(split_package)

    # Add splits to data package
    data_package["splits"] = cv_splits

    # Add stats to metadata
    if "metadata" not in data_package:
        data_package["metadata"] = {}

    data_package["metadata"]["cross_validation"] = {
        "n_splits": len(cv_splits),
        "split_method": split_method,
        "split_sizes": [(s["train_size"], s["val_size"]) for s in cv_splits],
    }

    logger.info(f"Created {len(cv_splits)} cross-validation splits")
    return data_package

def _log_data_package_summary(data_package: Dict[str, Any]) -> None:
    """Log a summary of the data package contents for debugging."""
    # Check for expected keys
    expected_keys = ["data_loaders", "graph_data", "time_series", "metadata"]
    missing_keys = [k for k in expected_keys if k not in data_package]

    if missing_keys:
        logger.warning(f"Data package missing expected keys: {missing_keys}")

    # Log data loaders info
    data_loaders = data_package.get("data_loaders", {})
    logger.info(f"Data loaders: {', '.join(data_loaders.keys())}")

    # Log graph data info
    graph_data = data_package.get("graph_data", {})
    adj_matrix = graph_data.get("adj_matrix")
    node_ids = graph_data.get("node_ids", [])

    if adj_matrix is not None:
        logger.info(f"Adjacency matrix: shape={adj_matrix.shape}")
    else:
        logger.warning("Missing adjacency matrix")

    logger.info(f"Node IDs: {len(node_ids)} nodes")

    # Log time series info
    time_series = data_package.get("time_series", {})
    validation_data = time_series.get("validation", {})

    if validation_data:
        n_sensors = len(validation_data)
        sample_lengths = [len(series) for series in validation_data.values()]

        if sample_lengths:
            min_len = min(sample_lengths)
            max_len = max(sample_lengths)
            avg_len = sum(sample_lengths) / len(sample_lengths)

            logger.info(f"Time series: {n_sensors} sensors, lengths: "
                      f"min={min_len}, max={max_len}, avg={avg_len:.1f}")
    else:
        logger.warning("Missing validation time series data")

    # Log metadata info
    metadata = data_package.get("metadata", {})
    mode = metadata.get("mode")

    if mode:
        logger.info(f"Data package mode: {mode}")

    # Log preprocessing stats if available
    preprocessing_stats = metadata.get("preprocessing_stats", {})
    standardization = preprocessing_stats.get("standardization", {})

    if standardization:
        mean = standardization.get("mean")
        std = standardization.get("std")
        logger.info(f"Standardization: mean={mean}, std={std}")

async def prepare_data_for_experiment(
    data_file: Optional[Union[str, Path]] = None,
    output_cache: Optional[Union[str, Path]] = None,
    config: Optional[ExperimentConfig] = None,
    use_cross_validation: Optional[bool] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Prepare data for a training experiment, with optional caching.

    This high-level function handles:
    1. Fetching and processing data
    2. Creating cross-validation splits if requested
    3. Caching results for faster reuse

    Parameters:
    -----------
    data_file : str or Path, optional
        Path to data file
    output_cache : str or Path, optional
        Path to cache processed data
    config : ExperimentConfig, optional
        Configuration object
    use_cross_validation : bool, optional
        Whether to create CV splits (overrides config)
    force_refresh : bool
        Whether to force data refresh even if cache exists

    Returns:
    --------
    Dict[str, Any]
        Complete data package ready for training
    """
    if config is None:
        config = get_config()

    # Determine if cross-validation should be used
    if use_cross_validation is None:
        use_cross_validation = config.data.training.use_cross_validation

    # Check cache if provided
    cache_path = None
    if output_cache is not None:
        output_cache = Path(output_cache)
        os.makedirs(output_cache.parent, exist_ok=True)

        cache_path = output_cache

        if cache_path.exists() and not force_refresh:
            # Load from cache
            logger.info(f"Loading data from cache: {cache_path}")
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    data_package = pickle.load(f)

                # Validate cache
                if not isinstance(data_package, dict) or "metadata" not in data_package:
                    logger.warning("Invalid cache format, regenerating data")
                    data_package = None
                elif use_cross_validation and "splits" not in data_package:
                    logger.warning("Cache missing cross-validation splits, regenerating data")
                    data_package = None
                else:
                    logger.info("Successfully loaded data from cache")
                    return data_package

            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, regenerating data")

    # Fetch and process data
    logger.info("Fetching and processing data")
    data_package = await fetch_data(
        data_file=data_file,
        config=config,
        mode="training",
        verbose=True,
    )

    # Create cross-validation splits if requested
    if use_cross_validation:
        logger.info("Creating cross-validation splits")
        data_package = await create_cross_validation_splits(
            data_package=data_package,
            config=config,
        )

    # Save to cache if requested
    if cache_path is not None:
        logger.info(f"Saving data to cache: {cache_path}")
        try:
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(data_package, f)
            logger.info("Data successfully cached")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    return data_package