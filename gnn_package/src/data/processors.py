# gnn_package/src/data/processors.py
from enum import Enum
from typing import Dict, List, Optional, Union, TypedDict, Any
import pandas as pd
from pathlib import Path

from gnn_package.src.preprocessing import TimeSeriesPreprocessor
from gnn_package.src.dataloaders import create_dataloader

from gnn_package.config import ExperimentConfig
from gnn_package.src.data.data_sources import DataSource, FileDataSource, APIDataSource

# Define the structure of the data loaders, graph data, and time series data
# {
#     "data_loaders": {
#         "train_loader": train_loader,  # Only in training mode
#         "val_loader": val_loader,      # In both modes
#     },
#     "graph_data": {
#         "adj_matrix": adj_matrix,
#         "node_ids": node_ids,
#     },
#     "time_series": {
#         "validation": validation_dict,  # Original series for validation
#         "input": input_dict,           # Series used for prediction (optional)
#     },
#     "metadata": {
#         "preprocessing_stats": preprocessing_stats,
#         "mode": "training" or "prediction"
#     }
# }


class DataLoaders(TypedDict):
    train_loader: Optional[Any]  # Use specific loader type if available
    val_loader: Any


class GraphData(TypedDict):
    adj_matrix: Any  # numpy.ndarray or similar
    node_ids: List[str]


class TimeSeriesData(TypedDict):
    validation: Dict[str, Any]  # Dict mapping node IDs to time series
    input: Optional[Dict[str, Any]]


class ProcessorMetadata(TypedDict):
    preprocessing_stats: Dict[str, Any]
    mode: str  # "training" or "prediction"


class ProcessorResult(TypedDict):
    data_loaders: DataLoaders
    graph_data: GraphData
    time_series: TimeSeriesData
    metadata: ProcessorMetadata


class ProcessorMode(Enum):
    TRAINING = "training"
    PREDICTION = "prediction"


class DataProcessorFactory:
    """Factory for creating data processors based on mode"""

    @staticmethod
    def create_processor(
        mode: ProcessorMode,
        config: ExperimentConfig,
        data_source: Optional[DataSource] = None,
    ):
        """Create the appropriate data processor based on mode"""
        if mode == ProcessorMode.TRAINING:
            return TrainingDataProcessor(config, data_source)
        elif mode == ProcessorMode.PREDICTION:
            return PredictionDataProcessor(config, data_source)
        else:
            raise ValueError(f"Unknown processor mode: {mode}")

    @staticmethod
    def create_from_config(config: ExperimentConfig, data_file: Optional[Path] = None):
        """Create appropriate processor and data source based on config"""
        # Determine if we're in prediction mode based on provided config
        is_prediction = (
            hasattr(config, "is_prediction_mode") and config.is_prediction_mode
        )

        # Create appropriate data source
        if is_prediction:
            data_source = APIDataSource()
            mode = ProcessorMode.PREDICTION
        else:
            if data_file is None:
                raise ValueError("Data file must be provided for training mode")
            data_source = FileDataSource(data_file)
            mode = ProcessorMode.TRAINING

        return DataProcessorFactory.create_processor(mode, config, data_source)


class BaseDataProcessor:
    """Base class for data processors"""

    def __init__(
        self, config: ExperimentConfig, data_source: Optional[DataSource] = None
    ):
        self.config = config
        self.data_source = data_source
        self.resampled_data = None  # Will be set during processing

    async def get_data(self) -> Dict[str, pd.Series]:
        """Get raw data from the data source"""
        if self.data_source is None:
            raise ValueError("Data source not provided")
        return await self.data_source.get_data(self.config)

    async def process_data(self):
        """Process data based on configuration"""
        raise NotImplementedError("Subclasses must implement this method")

    def _load_graph_data(self):
        """Load and process graph data with filtering for available sensors"""
        from gnn_package.src.preprocessing import (
            load_graph_data,
            compute_adjacency_matrix,
        )

        # Load graph data
        adj_matrix, node_ids, _ = load_graph_data(
            prefix=self.config.data.general.graph_prefix, return_df=False
        )

        # Only filter if we have resampled_data
        if self.resampled_data is not None:
            # Get set of sensors in resampled data
            available_sensors = set(self.resampled_data.keys())
            valid_indices = [
                i for i, node_id in enumerate(node_ids) if node_id in available_sensors
            ]

            print(
                f"Found {len(valid_indices)} nodes that match available sensors (out of {len(node_ids)})"
            )

            if len(valid_indices) < len(node_ids):
                # Filter the adjacency matrix and node_ids
                print("Filtering adjacency matrix to match available sensors...")
                import numpy as np

                filtered_node_ids = [node_ids[i] for i in valid_indices]
                filtered_adj_matrix = adj_matrix[np.ix_(valid_indices, valid_indices)]

                print(f"Filtered adjacency matrix shape: {filtered_adj_matrix.shape}")
                print(f"Filtered node_ids length: {len(filtered_node_ids)}")

                # Replace with filtered versions
                adj_matrix = filtered_adj_matrix
                node_ids = filtered_node_ids

        # Compute weighted adjacency
        weighted_adj = compute_adjacency_matrix(adj_matrix, config=self.config)

        return weighted_adj, node_ids


class TrainingDataProcessor(BaseDataProcessor):
    """Processor for training data with complex splitting"""

    async def process_data(self):
        """Process data for training with full validation splits"""
        print("TrainingDataProcessor.process_data: Starting...")

        try:
            # Get raw data
            print("Fetching raw data...")
            raw_data = await self.get_data()
            print(f"Raw data fetched: {type(raw_data)}")
            if not raw_data:
                print("WARNING: Raw data is empty or None!")
                return None

            # Process data with appropriate splitting
            print("Creating TimeSeriesPreprocessor...")
            processor = TimeSeriesPreprocessor(config=self.config)

            # Resample data
            print("Resampling data...")
            resampled_data = processor.resample_sensor_data(
                raw_data, config=self.config
            )
            print(f"Resampled data: {type(resampled_data)}")

            # Extract stats if they exist, and store them for later access
            self.preprocessing_stats = {"standardization": {}}
            if isinstance(resampled_data, dict) and "__stats__" in resampled_data:
                self.preprocessing_stats["standardization"] = resampled_data[
                    "__stats__"
                ]
                # Remove the stats from the main dictionary so it doesn't interfere with later processing
                stats = resampled_data.pop("__stats__")
                print(f"Extracted standardization stats: {stats}")

            # Use the appropriate split method based on config
            split_method = self.config.data.training.split_method
            print(f"Using split method: {split_method}")

            if split_method == "time_based":
                print("Creating time-based split...")
                split_data = processor.create_time_based_split(
                    resampled_data, config=self.config
                )
            elif split_method == "rolling_window":
                print("Creating rolling window splits...")
                split_data = processor.create_rolling_window_splits(
                    resampled_data, config=self.config
                )
            else:
                raise ValueError(f"Unknown split method: {split_method}")

            print(f"Split data created: {type(split_data)}")
            if split_data is None or not split_data:
                print("WARNING: Split data is empty or None!")
                return None

            # Continue with window creation
            print("Creating windows for training and validation...")

            # Use the first split (or only split if time-based)
            if not isinstance(split_data, list) or not split_data:
                print("ERROR: split_data is not a proper list of splits")
                return None

            # Get the first split
            first_split = split_data[0]
            print(f"Using split with keys: {first_split.keys()}")

            # Create windows for training data
            print("Creating training windows...")
            X_train, masks_train, _ = processor.create_windows_from_grid(
                first_split["train"], config=self.config
            )

            # Create windows for validation data
            print("Creating validation windows...")
            X_val, masks_val, _ = processor.create_windows_from_grid(
                first_split["val"], config=self.config
            )

            print("Loading graph data...")

            # Store resampled data for filtering in _load_graph_data
            self.resampled_data = resampled_data
            adj_matrix, node_ids = self._load_graph_data()

            # Debug info
            print(
                f"Loaded graph with {len(node_ids)} nodes, adjacency matrix shape: {adj_matrix.shape}"
            )
            print(f"Resampled data has {len(resampled_data)} sensors")

            # Check for format consistency
            print(f"First few resampled data keys: {list(resampled_data.keys())[:5]}")
            print(f"First few node_ids: {node_ids[:5]}")

            # Convert node IDs to consistent format if needed
            sample_data_key = next(iter(resampled_data.keys()))
            if type(sample_data_key) != type(node_ids[0]):
                print(
                    f"Converting node IDs from {type(node_ids[0])} to {type(sample_data_key)}"
                )
                if isinstance(sample_data_key, str):
                    node_ids = [str(nid) for nid in node_ids]
                elif isinstance(sample_data_key, int):
                    node_ids = [int(nid) for nid in node_ids]

            # Get set of sensors in resampled data
            available_sensors = set(resampled_data.keys())
            valid_indices = [
                i for i, node_id in enumerate(node_ids) if node_id in available_sensors
            ]

            print(
                f"Found {len(valid_indices)} nodes that match available sensors (out of {len(node_ids)})"
            )

            if len(valid_indices) < len(node_ids):
                # Filter the adjacency matrix and node_ids
                print("Filtering adjacency matrix to match available sensors...")
                import numpy as np

                filtered_node_ids = [node_ids[i] for i in valid_indices]
                filtered_adj_matrix = adj_matrix[np.ix_(valid_indices, valid_indices)]

                print(f"Filtered adjacency matrix shape: {filtered_adj_matrix.shape}")
                print(f"Filtered node_ids length: {len(filtered_node_ids)}")

                # Replace with filtered versions
                adj_matrix = filtered_adj_matrix
                node_ids = filtered_node_ids

            # Create data loaders
            print("Creating data loaders...")

            train_loader = create_dataloader(
                X_train,
                masks_train,
                adj_matrix,
                node_ids,
                self.config.data.general.window_size,
                self.config.data.general.horizon,
                self.config.data.general.batch_size,
                shuffle=True,
            )

            val_loader = create_dataloader(
                X_val,
                masks_val,
                adj_matrix,
                node_ids,
                self.config.data.general.window_size,
                self.config.data.general.horizon,
                self.config.data.general.batch_size,
                shuffle=False,
            )

            # Now create the result dictionary
            result = {
                "data_loaders": {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                },
                "graph_data": {
                    "adj_matrix": adj_matrix,
                    "node_ids": node_ids,
                },
                "time_series": {
                    "validation": resampled_data,  # Or appropriate validation data
                    "input": None,  # Not needed for training
                },
                "metadata": {
                    "preprocessing_stats": self.preprocessing_stats,
                    "mode": "training",
                },
            }
            print(f"Returning result with keys: {result.keys()}")
            return result

        except Exception as e:
            print(f"ERROR in TrainingDataProcessor.process_data: {e}")
            import traceback

            traceback.print_exc()
            return None  # Return None on error


class PredictionDataProcessor(BaseDataProcessor):
    """Processor for prediction data without complex validation splits"""

    async def process_data(self):
        """Process data for prediction with simple holdout for last few points"""

        # Get raw data
        raw_data = await self.get_data()

        # Process the input data to create windows
        processor = TimeSeriesPreprocessor(config=self.config)

        # Resample data
        resampled_data = processor.resample_sensor_data(raw_data, config=self.config)

        # Extract stats if they exist, and store them for later access
        self.preprocessing_stats = {"standardization": {}}
        if "__stats__" in resampled_data:
            self.preprocessing_stats["standardization"] = resampled_data["__stats__"]
            # Remove the stats from the main dictionary so it doesn't interfere with later processing
            stats = resampled_data.pop("__stats__")

        # Create simple validation split (last horizon points)
        horizon = self.config.data.general.horizon

        # For each sensor, hold out the last 'horizon' points for validation
        validation_dict = {}
        input_dict = {}

        for node_id, series in resampled_data.items():
            if len(series) > horizon:
                # Keep full series for validation purposes
                validation_dict[node_id] = series
                # Use shortened series for prediction input
                input_dict[node_id] = series[:-horizon]
            else:
                # Not enough data - use same data for both but note the limitation
                validation_dict[node_id] = series
                input_dict[node_id] = series

        X_input, masks_input, _ = processor.create_windows_from_grid(
            input_dict, config=self.config
        )

        # Store resampled data for filtering in _load_graph_data
        self.resampled_data = resampled_data

        # Load graph data
        adj_matrix, node_ids = self._load_graph_data()

        # Create dataloader for prediction (just validation, no training)
        val_loader = create_dataloader(
            X_input,
            masks_input,
            adj_matrix,
            node_ids,
            self.config.data.general.window_size,
            self.config.data.general.horizon,
            self.config.data.general.batch_size,
            shuffle=False,
        )

        result = {
            "data_loaders": {
                "val_loader": val_loader,
                # No train_loader needed
            },
            "graph_data": {
                "adj_matrix": adj_matrix,
                "node_ids": node_ids,
            },
            "time_series": {
                "validation": validation_dict,
                "input": input_dict,
            },
            "metadata": {
                "preprocessing_stats": self.preprocessing_stats,
                "mode": "prediction",
            },
        }
        return result
