from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TimeWindow:
    start_idx: int
    end_idx: int
    node_id: str
    mask: np.ndarray  # 1 for valid data, 0 for missing


class TimeSeriesPreprocessor:
    def __init__(
        self,
        window_size: int,
        stride: int,
        gap_threshold: pd.Timedelta,
        missing_value: float = -1.0,
    ):
        """
        Initialize the preprocessor for handling time series with gaps.

        Parameters:
        -----------
        window_size : int
            Size of the sliding window
        stride : int
            Number of steps to move the window
        gap_threshold : pd.Timedelta
            Maximum allowed time difference between consecutive points
            e.g., pd.Timedelta(hours=1) for hourly data
        missing_value : float
            Value to use for marking missing data
        """
        self.window_size = window_size
        self.stride = stride
        self.gap_threshold = gap_threshold
        self.missing_value = missing_value

    def find_continuous_segments(
        self, time_index: pd.DatetimeIndex, values: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Find continuous segments of data without gaps.

        Returns list of (start_idx, end_idx) tuples.
        """
        segments = []
        start_idx = 0

        for i in range(1, len(time_index)):
            time_diff = time_index[i] - time_index[i - 1]

            # Check for gaps in time or values
            if (time_diff > self.gap_threshold) or (
                np.isnan(values[i - 1]) or np.isnan(values[i])
            ):
                if i - start_idx >= self.window_size:
                    segments.append((start_idx, i))
                start_idx = i

        # Add the last segment if it's long enough
        if len(time_index) - start_idx >= self.window_size:
            segments.append((start_idx, len(time_index)))

        return segments

    def create_windows(
        self, time_series_dict: Dict[str, pd.Series], standardize: bool = True
    ) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[TimeWindow]]
    ]:
        """
        Create windowed data with masks for missing values.

        Parameters:
        -----------
        time_series_dict : Dict[str, pd.Series]
            Dictionary mapping node IDs to their time series
        standardize : bool
            Whether to standardize the data

        Returns:
        --------
        X_by_sensor : Dict[str, np.ndarray]
            Dictionary mapping node IDs to their windowed arrays
                Array of shape (n_windows, n_nodes, window_size)
        masks_by_sensor : Dict[str, np.ndarray]
            Dictionary mapping node IDs to their binary masks
        metadata : Dict[str,List[TimeWindow]
            Metadata for each window
        """
        X_by_sensor = {}
        masks_by_sensor = {}
        metadata_by_sensor = {}

        # Process each node's time series
        for node_id, series in time_series_dict.items():
            # Initialize lists for this node
            sensor_windows = []
            sensor_masks = []
            sensor_metadata = []

            # Find continuous segments
            segments = self.find_continuous_segments(series.index, series.values)

            # Process each segment
            for start_seg, end_seg in segments:
                segment_values = series.values[start_seg:end_seg]

                if standardize:
                    # Standardize non-missing values
                    valid_mask = ~np.isnan(segment_values)
                    valid_values = segment_values[valid_mask]
                    if len(valid_values) > 0:
                        mean = np.mean(valid_values)
                        std = np.std(valid_values)
                        segment_values = (segment_values - mean) / (std + 1e-8)

                # Create windows for this segment
                for i in range(
                    0, len(segment_values) - self.window_size + 1, self.stride
                ):
                    window = segment_values[i : i + self.window_size]
                    mask = ~np.isnan(window)

                    # Replace NaN with missing_value
                    window = np.where(mask, window, self.missing_value)

                    sensor_windows.append(window)
                    sensor_masks.append(mask)

                    sensor_metadata.append(
                        TimeWindow(
                            start_idx=start_seg + i,
                            end_idx=start_seg + i + self.window_size,
                            node_id=node_id,
                            mask=mask,
                        )
                    )

                # Only add if we found windows for this sensor
                if sensor_windows:
                    X_by_sensor[node_id] = np.array(sensor_windows)
                    masks_by_sensor[node_id] = np.array(sensor_masks)
                    metadata_by_sensor[node_id] = sensor_metadata

        return X_by_sensor, masks_by_sensor, metadata_by_sensor

    def prepare_batch_data(
        self,
        X_by_sensor: Dict[str, np.ndarray],
        masks_by_sensor: Dict[str, np.ndarray],
        adj_matrix: np.ndarray,
        node_ids: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for GNN training.

        Parameters:
        -----------
        X_by_sensor : Dict[str, np.ndarray]
            Dictionary mapping sensor IDs to their window arrays
        masks_by_sensor : Dict[str, np.ndarray]
            Dictionary mapping sensor IDs to their mask arrays
        adj_matrix : np.ndarray
            Adjacency matrix
        node_ids : List[str]
            List of node IDs in the same order as in the adjacency matrix

        Returns:
        --------
        dict containing:
            - node_features: Organized time series windows
            - adjacency: Adjacency matrix
            - mask: Binary mask for missing values
        """
        # Organize features in the same order as the adjacency matrix
        features = []
        masks = []

        for node_id in node_ids:
            if node_id in X_by_sensor:
                features.append(X_by_sensor[node_id])
                masks.append(masks_by_sensor[node_id])
            else:
                # Handle missing sensors
                # You might want to create a dummy placeholder or skip
                pass

        return {
            "node_features": features,  # Now a list of arrays organized by sensor
            "adjacency": adj_matrix,
            "mask": masks,  # Also organized by sensor
        }
