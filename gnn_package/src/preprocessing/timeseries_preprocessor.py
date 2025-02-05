import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import uoapi
import private_uoapi
from gnn_package import graph_utils
from dataclasses import dataclass


def get_timeseries_data_for_node(node_id):
    """
    Get time series data for a given node ID.

    Parameters:
    -----------
    node_id : str
        Node ID for the sensor

    Returns:
    --------
    data : dict
        Time series data
    """
    name_id_map = graph_utils.get_sensor_name_id_map()
    sensor_name = name_id_map[node_id]
    if node_id[0] == "1":
        print(f"DEBUG: Getting data for {node_id} from Private API")
        private_config = private_uoapi.APIConfig()
        private_auth = private_uoapi.APIAuth(private_config)
        client = private_uoapi.APIClient(private_config, private_auth)
        response = client.get_historical_traffic_counts(
            locations=sensor_name, days_back=365
        )
    elif node_id[0] == "7" or "8":
        print(f"DEBUG: Getting data for {node_id} from Public API")
        client = uoapi.APIClient()
        response = client.get_individual_raw_sensor_data(sensor_name, last_n_days=365)
        data = response["sensors"][0]["data"]["Walking"]
        return data


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
    ) -> Tuple[np.ndarray, np.ndarray, List[TimeWindow]]:
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
        X : np.ndarray
            Array of shape (n_windows, n_nodes, window_size)
        masks : np.ndarray
            Binary masks of same shape as X
        metadata : List[TimeWindow]
            Metadata for each window
        """
        all_windows = []
        all_masks = []
        window_metadata = []

        # Process each node's time series
        for node_id, series in time_series_dict.items():
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

                    all_windows.append(window)
                    all_masks.append(mask)

                    window_metadata.append(
                        TimeWindow(
                            start_idx=start_seg + i,
                            end_idx=start_seg + i + self.window_size,
                            node_id=node_id,
                            mask=mask,
                        )
                    )

        # Stack all windows
        X = np.stack(all_windows)
        masks = np.stack(all_masks)

        return X, masks, window_metadata

    def prepare_batch_data(
        self, X: np.ndarray, masks: np.ndarray, adj_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for GNN training.

        Returns:
        --------
        dict containing:
            - node_features: Time series windows
            - adjacency: Adjacency matrix
            - mask: Binary mask for missing values
        """
        return {
            "node_features": X,
            "adjacency": adj_matrix,
            "mask": masks.astype(np.float32),
        }


# # Example usage:
# if __name__ == "__main__":
#     # Create sample data
#     dates = pd.date_range('2024-01-01', '2024-01-31', freq='1H')
#     data = {
#         'node1': pd.Series(np.random.randn(len(dates)), index=dates),
#         'node2': pd.Series(np.random.randn(len(dates)), index=dates)
#     }

#     # Add some gaps
#     data['node1'].iloc[100:150] = np.nan
#     data['node2'].iloc[200:250] = np.nan

#     # Initialize preprocessor
#     preprocessor = TimeSeriesPreprocessor(
#         window_size=24,  # 24-hour windows
#         stride=12,       # 12-hour stride
#         gap_threshold=pd.Timedelta(hours=2)  # 2-hour gap threshold
#     )

#     # Process data
#     X, masks, metadata = preprocessor.create_windows(data)
#     print(f"Created {len(metadata)} windows")
#     print(f"Features shape: {X.shape}")
#     print(f"Masks shape: {masks.shape}")
