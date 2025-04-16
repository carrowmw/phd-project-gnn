from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from gnn_package.config import get_config


@dataclass
class TimeWindow:
    start_idx: int
    end_idx: int
    node_id: str
    mask: np.ndarray  # 1 for valid data, 0 for missing


class TimeSeriesPreprocessor:
    def __init__(
        self,
        window_size=None,
        stride=None,
        gap_threshold=None,
        missing_value=None,
        config=None,
    ):
        """
        Initialize the preprocessor for handling time series with gaps.

        Parameters:
        -----------
        window_size : int, optional
            Size of the sliding window, overrides config if provided
        stride : int, optional
            Number of steps to move the window, overrides config if provided
        gap_threshold : pd.Timedelta, optional
            Maximum allowed time difference between consecutive points, overrides config if provided
        missing_value : float, optional
            Value to use for marking missing data, overrides config if provided
        config : ExperimentConfig, optional
            Centralized configuration object. If not provided, will use global config.
        """
        # Get configuration
        if config is None:
            config = get_config()

        # Use parameters or config values
        self.window_size = (
            window_size if window_size is not None else config.data.window_size
        )
        self.stride = stride if stride is not None else config.data.stride

        # Handle gap_threshold (either Timedelta or minutes)
        if gap_threshold is not None:
            self.gap_threshold = gap_threshold
        else:
            self.gap_threshold = pd.Timedelta(minutes=config.data.gap_threshold_minutes)

        self.missing_value = (
            missing_value if missing_value is not None else config.data.missing_value
        )

    def create_windows_from_grid(
        self,
        time_series_dict,
        standardize=None,
        config=None,
    ):
        """
        Create windowed data with common time boundaries across all sensors.

        Parameters:
        -----------
        time_series_dict : Dict[str, pd.Series]
            Dictionary mapping node IDs to their time series
        standardize : bool, optional
            Whether to standardize the data, overrides config if provided
        config : ExperimentConfig, optional
            Centralized configuration object. If not provided, will use global config.

        Returns:
        --------
        X_by_sensor : Dict[str, np.ndarray]
            Dictionary mapping node IDs to their windowed arrays
                Array of shape (n_windows, window_size)
        masks_by_sensor : Dict[str, np.ndarray]
            Dictionary mapping node IDs to their binary masks
        metadata : Dict[str,List[TimeWindow]
            Metadata for each window
        """

        @dataclass
        class TimeWindow:
            start_idx: int
            end_idx: int
            node_id: str
            mask: np.ndarray  # 1 for valid data, 0 for missing

        # Get configuration
        if config is None:
            config = get_config()

        # Use parameter or config value
        if standardize is None:
            standardize = config.data.standardize

        # Find global time range
        all_timestamps = set()
        for series in time_series_dict.values():
            all_timestamps.update(series.index)

        all_timestamps = sorted(all_timestamps)

        # Create a common grid of window start points
        window_starts = range(
            0, len(all_timestamps) - self.window_size + 1, self.stride
        )

        X_by_sensor = {}
        masks_by_sensor = {}
        metadata_by_sensor = {}

        # Process each sensor using the common time grid
        for node_id, series in time_series_dict.items():
            sensor_windows = []
            sensor_masks = []
            sensor_metadata = []

            # Reindex the series to the common time grid, filling NaNs
            common_series = pd.Series(index=all_timestamps)
            common_series.loc[series.index] = series.values

            # Create windows at each common start point
            for start_idx in window_starts:
                end_idx = start_idx + self.window_size
                window = common_series.iloc[start_idx:end_idx].values

                # Check for unexpected NaN values
                if np.any(np.isnan(window)):
                    raise ValueError(
                        "Found NaN values in input data that should have been replaced already"
                    )

                # Create mask based ONLY on -1.0 values
                mask = window != self.missing_value

                # Standardize if requested
                if standardize and np.any(mask):
                    valid_values = window[mask]
                    mean = np.mean(valid_values)
                    std = np.std(valid_values)
                    window = np.where(
                        mask, (window - mean) / (std + 1e-8), self.missing_value
                    )

                # Add feature dimension if needed
                if len(window.shape) == 1:
                    window = window.reshape(-1, 1)
                if len(mask.shape) == 1:
                    mask = mask.reshape(-1, 1)

                sensor_windows.append(window)
                sensor_masks.append(mask)

                sensor_metadata.append(
                    TimeWindow(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        node_id=node_id,
                        mask=mask,
                    )
                )

            X_by_sensor[node_id] = np.array(sensor_windows)
            masks_by_sensor[node_id] = np.array(sensor_masks)
            metadata_by_sensor[node_id] = sensor_metadata

        return X_by_sensor, masks_by_sensor, metadata_by_sensor


def resample_sensor_data(time_series_dict, freq=None, fill_value=None, config=None):
    """
    Resample all sensor time series to a consistent frequency and fill gaps.

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping sensor IDs to their time series data
    freq : str, optional
        Pandas frequency string (e.g., '15min', '1H'), overrides config if provided
    fill_value : float, optional
        Value to use for filling gaps, overrides config if provided
    config : ExperimentConfig, optional
        Centralized configuration object. If not provided, will use global config.

    Returns:
    --------
    dict
        Dictionary with resampled time series
    """

    # Get configuration
    if config is None:
        config = get_config()

    # Use parameters or config values
    if freq is None:
        freq = config.data.resampling_frequency
    if fill_value is None:
        fill_value = config.data.missing_value

    # Find global min and max dates
    all_dates = []
    for series in time_series_dict.values():
        if len(series) > 0:
            all_dates.extend(series.index)

    global_min = min(all_dates)
    global_max = max(all_dates)

    # Create common date range
    date_range = pd.date_range(start=global_min, end=global_max, freq=freq)

    # Resample each sensor's data
    resampled_dict = {}

    for sensor_id, series in time_series_dict.items():
        # Skip empty series
        if series is None or len(series) == 0:
            continue

        # Create a Series with the full date range
        resampled = pd.Series(index=date_range, dtype=float)

        # Use original values where available (handle duplicates by taking the mean)
        grouper = series.groupby(series.index)
        non_duplicate_series = grouper.mean()

        # Align with the resampled index
        resampled[non_duplicate_series.index] = non_duplicate_series

        # Fill gaps with fill_value
        resampled = resampled.fillna(fill_value)

        resampled_dict[sensor_id] = resampled

    print(f"Resampled {len(resampled_dict)} sensors to frequency {freq}")
    print(f"Each sensor now has {len(date_range)} data points")

    return resampled_dict
