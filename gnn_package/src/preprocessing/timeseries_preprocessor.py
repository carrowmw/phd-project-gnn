from typing import Dict, List, Tuple
from datetime import timedelta
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
            print("TimeSeriesPreprocessor: No config provided, using global config")
            config = get_config()

        self.window_size = config.data.general.window_size
        self.stride = config.data.general.stride
        self.gap_threshold = pd.Timedelta(
            minutes=config.data.general.gap_threshold_minutes
        )
        self.missing_value = config.data.general.missing_value

        # Store full config for other methods
        self.config = config

    def create_windows_from_grid(
        self,
        time_series_dict,
        config=None,
    ):
        """
        Create windowed data with common time boundaries across all sensors.

        Parameters:
        -----------
        time_series_dict : Dict[str, pd.Series]
            Dictionary mapping node IDs to their time series
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

        # Get configuration
        if config is None:
            config = self.config

        # Note: We removed standardize parameter since it's now handled earlier in the pipeline

        @dataclass
        class TimeWindow:
            start_idx: int
            end_idx: int
            node_id: str
            mask: np.ndarray  # 1 for valid data, 0 for missing

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

                # Create mask based ONLY on missing value
                mask = window != self.missing_value

                # Note: Standardization is now removed from here since it's done globally

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

    def create_rolling_window_splits(
        self,
        time_series_dict,
        config=None,
    ):
        """
        Create multiple time-based splits using a rolling window approach.

        Parameters:
        -----------
        time_series_dict : Dict[str, pd.Series]
            Dictionary mapping node IDs to their time series data

        Returns:
        --------
        List[Dict[str, Dict[str, pd.Series]]]
            List of dictionaries, each containing a train/validation split
        """
        # Get configuration
        if config is None:
            config = self.config

        train_ratio = config.data.training.train_ratio
        n_splits = config.data.training.n_splits

        # Get the window size from config to use as buffer
        window_size_timedelta = pd.Timedelta(
            minutes=15 * self.window_size * config.data.general.buffer_factor
        )  # Assuming 15-min frequency

        # Find global min and max dates
        all_dates = []
        for series in time_series_dict.values():
            if len(series) > 0:
                all_dates.extend(series.index)

        min_date = min(all_dates)
        max_date = max(all_dates)
        total_days = (max_date - min_date).days

        splits = []

        # Create splits based on training ratio
        for i in range(n_splits):
            # Calculate step size for this approach
            step_size = (
                (total_days * (1 - train_ratio)) / (n_splits - 1) if n_splits > 1 else 0
            )

            # Calculate cutoff point (end of training data)
            train_days = total_days * train_ratio + (i * step_size)

            # Training data cutoff
            train_cutoff = min_date + timedelta(days=train_days)

            # Add buffer period between training and validation
            buffer_cutoff = train_cutoff + window_size_timedelta

            # Validation end
            val_end = max_date  # Use all available data after buffer

            # Skip if buffer would go beyond available data
            if buffer_cutoff >= max_date:
                continue

            train_dict = {}
            val_dict = {}

            for node_id, series in time_series_dict.items():
                # Get training data (everything before train cutoff)
                train_series = series[series.index < train_cutoff]

                # Get validation data (everything after buffer cutoff)
                val_series = series[series.index >= buffer_cutoff]

                # Only include if both parts have data
                if len(train_series) > 0 and len(val_series) > 0:
                    train_dict[node_id] = train_series
                    val_dict[node_id] = val_series

            splits.append({"train": train_dict, "val": val_dict})

        return splits

    def create_time_based_split(
        self,
        time_series_dict,
        config=None,
    ):
        """
        Split data based on time, either using a ratio or a specific cutoff date (simple solution).

        Parameters:
        -----------
        time_series_dict : Dict[str, pd.Series]
            Dictionary mapping node IDs to their time series data
        train_ratio : float, optional
            Ratio of data to use for training (by time, not by sample count)
        cutoff_date : datetime, optional
            Specific date to use as the split point (overrides train_ratio)

        Returns:
        --------
        Dict[str, Dict[str, pd.Series]]
            Dictionary containing train and validation series for each node
        """
        # Get configuration
        if config is None:
            config = self.config

        train_ratio = config.data.training.train_ratio
        cutoff_date = config.data.training.cutoff_date

        # Get the window size from config to use as buffer
        window_size_timedelta = pd.Timedelta(
            minutes=15 * self.window_size * config.data.general.buffer_factor
        )  # Assuming 15-min frequency

        train_dict = {}
        val_dict = {}

        # Find global min and max dates if we need to calculate cutoff
        if cutoff_date is None:
            all_dates = []
            for series in time_series_dict.values():
                if len(series) > 0:
                    all_dates.extend(series.index)

            min_date = min(all_dates)
            max_date = max(all_dates)
            total_days = (max_date - min_date).days

            # Calculate cutoff date based on train_ratio
            cutoff_date = min_date + timedelta(days=int(total_days * train_ratio))

        # Calculate buffer end date
        buffer_end = cutoff_date + window_size_timedelta

        # Split each sensor's data
        for node_id, series in time_series_dict.items():
            # Split based on date
            train_series = series[series.index < cutoff_date]

            # Validation data starts after the buffer period
            val_series = series[series.index >= buffer_end]

            # Only include if both parts have data
            if len(train_series) > 0 and len(val_series) > 0:
                train_dict[node_id] = train_series
                val_dict[node_id] = val_series

        return [{"train": train_dict, "val": val_dict}]

    def resample_sensor_data(self, time_series_dict, config=None):
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
        freq = config.data.general.resampling_frequency
        fill_value = config.data.general.missing_value

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

        # Create a new dictionary to return
        result_dict = resampled_dict.copy()

        # Apply standardization if enabled in config
        if config.data.general.standardize:
            print("Applying global standardization to sensor data")
            resampled_dict, stats = self.standardize_sensor_data(resampled_dict, config)

            print(
                f"Standardization complete: mean={stats['mean']:.2f}, std={stats['std']:.2f}"
            )

            # Store stats in a special key that won't interfere with sensor data
            result_dict = resampled_dict.copy()  # Update with standardized data
            result_dict["__stats__"] = (
                stats  # Use a special key unlikely to conflict with sensor IDs
            )

        return result_dict

    def standardize_sensor_data(self, time_series_dict, config=None):
        """
        Standardize sensor data globally across all sensors while preserving missing values.

        Parameters:
        -----------
        time_series_dict : Dict[str, pd.Series]
            Dictionary mapping sensor IDs to their time series data
        config : ExperimentConfig, optional
            Configuration object. If not provided, will use global config.

        Returns:
        --------
        Tuple[Dict[str, pd.Series], Dict[str, float]]
            Tuple containing:
            - Dictionary with standardized time series
            - Dictionary with statistics (mean, std) for inverse transformation
        """
        # Get configuration
        if config is None:
            config = get_config()

        # Get missing value from config
        missing_value = config.data.general.missing_value

        # Step 1: Collect all valid values across all sensors
        all_valid_values = []
        for series in time_series_dict.values():
            # Skip completely empty series
            if len(series) == 0:
                continue

            valid_mask = series.values != missing_value
            all_valid_values.append(series.values[valid_mask])

        # If we have no valid values, return the original data
        if not all_valid_values:
            print("Warning: No valid values found for standardization")
            return time_series_dict, {"mean": 0.0, "std": 1.0}

        all_valid_values = np.concatenate(all_valid_values)

        # Step 2: Calculate global statistics from all valid values
        global_mean = np.mean(all_valid_values)
        global_std = np.std(all_valid_values)

        # Add a small epsilon to avoid division by zero
        if global_std < 1e-8:
            print(
                "Warning: Very small standard deviation detected, using default value"
            )
            global_std = 1.0

        # Store these for later inverse transformation
        stats = {"mean": global_mean, "std": global_std}

        print(f"Global standardization: mean={global_mean:.2f}, std={global_std:.2f}")

        # Step 3: Apply standardization while preserving missing values
        standardized_dict = {}
        for sensor_id, series in time_series_dict.items():
            standardized_series = series.copy()
            valid_mask = series.values != missing_value

            # Only standardize valid values
            standardized_series.values[valid_mask] = (
                series.values[valid_mask] - global_mean
            ) / global_std

            standardized_dict[sensor_id] = standardized_series

        return standardized_dict, stats


# For backward compatibility
def resample_sensor_data(time_series_dict, freq=None, fill_value=None, config=None):
    """Wrapper for backward compatibility"""
    print("USING LEGACY RESAMPLING FUNCTION")
    preprocessor = TimeSeriesPreprocessor(config=config)
    return preprocessor.resample_sensor_data(time_series_dict, config)


def standardize_sensor_data(time_series_dict, config=None):
    """Wrapper for backward compatibility"""
    print("USING LEGACY STANDARDIZATION FUNCTION")
    preprocessor = TimeSeriesPreprocessor(config=config)
    return preprocessor.standardize_sensor_data(time_series_dict, config)
