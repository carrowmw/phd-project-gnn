# src/data/data_sources.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, ClassVar, List, Union
from datetime import datetime, timedelta
import logging
import pandas as pd
from pathlib import Path

from private_uoapi import (
    LSConfig,
    LSAuth,
    LightsailWrapper,
    DateRangeParams,
    convert_to_dataframe,
)
from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map
from gnn_package.src.preprocessing import load_sensor_data
from gnn_package.config import ExperimentConfig


# Set up logger
logger = logging.getLogger(__name__)


class DataSourceException(Exception):
    """Base exception for data source errors"""

    pass


class DataSourceConnectionError(DataSourceException):
    """Exception raised when connection to data source fails"""

    pass


class DataSourceFormatError(DataSourceException):
    """Exception raised when data format is invalid"""

    pass


class DataSource(ABC):
    """
    Abstract base class for data sources.

    All data sources must implement the get_data method which returns
    a dictionary mapping sensor IDs to their time series data.
    """

    # Class variable to store source type
    source_type: ClassVar[str] = "unknown"

    @abstractmethod
    async def get_data(self, config: ExperimentConfig) -> Dict[str, pd.Series]:
        """
        Get time series data according to the provided configuration.

        Parameters:
        -----------
        config : ExperimentConfig
            Configuration object containing parameters for data retrieval

        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary mapping sensor IDs to their time series data

        Raises:
        -------
        DataSourceException
            If any error occurs during data retrieval
        """
        pass

    def validate_data(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Validate the retrieved data for consistency and format.

        Parameters:
        -----------
        data : Dict[str, pd.Series]
            Dictionary mapping sensor IDs to their time series data

        Returns:
        --------
        Dict[str, pd.Series]
            Validated data, potentially with fixes applied

        Raises:
        -------
        DataSourceFormatError
            If the data format is invalid and cannot be fixed
        """
        if not data:
            logger.warning("Data source returned empty data")
            return {}

        # Check each series
        valid_data = {}
        for sensor_id, series in data.items():
            # Skip empty series
            if series is None or len(series) == 0:
                logger.warning(f"Empty series for sensor {sensor_id}")
                continue

            # Ensure index is datetime
            if not isinstance(series.index, pd.DatetimeIndex):
                try:
                    # Try to convert index to datetime
                    series.index = pd.to_datetime(series.index)
                    logger.info(f"Converted index to datetime for sensor {sensor_id}")
                except Exception as e:
                    logger.error(
                        f"Could not convert index to datetime for sensor {sensor_id}: {e}"
                    )
                    continue

            # Remove duplicates in index
            if series.index.duplicated().any():
                original_len = len(series)
                series = series[~series.index.duplicated(keep="first")]
                logger.info(
                    f"Removed {original_len - len(series)} duplicate timestamps for sensor {sensor_id}"
                )

            # Add to valid data
            valid_data[sensor_id] = series

        return valid_data


class FileDataSource(DataSource):
    """
    Data source that loads time series data from a file.

    Supports loading pickled data in the format Dict[str, pd.Series].
    """

    source_type = "file"

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the file data source.

        Parameters:
        -----------
        file_path : str or Path
            Path to the file containing the data
        """
        self.file_path = Path(file_path)

    async def get_data(self, config: ExperimentConfig) -> Dict[str, pd.Series]:
        """
        Load time series data from a file.

        Parameters:
        -----------
        config : ExperimentConfig
            Configuration object (not used for file sources)

        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary mapping sensor IDs to their time series data

        Raises:
        -------
        DataSourceException
            If the file cannot be loaded
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            data = load_sensor_data(self.file_path)
            return self.validate_data(data)
        except Exception as e:
            logger.error(f"Error loading data from {self.file_path}: {str(e)}")
            raise DataSourceException(
                f"Failed to load data from {self.file_path}: {str(e)}"
            ) from e


class APIDataSource(DataSource):
    """
    Data source that fetches time series data from the API.

    Uses the private_uoapi package to fetch data from the API.
    """

    source_type = "api"

    def __init__(self, api_config: Optional[LSConfig] = None):
        """
        Initialize the API data source.

        Parameters:
        -----------
        api_config : LSConfig, optional
            API configuration object. If None, a default configuration is used.
        """
        self.api_config = api_config or LSConfig()
        self.auth = LSAuth(self.api_config)
        self.client = LightsailWrapper(self.api_config, self.auth)

    async def get_data(self, config: ExperimentConfig) -> Dict[str, pd.Series]:
        """
        Fetch recent data from the API based on the prediction configuration.

        Parameters:
        -----------
        config : ExperimentConfig
            Configuration object containing parameters for API request

        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary mapping sensor IDs to their time series data

        Raises:
        -------
        DataSourceConnectionError
            If connection to the API fails
        DataSourceFormatError
            If the API response format is invalid
        """
        try:
            # Get date range parameters
            date_range = await self._get_date_range_params(config)

            # Fetch data from API
            count_data = await self._fetch_data_from_api(date_range)

            # Process the response
            time_series_dict = await self._process_api_response(count_data, config)

            # Validate and return
            return self.validate_data(time_series_dict)

        except Exception as e:
            logger.error(f"Error fetching data from API: {str(e)}")
            raise DataSourceConnectionError(
                f"Failed to fetch data from API: {str(e)}"
            ) from e

    async def _get_date_range_params(self, config: ExperimentConfig) -> DateRangeParams:
        """
        Create date range parameters for the API request.

        Parameters:
        -----------
        config : ExperimentConfig
            Configuration object containing date range parameters

        Returns:
        --------
        DateRangeParams
            Date range parameters for the API request
        """
        # Get days back from config
        days_back = config.data.prediction.days_back

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Fetching data from {start_date} to {end_date} ({days_back} days)")

        # Create date range parameters
        return DateRangeParams(
            start_date=start_date,
            end_date=end_date,
            max_date_range=timedelta(days=days_back + 1),
        )

    async def _fetch_data_from_api(self, date_range: DateRangeParams) -> Any:
        """
        Fetch data from the API.

        Parameters:
        -----------
        date_range : DateRangeParams
            Date range parameters for the API request

        Returns:
        --------
        Any
            Raw API response

        Raises:
        -------
        DataSourceConnectionError
            If connection to the API fails
        """
        try:
            logger.info(f"Fetching data from API")
            return await self.client.get_traffic_data(date_range_params=date_range)
        except Exception as e:
            logger.error(f"Error connecting to API: {str(e)}")
            raise DataSourceConnectionError(
                f"Failed to connect to API: {str(e)}"
            ) from e

    async def _process_api_response(
        self, count_data: Any, config: ExperimentConfig
    ) -> Dict[str, pd.Series]:
        """
        Process the API response into a dictionary of time series.

        Parameters:
        -----------
        count_data : Any
            Raw API response
        config : ExperimentConfig
            Configuration object

        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary mapping sensor IDs to their time series data

        Raises:
        -------
        DataSourceFormatError
            If the API response format is invalid
        """
        try:
            # Convert to DataFrame
            counts_df = convert_to_dataframe(count_data)

            # Get sensor name to ID mapping
            name_id_map = get_sensor_name_id_map(config=config)
            id_to_name_map = {v: k for k, v in name_id_map.items()}

            # Create time series dictionary
            time_series_dict = {}

            for node_id in name_id_map.values():
                # Look up location name for this node ID
                location = id_to_name_map.get(node_id)
                if not location:
                    logger.warning(f"No location found for node ID {node_id}")
                    continue

                # Filter data for this location
                df = counts_df[counts_df["location"] == location]

                if df.empty:
                    logger.warning(f"No data found for location {location}")
                    continue

                # Create time series
                series = pd.Series(df["value"].values, index=df["dt"])

                # Store in dictionary
                time_series_dict[node_id] = series

            logger.info(f"Processed data for {len(time_series_dict)} sensors")
            return time_series_dict

        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            raise DataSourceFormatError(
                f"Failed to process API response: {str(e)}"
            ) from e
