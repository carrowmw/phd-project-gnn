# gnn_package/src/data/data_sources.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
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


class DataSource(ABC):
    """Abstract base class for data sources"""

    @abstractmethod
    async def get_data(self, config) -> Dict[str, pd.Series]:
        """Get time series data according to the provided configuration"""
        pass


class FileDataSource(DataSource):
    """Data source that loads data from a file"""

    def __init__(self, file_path: Path):
        self.file_path = file_path

    async def get_data(self, config) -> Dict[str, pd.Series]:
        """Load time series data from a file"""

        # Load data from file
        return load_sensor_data(self.file_path)


class APIDataSource(DataSource):
    """Data source that fetches data from API for prediction"""

    async def get_data(self, config) -> Dict[str, pd.Series]:
        """Fetch recent data from API based on prediction config"""

        # Initialize API client
        api_config = LSConfig()
        auth = LSAuth(api_config)
        client = LightsailWrapper(api_config, auth)

        # Get sensor name to ID mapping
        name_id_map = get_sensor_name_id_map(config=config)
        id_to_name_map = {v: k for k, v in name_id_map.items()}

        # Determine date range for API request
        days_back = config.data.prediction.days_back
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Create date range parameters
        date_range_params = DateRangeParams(
            start_date=start_date,
            end_date=end_date,
            max_date_range=timedelta(days=days_back + 1),
        )

        # Fetch data from API
        count_data = await client.get_traffic_data(date_range_params)
        counts_df = convert_to_dataframe(count_data)

        # Create time series dictionary
        time_series_dict = {}

        for node_id in name_id_map.values():
            # Look up location name for this node ID
            location = id_to_name_map.get(node_id)
            if not location:
                continue

            # Filter data for this location
            df = counts_df[counts_df["location"] == location]

            if df.empty:
                continue

            # Create time series
            series = pd.Series(df["value"].values, index=df["dt"])

            # Remove duplicates
            series = series[~series.index.duplicated(keep="first")]

            # Store in dictionary
            time_series_dict[node_id] = series

        return time_series_dict
