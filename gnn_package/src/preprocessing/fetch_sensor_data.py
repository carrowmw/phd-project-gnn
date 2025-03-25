# gnn_package/src/preprocessing/fetch_sensor_data.py

import pickle
from datetime import datetime, timedelta
import pandas as pd
import os
from private_uoapi import (
    LSConfig,
    LSAuth,
    LightsailWrapper,
    DateRangeParams,
    convert_to_dataframe,
)
from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map


def load_sensor_data(data_file):
    """
    Load sensor data from a pickle file.

    Parameters:
    -----------
    data_file : str
        Path to the pickle file

    Returns:
    --------
    dict
        Dictionary mapping sensor IDs to time series data

    Raises:
    -------
    FileNotFoundError
        If the data file doesn't exist
    """
    if os.path.exists(data_file):
        print(f"Loading sensor data from {data_file}")
        with open(data_file, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Sensor data file {data_file} not found. "
            f"Run fetch_sensor_data.py to create it."
        )


async def fetch_and_save_sensor_data(
    data_file, days_back=7, start_date=None, end_date=None
):
    print(f"Fetching sensor data from API")

    # Initialize API client
    config = LSConfig()
    auth = LSAuth(config)
    client = LightsailWrapper(config, auth)

    print(f"Using base URL: {config.base_url}")
    print(f"Using username: {config.username}")
    print(f"Using secret key: {'*' * len(config.secret_key)}")

    # Get sensor locations
    sensor_locations = client.get_traffic_sensors()
    sensor_locations = pd.DataFrame(sensor_locations)

    # Determine date range
    if start_date is None or end_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

    # Create date range parameters
    date_range_params = DateRangeParams(
        start_date=start_date,
        end_date=end_date,
        max_date_range=timedelta(days=400),
    )

    # Get sensor name to ID mapping
    name_id_map = get_sensor_name_id_map()

    # Fetch data
    count_data = await client.get_traffic_data(date_range_params)
    counts_df = convert_to_dataframe(count_data)

    # Create time series dictionary
    counts_dict = {}
    for location in sensor_locations["location"]:
        df = counts_df[counts_df["location"] == location]
        series = pd.Series(df["value"].values, index=df["dt"])
        location_id = name_id_map[location]
        counts_dict[location_id] = series if not df.empty else None

    # Filter out None values and remove duplicates
    results_containing_data = {}
    for node_id, data in counts_dict.items():
        if data is not None:
            data = data[~data.index.duplicated(keep="first")]
            results_containing_data[node_id] = data

    # Save to file
    with open(data_file, "wb") as f:
        pickle.dump(results_containing_data, f)

    print(f"Saved sensor data to {data_file}")
    return results_containing_data
