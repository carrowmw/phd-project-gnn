# gnn_package/src/utils/sensor_utils.py
import os
import json
from pathlib import Path
import pandas as pd
from gnn_package.config.paths import SENSORS_DATA_DIR  # Import from paths module

from private_uoapi import LSConfig, LSAuth, LightsailWrapper

from gnn_package.config import get_config


def get_sensor_name_id_map(config=None):
    """
    Create unique IDs for each sensor from the private UOAPI.

    location: id

    For the private API, where no IDs are provided, we generate
    unique IDs of the form '1XXXX' where XXXX is a zero-padded
    index (e.g. i=1 > 10001 and i=100 > 10100).

    Returns:
    dict: Mapping between sensor names (keys) and IDs (values)
    """

    # Get configuration
    if config is None:
        config = get_config()

    sensor_id_prefix = config.data.sensor_id_prefix

    # Check if the mapping file already exists
    if not os.path.exists(SENSORS_DATA_DIR / "sensor_name_id_map.json"):

        config = LSConfig()
        auth = LSAuth(config)
        client = LightsailWrapper(config, auth)
        sensors = client.get_traffic_sensors()

        sensors = pd.DataFrame(sensors)

        # Create mapping using configured format
        mapping = {
            location: f"{sensor_id_prefix}{str(i).zfill(4)}"
            for i, location in enumerate(sensors["location"])
        }

        # Save the mapping to a JSON file
        print("Saving sensor name to ID mapping to file.")
        with open(
            SENSORS_DATA_DIR / "sensor_name_id_map.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(mapping, f, indent=4)
    else:
        # Load the mapping from the JSON file
        print("Loading sensor name to ID mapping from file.")
        with open(
            SENSORS_DATA_DIR / "sensor_name_id_map.json",
            "r",
            encoding="utf-8",
        ) as f:
            mapping = json.load(f)

    return mapping
