# gnn_package/src/utils/sensor_utils.py
import os
import json
from pathlib import Path
from gnn_package.config.paths import SENSORS_DATA_DIR  # Import from paths module

from private_uoapi import LSConfig, LSAuth, LightsailWrapper


def get_sensor_name_id_map():
    """
    Create unique IDs for each sensor from the private UOAPI.

    location: id

    For the private API, where no IDs are provided, we generate
    unique IDs of the form '1XXXX' where XXXX is a zero-padded
    index (e.g. i=1 > 10001 and i=100 > 10100).

    Returns:
    dict: Mapping between sensor names (keys) and IDs (values)
    """
    # Move the implementation here from graph_utils.py
    if not os.path.exists(SENSORS_DATA_DIR / "sensor_name_id_map.json"):

        config = LSConfig()
        auth = LSAuth(config)
        client = LightsailWrapper(config, auth)
        sensors = client.get_traffic_sensors()
        import pandas as pd

        sensors = pd.DataFrame(sensors)
        mapping = {
            location: f"1{str(i).zfill(4)}"
            for i, location in enumerate(sensors["location"])
        }

        with open(
            SENSORS_DATA_DIR / "sensor_name_id_map.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(mapping, f, indent=4)
    else:
        with open(
            SENSORS_DATA_DIR / "sensor_name_id_map.json",
            "r",
            encoding="utf-8",
        ) as f:
            mapping = json.load(f)

    return mapping
