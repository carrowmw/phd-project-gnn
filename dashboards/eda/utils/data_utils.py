import json
import pickle
import numpy as np
import pandas as pd


# Helper function to load data from file
def load_data(file_path="dashboards/data/test_data_1mnth.pkl"):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# Find continuous segments in time series
def find_continuous_segments(
    time_index, values, gap_threshold=pd.Timedelta(minutes=15)
):
    segments = []
    start_idx = 0

    for i in range(1, len(time_index)):
        time_diff = time_index[i] - time_index[i - 1]

        # Check for gaps in time or values
        if (time_diff > gap_threshold) or (
            np.isnan(values[i - 1]) or np.isnan(values[i])
        ):
            if i - start_idx >= 24:  # Assuming minimum window size of 24
                segments.append((start_idx, i))
            start_idx = i

    # Add the last segment if it's long enough
    if len(time_index) - start_idx >= 24:
        segments.append((start_idx, len(time_index)))

    return segments


# Load sensor location data from file
def load_sensor_geojson(file_path="dashboards/data/sensors.geojson"):
    with open(file_path, "r", encoding="utf8") as f:
        return json.load(f)
