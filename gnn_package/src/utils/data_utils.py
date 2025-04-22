# gnn_package/src/utils/data_utils.py

import pandas as pd
import numpy as np


def read_pickled_gdf(dir_path, file_name):
    """
    Read a pickled GeoDataFrame from a specified directory.
    Parameters:
    ----------
    dir_path : str
        Directory path where the GeoDataFrame is stored
    file_name : str
        Name of the pickled GeoDataFrame file
    Returns:
    -------
    GeoDataFrame
        The loaded GeoDataFrame
    Raises:
    -------
    FileNotFoundError
        If the file doesn't exist
    """
    cropped_gdf = pd.read_pickle(dir_path + file_name)
    return cropped_gdf


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
