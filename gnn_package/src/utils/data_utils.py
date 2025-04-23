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


def validate_data_package(data_package, required_components=None, mode=None):
    """
    Validate that a data package has the required structure and components.

    Parameters:
    -----------
    data_package : dict
        The data package to validate
    required_components : list, optional
        List of required components, can include:
        - 'train_loader': Training data loader
        - 'val_loader': Validation data loader
        - 'adj_matrix': Adjacency matrix
        - 'node_ids': Node IDs
        - 'time_series': Time series data
    mode : str, optional
        Expected mode ('training' or 'prediction')

    Returns:
    --------
    dict
        The validated data package

    Raises:
    -------
    ValueError
        If the data package is invalid or missing required components
    """
    # Default required components if not specified
    if required_components is None:
        required_components = []

    # Basic validation
    if not isinstance(data_package, dict):
        raise ValueError("data_package must be a dictionary")

    # Check top-level keys
    expected_keys = ["data_loaders", "graph_data", "time_series", "metadata"]
    missing_keys = [key for key in expected_keys if key not in data_package]
    if missing_keys:
        raise ValueError(
            f"data_package is missing required keys: {', '.join(missing_keys)}"
        )

    # Check data_loaders structure
    data_loaders = data_package.get("data_loaders", {})
    if not isinstance(data_loaders, dict):
        raise ValueError("data_loaders must be a dictionary")

    # Check required components
    if "train_loader" in required_components and "train_loader" not in data_loaders:
        raise ValueError("data_loaders must contain 'train_loader'")

    if "val_loader" in required_components and "val_loader" not in data_loaders:
        raise ValueError("data_loaders must contain 'val_loader'")

    # Check graph_data structure
    graph_data = data_package.get("graph_data", {})
    if not isinstance(graph_data, dict):
        raise ValueError("graph_data must be a dictionary")

    if "adj_matrix" in required_components and "adj_matrix" not in graph_data:
        raise ValueError("graph_data must contain 'adj_matrix'")

    if "node_ids" in required_components and "node_ids" not in graph_data:
        raise ValueError("graph_data must contain 'node_ids'")

    # Check metadata
    metadata = data_package.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")

    # Check mode if specified
    if mode is not None and metadata.get("mode") != mode:
        raise ValueError(f"Expected mode '{mode}' but found '{metadata.get('mode')}'")

    # All validation passed
    return data_package
