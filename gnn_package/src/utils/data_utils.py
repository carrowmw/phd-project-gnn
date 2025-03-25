# gnn_package/src/utils/data_utils.py

import pandas as pd


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
