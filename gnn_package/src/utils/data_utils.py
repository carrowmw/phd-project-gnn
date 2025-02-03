import pandas as pd


def read_pickled_gdf(dir_path, file_name):
    cropped_gdf = pd.read_pickle(dir_path + file_name)
    return cropped_gdf
