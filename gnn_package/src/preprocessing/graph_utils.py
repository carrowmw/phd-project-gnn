# gnn_package/src/preprocessing/graph_utils.py

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import private_uoapi
from shapely.geometry import Polygon
from gnn_package.config.paths import (
    PREPROCESSED_GRAPH_DIR,
    SENSORS_DATA_DIR,
)
from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map


def read_or_create_sensor_nodes():
    FILE_PATH = SENSORS_DATA_DIR / "sensors.shp"
    if os.path.exists(FILE_PATH):
        print("Reading private sensors from file")
        sensors_gdf = gpd.read_file(FILE_PATH)
        return sensors_gdf
    else:
        config = private_uoapi.LSConfig()
        auth = private_uoapi.LSAuth(config)
        client = private_uoapi.LightsailWrapper(config, auth)
        locations = client.get_traffic_sensors()
        locations = pd.DataFrame(locations)
        sensors_gdf = gpd.GeoDataFrame(
            locations["location"],
            geometry=gpd.points_from_xy(locations["lon"], locations["lat"]),
            crs="EPSG:4326",
        )
        sensors_gdf = sensors_gdf.to_crs("EPSG:27700")
        # Add sensor IDs to the GeoDataFrame
        sensor_name_id_map = get_sensor_name_id_map()
        sensors_gdf["id"] = sensors_gdf["location"].apply(
            lambda x: sensor_name_id_map[x]
        )
        print(f"DEBUG: Column names: {sensors_gdf.columns}")
        sensors_gdf.to_file(FILE_PATH)
        return sensors_gdf


def get_bbox_transformed():
    polygon_bbox = Polygon(
        [
            [-1.65327, 54.93188],
            [-1.54993, 54.93188],
            [-1.54993, 55.02084],
            [-1.65327, 55.02084],
        ]
    )
    #     polygon_bbox = Polygon(
    #     [
    #         [-1.61327, 54.96188],
    #         [-1.59993, 54.96188],
    #         [-1.59993, 54.98084],
    #         [-1.61327, 54.98084],
    #     ]
    #   )

    # Create a GeoDataFrame from the bounding box polygon
    bbox_gdf = gpd.GeoDataFrame(geometry=[polygon_bbox], crs="EPSG:4326")

    # Assuming your road data is in British National Grid (EPSG:27700)
    # Transform the bbox to match the road data's CRS
    bbox_transformed = bbox_gdf.to_crs("EPSG:27700")
    return bbox_transformed


def get_street_network_gdfs(place_name, to_crs="EPSG:27700"):
    """
    Extract the walkable network for a specified area as GeoDataFrames.

    Parameters:
    place_name (str): Name of the place (e.g., 'Newcastle upon Tyne, UK')
    to_crs (str): Target coordinate reference system (default: 'EPSG:27700' for British National Grid)

    Returns:
    GeoDataFrame: Network edges as linestrings
    """
    # Configure OSMnx settings
    ox.settings.use_cache = True
    ox.settings.log_console = True

    # Custom filter for pedestrian-specific infrastructure
    custom_filter = (
        '["highway"~"footway|path|pedestrian|steps|corridor|'
        'track|service|living_street|residential|unclassified"]'
        '["area"!~"yes"]["access"!~"private"]'
    )

    try:
        print(f"\nDownloading network for: {place_name}")
        # Download and project the network
        G = ox.graph_from_place(
            place_name, network_type="walk", custom_filter=custom_filter, simplify=True
        )
        G = ox.project_graph(G, to_crs=to_crs)

        # Convert to GeoDataFrames and return only edges
        _, edges_gdf = ox.graph_to_gdfs(G)
        print(f"Network downloaded and projected to: {to_crs}")
        print(f"Number of edges: {len(edges_gdf)}")

        return edges_gdf

    except Exception as e:
        print(f"Error downloading network: {str(e)}")
        raise


def save_graph_data(adj_matrix, node_ids, prefix="graph"):
    """
    Save adjacency matrix and node IDs with proper metadata.

    Parameters:
    -----------
    adj_matrix : np.ndarray
        The adjacency matrix
    node_ids : list or np.ndarray
        List of node IDs corresponding to matrix rows/columns
    output_dir : str or Path
        Directory to save the files
    prefix : str
        Prefix for the saved files
    """
    output_dir = Path(PREPROCESSED_GRAPH_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the adjacency matrix
    np.save(output_dir / f"{prefix}_adj_matrix.npy", adj_matrix)

    # Save node IDs with metadata
    node_metadata = {
        "node_ids": list(
            map(str, node_ids)
        ),  # Convert to strings for JSON compatibility
        "matrix_shape": adj_matrix.shape,
        "creation_metadata": {
            "num_nodes": len(node_ids),
            "matrix_is_symmetric": np.allclose(adj_matrix, adj_matrix.T),
        },
    }

    with open(output_dir / f"{prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(node_metadata, f, indent=2)


def load_graph_data(prefix="graph", return_df=False):
    """
    Load adjacency matrix with associated node IDs.

    Parameters:
    -----------
    input_dir : str or Path
        Directory containing the saved files
    prefix : str
        Prefix of the saved files
    return_df : bool
        If True, returns a pandas DataFrame instead of numpy array

    Returns:
    --------
    tuple : (adj_matrix, node_ids, metadata)
        - adj_matrix: numpy array or DataFrame of the adjacency matrix
        - node_ids: list of node IDs
        - metadata: dict containing additional graph information
    """
    input_dir = Path(PREPROCESSED_GRAPH_DIR)

    # Load the adjacency matrix
    adj_matrix = np.load(input_dir / f"{prefix}_adj_matrix.npy")

    # Load metadata
    with open(input_dir / f"{prefix}_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    node_ids = metadata["node_ids"]

    # Verify matrix shape matches metadata
    assert adj_matrix.shape == tuple(metadata["matrix_shape"]), "Matrix shape mismatch!"

    # Optionally convert to DataFrame
    if return_df:
        adj_matrix = pd.DataFrame(adj_matrix, index=node_ids, columns=node_ids)

    return adj_matrix, node_ids, metadata


def graph_to_adjacency_matrix_and_nodes(G) -> tuple:
    """
    Convert a NetworkX graph to an adjacency matrix.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    np.ndarray
        The adjacency matrix as a dense numpy array.
    list
        The list of node IDs in the same order as the rows/columns of the matrix.
    """
    # Get a sorted list of node IDs to ensure consistent ordering
    node_ids = sorted(list(G.nodes()))

    # Create the adjacency matrix using NetworkX's built-in function
    adj_matrix = nx.adjacency_matrix(G, nodelist=node_ids, weight="weight")

    # Convert to dense numpy array for easier viewing
    adj_matrix_dense = adj_matrix.todense()

    return adj_matrix_dense, node_ids


def create_networkx_graph_from_adj_matrix(adj_matrix, node_ids, names_dict=None):
    """
    Create a NetworkX graph from adjacency matrix and node IDs.

    Parameters:
    -----------
    adj_matrix : np.ndarray
        The adjacency matrix
    node_ids : list
        List of node IDs
    names_dict : dict, optional
        Dictionary mapping node IDs to names

    Returns:
    --------
    networkx.Graph
        The reconstructed graph with all metadata
    """
    G = nx.Graph()

    # Add nodes with names if provided
    for i, node_id in enumerate(node_ids):
        node_attrs = {"id": node_id}
        if names_dict and str(node_id) in names_dict:
            node_attrs["name"] = names_dict[str(node_id)]
        G.add_node(node_id, **node_attrs)

    # Add edges with weights
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            weight = adj_matrix[i, j]
            if weight > 0:
                G.add_edge(node_ids[i], node_ids[j], weight=weight)

    return G
