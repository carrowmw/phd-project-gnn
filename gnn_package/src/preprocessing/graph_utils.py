# gnn_package/src/preprocessing/graph_utils.py

import os
import json
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
from pathlib import Path
import uoapi
import private_uoapi
from gnn_package import PREPROCESSED_GRAPH_DIR, URBAN_OBSERVATORY_DATA_DIR


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


def get_sensor_name_id_map():
    """
    Get the mapping between sensor names and IDs from the public
    and private Urban Observatory APIs.

    For the private API, where no IDs are provided, we generate
    unique IDs of the form '1XXXX' where XXXX is a zero-padded
    index (e.g. i=1 > 10001 and i=100 > 10100).

    Returns:
    dict: Mapping between sensor names and IDs
    """

    if not os.path.exists(URBAN_OBSERVATORY_DATA_DIR / "sensor_name_id_map.json"):
        private_config = private_uoapi.APIConfig()
        private_auth = private_uoapi.APIAuth(private_config)
        private_client = private_uoapi.APIClient(private_config, private_auth)
        private_sensors = private_client.get_sensor_locations()
        private_mapping = {
            f"1{str(i).zfill(3)}": location
            for i, location in enumerate(private_sensors["location"])
        }

        public_client = uoapi.APIClient()
        public_sensors = public_client.get_sensors(theme="People")
        public_mapping = {
            sensor["Raw ID"]: sensor["Sensor Name"]
            for sensor in public_sensors["sensors"]
        }

        # Combine the two mappings
        sensor_name_id_map = {**private_mapping, **public_mapping}

        with open(
            URBAN_OBSERVATORY_DATA_DIR / "sensor_name_id_map.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(sensor_name_id_map, f, indent=4)
    else:
        with open(
            URBAN_OBSERVATORY_DATA_DIR / "sensor_name_id_map.json",
            "r",
            encoding="utf-8",
        ) as f:
            sensor_name_id_map = json.load(f)

    return sensor_name_id_map


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
