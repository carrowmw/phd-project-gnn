# gnn_package/src/preprocessing/graph_computation.py

import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
from itertools import combinations

from gnn_package.config import get_config


def compute_shortest_paths(network_gdf, snapped_points_gdf, tolerance=None):
    """
    Compute shortest paths between all pairs of snapped points.
    Assumes points have been validated using validate_snapped_points().

    Parameters:
    network_gdf (GeoDataFrame): Network edges
    snapped_points_gdf (GeoDataFrame): Validated snapped sensor points
    tolerance (int): Number of decimal places for coordinate rounding

    Returns:
    GeoDataFrame: Shortest paths between points
    """
    config = get_config()
    if tolerance is None:
        tolerance = config.data.tolerance_decimal_places

    # Create NetworkX graph from network GeoDataFrame
    G = nx.Graph()
    for idx, row in network_gdf.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords) - 1):
            start = tuple(round(x, tolerance) for x in coords[i])
            end = tuple(round(x, tolerance) for x in coords[i + 1])
            weight = Point(coords[i]).distance(Point(coords[i + 1]))
            G.add_edge(start, end, weight=weight)

    # Get point pairs with rounded coordinates
    point_coords = {
        row.original_id: tuple(
            round(x, tolerance) for x in (row.geometry.x, row.geometry.y)
        )
        for idx, row in snapped_points_gdf.iterrows()
    }

    point_pairs = list(combinations(point_coords.items(), 2))
    print(f"Attempting to find paths between {len(point_pairs)} pairs of points")

    # Compute paths
    paths = []
    total_pairs = len(point_pairs)
    failed_pairs = 0

    for i, ((id1, start_point), (id2, end_point)) in enumerate(point_pairs):
        if start_point == end_point:
            continue

        try:
            path_length = nx.shortest_path_length(
                G, start_point, end_point, weight="weight"
            )
            path = nx.shortest_path(G, start_point, end_point, weight="weight")
            path_line = LineString([Point(p) for p in path])

            paths.append(
                {
                    "start_id": id1,
                    "end_id": id2,
                    "geometry": path_line,
                    "path_length": path_length,
                    "n_points": len(path),
                }
            )

        except nx.NetworkXNoPath:
            failed_pairs += 1
            print(f"No path found between points {id1} and {id2}")
            continue

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_pairs} pairs...")

    if paths:
        paths_gdf = gpd.GeoDataFrame(paths, crs=snapped_points_gdf.crs)
        paths_gdf = paths_gdf.sort_values("path_length")

        print("\nPath finding summary:")
        print(f"Total pairs attempted: {total_pairs}")
        print(f"Failed pairs: {failed_pairs}")
        print(f"Successful paths: {len(paths)}")

        return paths_gdf
    else:
        print("No valid paths found!")
        return None


def create_weighted_graph_from_paths(paths_gdf):
    """
    Create a NetworkX graph from shortest paths data where:
    - Nodes are sensor locations
    - Edges connect sensors with weights as path lengths

    Parameters:
    -----------
    paths_gdf : GeoDataFrame
        Contains shortest paths data with start_id, end_id, and path_length

    Returns:
    --------
    G : NetworkX Graph
        Undirected weighted graph of sensor connections
    """
    # Create new undirected graph
    G = nx.Graph()

    # Add edges with weights
    for idx, row in paths_gdf.iterrows():
        G.add_edge(
            row["start_id"],
            row["end_id"],
            weight=row["path_length"],
            n_points=row["n_points"],
        )

    # Print some basic statistics
    print(f"Graph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(
        f"Average path length: {np.mean([d['weight'] for (u, v, d) in G.edges(data=True)]):.2f} meters"
    )
    print(
        f"Min path length: {min([d['weight'] for (u, v, d) in G.edges(data=True)]):.2f} meters"
    )
    print(
        f"Max path length: {max([d['weight'] for (u, v, d) in G.edges(data=True)]):.2f} meters"
    )

    # Check if graph is connected
    is_connected = nx.is_connected(G)
    print(f"Graph is {'connected' if is_connected else 'not connected'}")

    if not is_connected:
        components = list(nx.connected_components(G))
        print(f"Number of connected components: {len(components)}")
        print(f"Sizes of components: {[len(c) for c in components]}")

    return G


def compute_adjacency_matrix(
    adj_matrix: np.ndarray,
    sigma_squared=None,
    epsilon=None,
    normalization_factor=None,
) -> np.ndarray:
    """
    Computes a weighted adjacency matrix from a distance matrix using a Gaussian kernel function.
    The function first normalizes distances, then applies a Gaussian decay and thresholds weak connections.

    Parameters:
    -----------
    adj_matrix : np.ndarray
        Input matrix of distances between nodes
    sigma_squared : float, default=0.1
        Variance parameter that controls the rate of weight decay with distance.
        Smaller values cause weights to decay more quickly, while larger values
        preserve stronger long-range connections.
    epsilon : float, default=0.95
        Threshold for keeping connections. Any connection with weight below epsilon
        is removed (set to 0). For small geographical areas, a lower value like 0.5
        may be more appropriate to ensure connectivity.

    Returns:
    --------
    np.ndarray
        Weighted adjacency matrix where weights are computed using a Gaussian kernel
        function (e^(-d²/σ²)) and thresholded by epsilon. Self-connections (diagonal
        elements) are set to 0.

    Notes:
    ------
    - Distances are normalized by dividing by 10000 before computation
    - The Gaussian kernel means weights decay exponentially with squared distance
    - Higher epsilon values lead to sparser graphs as more weak connections are removed
    """
    # sigma_squared is the variance of the Gaussian kernel which controls how quickly the connection strength decays with distance
    # smaller sigma squared means weights decay more quickly with distance
    # epsilon is the threshold for the weights
    # a high value e.g. 0.95 means that only very strong connections are kept
    # for small areas epsilon=0.5 will likely be fully connected

    config = get_config()
    if sigma_squared is None:
        sigma_squared = config.data.sigma_squared
    if epsilon is None:
        epsilon = config.data.epsilon
    if normalization_factor is None:
        normalization_factor = config.data.normalization_factor

    a = adj_matrix / normalization_factor  # Normalize distances
    a_squared = a * a  # Square distances
    n = a.shape[0]
    w_mask = np.ones([n, n]) - np.identity(n)  # Mask of ones except for the diagonal
    w = (
        np.exp(-a_squared / sigma_squared)
        * (np.exp(-a_squared / sigma_squared) >= epsilon)
        * w_mask
    )  # Test whether the weights are greater than epsilon, apply the mask, and multiply again to return real values of weights
    return w
