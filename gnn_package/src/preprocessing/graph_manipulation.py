# gnn_package/src/preprocessing/graph_manipulation.py

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString, MultiLineString
from itertools import combinations

from gnn_package.config import get_config


def snap_points_to_network(
    points_gdf, network_gdf, config=None, tolerance_decimal_places=None
):
    """
    Snap points to their nearest location on the network.

    Parameters:
    -----------
    points_gdf : GeoDataFrame
        GeoDataFrame containing points to snap
    network_gdf : GeoDataFrame
        Network edges as GeoDataFrame
    config : ExperimentConfig, optional
        Centralized configuration object. If not provided, will use global config.
    tolerance_decimal_places : int, optional
        Rounding tolerance for coordinate comparison, overrides config if provided

    Returns:
    --------
    GeoDataFrame
        Points snapped to nearest network vertices
    """

    # Get configuration
    if config is None:
        config = get_config()

    # Use parameter or config value
    if tolerance_decimal_places is None:
        tolerance_decimal_places = config.data.tolerance_decimal_places

    # Create unified network geometry
    print("Creating unified network geometry...")
    network_unary = network_gdf.geometry.union_all()

    # Get all network vertices
    print("Extracting network vertices...")
    network_vertices = set()
    for geom in network_gdf.geometry:
        if isinstance(geom, LineString):
            network_vertices.update(
                [
                    tuple(round(x, tolerance_decimal_places) for x in coord)
                    for coord in geom.coords
                ]
            )

    print(f"Number of network vertices: {len(network_vertices)}")

    # Snap points to network
    snapped_points = []
    unsnapped_points = []

    for idx, point in points_gdf.iterrows():
        try:
            # Get the nearest point on the network
            nearest_geom = nearest_points(point.geometry, network_unary)[1]
            point_coord = (
                round(nearest_geom.x, tolerance_decimal_places),
                round(nearest_geom.y, tolerance_decimal_places),
            )

            # Find the closest network vertex
            min_dist = float("inf")
            closest_vertex = None

            for vertex in network_vertices:
                dist = Point(vertex).distance(Point(point_coord))
                if dist < min_dist:
                    min_dist = dist
                    closest_vertex = vertex

            if closest_vertex is None:
                print(
                    f"Warning: Could not find closest vertex for point {point.get('id', idx)}"
                )
                unsnapped_points.append(point.get("id", idx))
                continue

            # Create point record
            point_record = {
                "original_id": point.get("id", idx),
                "geometry": Point(closest_vertex),
                "snap_distance": min_dist,
            }

            # Add any additional attributes from the original points
            for col in points_gdf.columns:
                if col not in ["geometry", "id"]:
                    point_record[col] = point[col]

            snapped_points.append(point_record)

        except Exception as e:
            print(f"Error processing point {point.get('id', idx)}: {str(e)}")
            unsnapped_points.append(point.get("id", idx))

    # Create result GeoDataFrame
    result_gdf = gpd.GeoDataFrame(snapped_points, crs=points_gdf.crs)

    # Print summary
    print("\nSnapping Summary:")
    print(f"Total points processed: {len(points_gdf)}")
    print(f"Successfully snapped points: {len(snapped_points)}")
    print(f"Failed to snap points: {len(unsnapped_points)}")

    if unsnapped_points:
        print("\nPoints that failed to snap:", unsnapped_points)

    return result_gdf


def connect_components(edges_gdf, config=None, max_distance=None):
    """
    Connect nearby components in the network using NetworkX for speed.

    Parameters:
    -----------
    edges_gdf : GeoDataFrame
        Network edges
    config : ExperimentConfig, optional
        Centralized configuration object. If not provided, will use global config.
    max_distance : float, optional
        Maximum distance to connect components, overrides config if provided

    Returns:
    --------
    GeoDataFrame
        Updated network edges with new connections
    """
    import networkx as nx
    import numpy as np
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString
    from tqdm import tqdm
    from gnn_package.config import get_config

    # Get configuration
    if config is None:
        config = get_config()

    # Use parameter or config value
    if max_distance is None:
        max_distance = config.data.max_distance

    # First convert to NetworkX graph for faster component analysis
    G = nx.Graph()

    # Add edges from the GeoDataFrame
    for idx, row in edges_gdf.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]
            G.add_edge(start, end, geometry=row.geometry)

    # Get initial components
    components = list(nx.connected_components(G))
    print(f"Initial number of components: {len(components)}")

    # Track new connections
    new_connections = []
    connections_made = 0

    # Connect components using NetworkX for speed
    for i, comp1 in enumerate(components):
        comp1_list = list(comp1)

        for j, comp2 in enumerate(components[i + 1 :], i + 1):
            comp2_list = list(comp2)

            # Find closest pair of nodes between components
            min_dist = float("inf")
            closest_pair = None

            # Use numpy for vectorized distance calculation
            coords1 = np.array(comp1_list)
            coords2 = np.array(comp2_list)

            # Calculate pairwise distances using broadcasting
            distances = np.sqrt(
                np.sum(
                    (coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]) ** 2, axis=2
                )
            )
            min_idx = np.argmin(distances)
            min_dist = distances.flat[min_idx]

            if min_dist < max_distance:
                idx1, idx2 = np.unravel_index(min_idx, distances.shape)
                closest_pair = (tuple(coords1[idx1]), tuple(coords2[idx2]))

                # Add edge to graph and track new connection
                G.add_edge(*closest_pair)
                new_connections.append(
                    {
                        "geometry": LineString([closest_pair[0], closest_pair[1]]),
                        "length": min_dist,
                        "type": "connection",
                    }
                )
                connections_made += 1

                if connections_made % 10 == 0:
                    print(f"Made {connections_made} connections...")

    # Convert new connections to GeoDataFrame
    if new_connections:
        connections_gdf = gpd.GeoDataFrame(new_connections, crs=edges_gdf.crs)
        edges_connected = pd.concat([edges_gdf, connections_gdf])
        print(f"Added {len(new_connections)} new connections")
    else:
        edges_connected = edges_gdf.copy()

    # Verify final connectivity
    final_components = nx.number_connected_components(G)
    print(f"Final number of components: {final_components}")

    return edges_connected


def create_adjacency_matrix(snapped_points_gdf, network_gdf):
    """
    Create adjacency matrix from snapped points and network.

    Parameters:
    snapped_points_gdf (GeoDataFrame): Snapped sensor points
    network_gdf (GeoDataFrame): Network edges

    Returns:
    tuple: (adjacency matrix, node IDs)
    """
    # Calculate shortest paths between all pairs of points
    paths = []
    point_pairs = list(combinations(snapped_points_gdf.iterrows(), 2))

    print(f"Calculating paths between {len(point_pairs)} point pairs...")

    # Create a NetworkX graph for shortest path calculation
    G = nx.Graph()

    # Add edges from network
    for _, row in network_gdf.iterrows():
        line = row.geometry
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            G.add_edge(
                coords[i],
                coords[i + 1],
                weight=Point(coords[i]).distance(Point(coords[i + 1])),
            )

    # Calculate paths between all pairs
    for (_, point1), (_, point2) in point_pairs:
        try:
            path_length = nx.shortest_path_length(
                G,
                source=tuple(point1.geometry.coords)[0],
                target=tuple(point2.geometry.coords)[0],
                weight="weight",
            )

            paths.append(
                {
                    "start_id": point1.original_id,
                    "end_id": point2.original_id,
                    "distance": path_length,
                }
            )

        except nx.NetworkXNoPath:
            print(
                f"No path between points {point1.original_id} and {point2.original_id}"
            )

    # Create the adjacency matrix
    if paths:
        # Create DataFrame from paths
        paths_df = pd.DataFrame(paths)

        # Get unique node IDs
        node_ids = sorted(snapped_points_gdf.original_id.unique())

        # Create empty matrix
        n = len(node_ids)
        adj_matrix = np.zeros((n, n))

        # Fill matrix with distances
        id_to_idx = {id_: i for i, id_ in enumerate(node_ids)}
        for _, row in paths_df.iterrows():
            i = id_to_idx[row.start_id]
            j = id_to_idx[row.end_id]
            adj_matrix[i, j] = row.distance
            adj_matrix[j, i] = row.distance  # Symmetric matrix

        return adj_matrix, node_ids
    else:
        print("No valid paths found!")
        return None, None


def explode_multilinestrings(gdf):
    # Create list to store new rows
    rows = []

    # Iterate through each row in the GDF
    for idx, row in gdf.iterrows():
        if isinstance(row.geometry, MultiLineString):
            # If geometry is MultiLineString, create new row for each LineString
            for line in row.geometry.geoms:
                new_row = row.copy()
                new_row.geometry = line
                rows.append(new_row)
        else:
            # If geometry is already LineString, keep as is
            rows.append(row)

    # Create new GeoDataFrame from expanded rows
    new_gdf = gpd.GeoDataFrame(rows, crs=gdf.crs)
    return new_gdf
