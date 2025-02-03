# gnn_package/src/preprocessing/graph_manipulation.py

import networkx as nx
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString


def snap_points_to_network(points_gdf, network, tolerance=1e-6):
    """
    Snap points to their nearest location on the network.

    Parameters:
    points_gdf (GeoDataFrame): GeoDataFrame containing points to snap
    network: Either a GeoDataFrame of linestrings or a NetworkX graph
    tolerance (float): Rounding tolerance for coordinate comparison

    Returns:
    GeoDataFrame: Points snapped to nearest network vertices
    """

    # Convert network to GeoDataFrame if it's a graph
    if isinstance(network, nx.Graph):
        print("Converting NetworkX graph to GeoDataFrame...")
        # Create linestrings from node coordinates
        edges_with_geometry = []
        for u, v in network.edges():
            # Get coordinates for both nodes
            start_coords = (network.nodes[u]["x"], network.nodes[u]["y"])
            end_coords = (network.nodes[v]["x"], network.nodes[v]["y"])

            # Create linestring
            edge_geom = LineString([start_coords, end_coords])
            edges_with_geometry.append(
                {"geometry": edge_geom, "start_node": u, "end_node": v}
            )

        network_gdf = gpd.GeoDataFrame(edges_with_geometry, crs=network.graph["crs"])
    elif isinstance(network, gpd.GeoDataFrame):
        print("Using provided GeoDataFrame as network...")
        network_gdf = network
    else:
        raise ValueError("Network must be either a GeoDataFrame or NetworkX graph")

    # Create unified network geometry
    print("Creating unified network geometry...")
    network_unary = network_gdf.geometry.union_all()

    # Get all network vertices
    network_vertices = set()
    for geom in network_gdf.geometry:
        if isinstance(geom, LineString):
            network_vertices.update(
                [tuple(round(x, tolerance) for x in coord) for coord in geom.coords]
            )
        elif isinstance(geom, Point):
            network_vertices.add(tuple(round(x, tolerance) for x in (geom.x, geom.y)))

    print(f"Number of network vertices: {len(network_vertices)}")

    # Snap points to network
    print("Snapping points to network...")
    snapped_points = []
    unsnapped_points = []

    for idx, point in points_gdf.iterrows():
        try:
            # Get the nearest point on the network
            nearest_geom = nearest_points(point.geometry, network_unary)[1]
            point_coord = (
                round(nearest_geom.x, tolerance),
                round(nearest_geom.y, tolerance),
            )

            # Find the closest network vertex to our snapped point
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

            # Create a point record
            point_record = {
                "original_id": point.get("id", idx),
                "geometry": Point(closest_vertex),
                "snap_distance": min_dist,  # Add this to check snapping quality
            }

            # Add any additional attribute from the original points
            for col in points_gdf.columns:
                if col not in ["geometry", "id"]:
                    point_record[col] = point[col]

            snapped_points.append(point_record)

        except Exception as e:
            print(f"Error processing point {point.get('id', idx)}: {str(e)}")
            unsnapped_points.append(point.get("id", idx))

    result_gdf = gpd.GeoDataFrame(snapped_points, crs=points_gdf.crs)

    # Print summary
    print("\nSnapping Summary:")
    print(f"Total points processed: {len(points_gdf)}")
    print(f"Successfully snapped points: {len(snapped_points)}")
    print(f"Failed to snap points: {len(unsnapped_points)}")

    if unsnapped_points:
        print(f"Points that failed to snap: {unsnapped_points}")

    return result_gdf


def connect_nearby_components(G, max_distance=100):
    """Add edges between nearby nodes in different components."""
    # Create a copy of the graph to modify
    G_connected = G.copy()
    print(f"G_connected is directed: {G_connected.is_directed()}")

    # Get initial components using the appropriate function based on graph type
    if G_connected.is_directed():
        components = list(nx.weakly_connected_components(G_connected))
    else:
        components = list(nx.connected_components(G_connected))

    print(f"Initial number of components: {len(components)}")

    # Track number of connections made
    connections_made = 0

    for i, comp1 in enumerate(components):
        comp1_list = list(comp1)  # Convert set to list for indexing

        for j, comp2 in enumerate(components[i + 1 :], i + 1):
            comp2_list = list(comp2)  # Convert set to list for indexing

            # Find closest pair of nodes between components
            min_dist = float("inf")
            closest_pair = None

            # Optimize by checking only a subset of nodes if components are large
            for node1 in comp1_list:
                for node2 in comp2_list:
                    dist = Point(node1).distance(Point(node2))
                    if dist < min_dist and dist < max_distance:
                        min_dist = dist
                        closest_pair = (node1, node2)

            # Add edges if found close enough nodes
            if closest_pair:
                if G_connected.is_directed():
                    G_connected.add_edge(
                        closest_pair[0], closest_pair[1], weight=min_dist
                    )
                    G_connected.add_edge(
                        closest_pair[1], closest_pair[0], weight=min_dist
                    )
                else:
                    G_connected.add_edge(*closest_pair, weight=min_dist)
                connections_made += 1
                if connections_made % 100 == 0:  # Progress update every 100 connections
                    print(f"Made {connections_made} connections...")

    # Verify the result
    if G_connected.is_directed():
        final_components = list(nx.weakly_connected_components(G_connected))
    else:
        final_components = list(nx.connected_components(G_connected))
    print(f"\nFinal number of components: {len(final_components)}")
    print(f"Total connections made: {connections_made}")

    return G_connected


def convert_graph_to_gdf(G):
    """
    Convert a NetworkX graph to a GeoDataFrame of linestrings.

    Parameters:
    G (networkx.Graph): Input graph with geometry attributes

    Returns:
    GeoDataFrame: Network as a GeoDataFrame
    """
    edges = []
    for u, v, data in G.edges(data=True):
        # Get node coordinates
        start_node = G.nodes[u]
        end_node = G.nodes[v]

        try:
            # Create linestring from node coordinates
            line = LineString(
                [(start_node["x"], start_node["y"]), (end_node["x"], end_node["y"])]
            )

            # Store edge data
            edge_data = {"geometry": line, "start_node": u, "end_node": v}
            edge_data.update(data)  # Add any additional edge attributes
            edges.append(edge_data)

        except Exception as e:
            print(f"Warning: Could not process edge {u}-{v}: {str(e)}")
            continue

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(edges, crs=G.graph.get("crs"))

    return gdf
