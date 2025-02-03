# gnn_package/src/preprocessing/graph_analysis.py
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np


def analyze_network_graph(G):
    """
    Analyze the network properties.

    Parameters:
    G (networkx.MultiDiGraph): Network graph to analyze
    """
    print("\nAnalyzing network properties...")

    # Get the graph's CRS
    graph_crs = G.graph.get("crs", "Unknown")
    print(f"Network CRS: {graph_crs}")

    # Basic network statistics
    stats = {
        "Nodes": len(G.nodes()),
        "Edges": len(G.edges()),
        "Average node degree": np.mean([d for n, d in G.degree()]),
        "Network type": "Directed" if G.is_directed() else "Undirected",
    }

    # Print statistics
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Calculate network area
    try:
        # Get network bounds
        nodes = pd.DataFrame(
            {
                "x": [G.nodes[node]["x"] for node in G.nodes()],
                "y": [G.nodes[node]["y"] for node in G.nodes()],
            }
        )

        # Create a polygon from the bounds
        bbox = box(
            nodes["x"].min(), nodes["y"].min(), nodes["x"].max(), nodes["y"].max()
        )

        # Since we're already in EPSG:27700 (British National Grid),
        # we can calculate the area directly
        area = bbox.area

        # Convert to km²
        area_km2 = area / 1_000_000  # Convert square meters to square kilometers

        print(f"\nNetwork area: {area_km2:.2f} km²")

        # Calculate network density
        network_length = sum(d.get("length", 0) for u, v, d in G.edges(data=True))

        density = network_length / area if area > 0 else 0
        print(f"Network density: {density:.2f} meters per square meter")

        # Add to stats
        stats.update(
            {
                "Area (km²)": area_km2,
                "Total network length (km)": network_length / 1000,
                "Network density (km/km²)": density,
            }
        )

    except Exception as e:
        print(f"\nWarning: Could not calculate network area: {str(e)}")

    # Additional network metrics
    try:
        # Average street length
        avg_street_length = np.mean(
            [d.get("length", 0) for u, v, d in G.edges(data=True)]
        )
        print(f"Average street segment length: {avg_street_length:.2f} meters")

        # Number of connected components
        if G.is_directed():
            n_components = nx.number_weakly_connected_components(G)
            print(f"Number of weakly connected components: {n_components}")
        else:
            n_components = nx.number_connected_components(G)
            print(f"Number of connected components: {n_components}")

        stats["Average segment length (m)"] = avg_street_length
        stats["Number of components"] = n_components

    except Exception as e:
        print(f"\nWarning: Could not calculate some network metrics: {str(e)}")

    return stats


def analyze_graph_components(G, snapped_points_gdf, tolerance=1e-6):
    """
    Analyze which components the snapped points belong to and verify network connectivity.

    Args:
        G: NetworkX graph (directed or undirected)
        snapped_points_gdf: GeoDataFrame of snapped points
        tolerance: Distance tolerance for considering a point connected to the network

    Returns:
        GeoDataFrame with component information and connectivity status
    """
    print("\nAnalyzing network connectivity...")
    print(f"\nDetected a {'directed' if G.is_directed() else 'undirected'} graph.")

    # First verify if the graph is directed
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    # Get all network nodes as Points using x and y coordinates from node attributes
    network_nodes = {}
    for node in G.nodes():
        # Get coordinates from node attributes
        node_data = G.nodes[node]
        if "x" in node_data and "y" in node_data:
            coords = (node_data["x"], node_data["y"])
            network_nodes[node] = Point(coords)
        else:
            # If node is already a coordinate tuple
            try:
                if isinstance(node, (tuple, list)) and len(node) >= 2:
                    network_nodes[node] = Point(node)
            except Exception:
                print(f"Warning: Could not get coordinates for node {node}")
                continue

    # Create a mapping of nodes to their component index
    node_to_component = {}
    for i, component in enumerate(components):
        for node in component:
            node_to_component[node] = i

    # Check each snapped point
    point_components = []
    unconnected_points = []

    for idx, point in snapped_points_gdf.iterrows():
        coords = tuple(round(x, 6) for x in (point.geometry.x, point.geometry.y))

        # Find the closest network node and its component
        min_dist = float("inf")
        closest_node = None
        component_idx = None

        for node, node_point in network_nodes.items():
            dist = point.geometry.distance(node_point)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
                component_idx = node_to_component.get(node, -1)

        # Check if the point is connected (within tolerance)
        if min_dist <= tolerance:
            if min_dist > 0:  # Only print warning if not exact match
                print(
                    f"Warning: Point {point.original_id} was not exactly on network node but within {min_dist:.6f} units of node {closest_node}."
                )
        else:
            component_idx = -1
            unconnected_points.append(
                {
                    "original_id": point.original_id,
                    "coords": coords,
                    "min_distance": min_dist,
                }
            )

        point_components.append(
            {
                "original_id": point.original_id,
                "component": component_idx,
                "geometry": point.geometry,
                "connected": component_idx != -1,
                "distance_to_network": min_dist,
            }
        )

    # Create new GeoDataFrame with component information
    result_gdf = gpd.GeoDataFrame(point_components, crs=snapped_points_gdf.crs)

    # Print summary statistics
    print("\nNetwork Connectivity Analysis:")
    print(f"Total points: {len(result_gdf)}")
    print(f"Connected points: {sum(result_gdf['connected'])}")
    print(f"Unconnected points: {sum(~result_gdf['connected'])}")

    if unconnected_points:
        print("\nWARNING: The following points are not connected to the network:")
        for p in unconnected_points:
            print(
                f"Point ID: {p['original_id']}: distance to nearest node = {p['min_distance']:.6f}"
            )

    print("\nPoints per component:")
    component_counts = result_gdf[result_gdf["connected"]]["component"].value_counts()
    print(component_counts)

    # Calculate average distance to network
    avg_distance = np.mean(result_gdf["distance_to_network"])
    max_distance = np.max(result_gdf["distance_to_network"])
    print(f"\nAverage distance to network: {avg_distance:.6f}")
    print(f"Maximum distance to network: {max_distance:.6f}")

    return result_gdf
