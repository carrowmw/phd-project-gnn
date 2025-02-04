#  gnn_package/src/preprocessing/graph_visualisation.py

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString


def visualize_network_components(road_network_gdf):
    """
    Visualize all components of the road network in different colors.

    Parameters:
    road_network_gdf (GeoDataFrame): Network edges as GeoDataFrame

    Returns:
    tuple: (figure, axis, GeoDataFrame with component information)
    """
    # Find connected components using spatial operations
    components_gdf = road_network_gdf.copy()
    components_gdf["component"] = -1

    merged_lines = components_gdf.geometry.unary_union

    # If it's a single geometry, convert to list
    if isinstance(merged_lines, LineString):
        merged_lines = [merged_lines]
    elif isinstance(merged_lines, MultiLineString):
        merged_lines = list(merged_lines.geoms)

    # Assign component IDs
    for i, merged_line in enumerate(merged_lines):
        # Find all linestrings that intersect with this component
        mask = components_gdf.geometry.intersects(merged_line)
        components_gdf.loc[mask, "component"] = i

    # Count segments in each component
    component_sizes = components_gdf.component.value_counts()
    n_components = len(component_sizes)

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))

    # Create color map
    colors = plt.cm.rainbow(np.linspace(0, 1, n_components))

    # Plot each component
    for i, color in enumerate(colors):
        mask = components_gdf["component"] == i
        subset = components_gdf[mask]
        size = len(subset)

        # Only label larger components
        if (
            size > len(road_network_gdf) * 0.05
        ):  # Label components with >5% of total segments
            label = f"Component {i} ({size} segments)"
        else:
            label = None

        subset.plot(ax=ax, color=color, linewidth=1, alpha=0.7, label=label)

    # Add legend and title
    if ax.get_legend():  # Only add legend if there are labels
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(f"Road Network Components (Total: {n_components} components)")
    plt.tight_layout()

    # Print summary
    print("\nComponent Summary:")
    print(f"Total components: {n_components}")
    print("\nLargest components:")
    print(component_sizes.head())

    return fig, ax, components_gdf


def visualize_sensor_graph(G, points_gdf):
    """
    Visualize the sensor graph with edge weights.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create position dictionary from points GeoDataFrame
    pos = {
        row.original_id: (row.geometry.x, row.geometry.y)
        for idx, row in points_gdf.iterrows()
    }

    # Draw edges with width proportional to weight
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(weights)
    normalized_weights = [w / max_weight for w in weights]

    # Draw the graph
    nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="red")

    plt.title("Fully Connected Sensor Network Graph")
    plt.axis("on")
    ax.set_aspect("equal")

    return fig, ax
