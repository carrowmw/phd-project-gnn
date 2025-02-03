# gnn_package/src/preprocessing/import_graph.py

import osmnx as ox


def get_pedestrian_network(place_name, network_type="walk", to_crs="EPSG:27700"):
    """
    Extract the walkable pedestrian network for a specified area.

    Parameters:
    place_name (str): Name of the place (e.g., 'Berkeley, California')
    network_type (str): Type of network to extract (default: 'walk')
    to_crs (str): Target coordinate reference system (default: 'EPSG:27700' for British National Grid)

    Returns:
    G (networkx.MultiDiGraph): Network graph of pedestrian paths, projected to specified CRS
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
        # Download the street network
        print(f"\nDownloading network for: {place_name}")
        G = ox.graph_from_place(
            place_name,
            network_type=network_type,
            custom_filter=custom_filter,
            simplify=True,
        )

        # Get the original CRS
        original_crs = G.graph["crs"]
        print(f"Original network CRS: {original_crs}")

        # Project the graph to the specified CRS
        try:
            G = ox.project_graph(G, to_crs=to_crs)
            print(f"Network successfully projected to: {to_crs}")
        except Exception as e:
            print(f"Warning: Could not project to {to_crs}. Error: {str(e)}")
            print("Falling back to UTM projection...")
            # Fall back to UTM projection
            G = ox.project_graph(G)
            print(f"Network projected to UTM: {G.graph['crs']}")

        return G

    except Exception as e:
        print(f"Error downloading network: {str(e)}")
        raise
