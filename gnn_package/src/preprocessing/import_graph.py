# gnn_package/src/preprocessing/import_graph.py

import osmnx as ox


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
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        print(f"Network downloaded and projected to: {to_crs}")
        print(f"Number of edges: {len(edges_gdf)}")

        return edges_gdf

    except Exception as e:
        print(f"Error downloading network: {str(e)}")
        raise
