import plotly.express as px
import pandas as pd
import numpy as np
from private_uoapi import LightsailWrapper, LSAuth, LSConfig
import geopandas as gpd
from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map


def create_sensors_map(completeness_dict):
    """
    Create an interactive map of sensor locations colored by data completeness.
    Includes bounds to prevent excessive panning and ensures the map loads properly.
    Fixed zoom level for better visibility of Newcastle upon Tyne area.

    Parameters:
    -----------
    completeness_dict : dict
        Dictionary mapping sensor IDs to their completeness percentage (0-1)

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive map with sensors
    """
    try:
        # Try to read the shapefile
        shapefile_path = "/Users/administrator/Code/python/phd-project-gnn/gnn_package/data/sensors/sensors.shp"
        gdf = gpd.read_file(shapefile_path)

        # Check the CRS - crucial step!
        print(f"Original GeoDataFrame CRS: {gdf.crs}")

        # If the data is in UK National Grid (EPSG:27700), convert to WGS84 (EPSG:4326)
        if gdf.crs == "EPSG:27700" or str(gdf.crs).find("27700") >= 0:
            print("Converting from EPSG:27700 (UK National Grid) to EPSG:4326 (WGS84)")
            gdf = gdf.to_crs("EPSG:4326")

        # Extract coordinates to a regular pandas DataFrame without geometry objects
        lat_values = []
        lon_values = []
        for point in gdf["geometry"]:
            # After conversion, y should be latitude and x should be longitude
            lat_values.append(float(point.y))
            lon_values.append(float(point.x))

        # Verify coordinates are in reasonable lat/lon range
        print(
            f"Coordinate range - Lat: {min(lat_values)} to {max(lat_values)}, Lon: {min(lon_values)} to {max(lon_values)}"
        )

        # Create a clean DataFrame without geometry objects
        df = pd.DataFrame(
            {"location": gdf["location"].tolist(), "lat": lat_values, "lon": lon_values}
        )

    except Exception as e:
        print(f"Error reading or processing shapefile: {e}")
        print("Attempting to fetch data from API instead...")

        # Fallback to using the API
        config = LSConfig()
        auth = LSAuth(config)
        client = LightsailWrapper(config, auth)
        locations = client.get_traffic_sensors()

        # Create a regular DataFrame (not GeoDataFrame)
        df = pd.DataFrame(locations)

    # Calculate center and bounds with appropriate lat/lon coordinates
    # If we have lat/lon values in df, use them
    if "lat" in df.columns and "lon" in df.columns and len(df) > 0:
        center_lat = np.mean(df["lat"])
        center_lon = np.mean(df["lon"])

        # Calculate appropriate zoom level based on the spread of data
        lat_range = max(df["lat"]) - min(df["lat"])
        lon_range = max(df["lon"]) - min(df["lon"])

        # Adjust zoom based on geographic spread (smaller range = higher zoom)
        # These values are calibrated for the Newcastle area
        if max(lat_range, lon_range) < 0.05:
            zoom_level = 13  # Very localized data
        elif max(lat_range, lon_range) < 0.1:
            zoom_level = 12  # Small area
        elif max(lat_range, lon_range) < 0.3:
            zoom_level = 11  # Medium area (typical for Newcastle)
        else:
            zoom_level = 10  # Larger area

        print(
            f"Auto-calculated zoom level: {zoom_level} based on data spread: {lat_range},{lon_range}"
        )
    else:
        # Fallback values for Newcastle upon Tyne
        center_lat = 54.97
        center_lon = -1.61
        zoom_level = 11

    # Set appropriate bounds
    lat_padding = 0.01  # Reduced padding for tighter view
    lon_padding = 0.01

    if "lat" in df.columns and "lon" in df.columns and len(df) > 0:
        min_lat = max(min(df["lat"]) - lat_padding, -90)
        max_lat = min(max(df["lat"]) + lat_padding, 90)
        min_lon = max(min(df["lon"]) - lon_padding, -180)
        max_lon = min(max(df["lon"]) + lon_padding, 180)
    else:
        # Fallback bounds for Newcastle
        min_lat = 54.93
        max_lat = 55.02
        min_lon = -1.65
        max_lon = -1.55

    # Get mapping between sensor names and IDs
    name_id_map = get_sensor_name_id_map()

    # Add completeness data
    df["sensor_id"] = df["location"].map(name_id_map)
    df["completeness"] = df["sensor_id"].map(lambda x: completeness_dict.get(x, 0))
    df["completeness_pct"] = df["completeness"] * 100

    # Debug info
    print(f"Map center: {center_lat}, {center_lon}")
    print(f"Map bounds: {min_lat}, {min_lon}, {max_lat}, {max_lon}")
    print(f"Number of sensors to display: {len(df)}")
    print(f"Using zoom level: {zoom_level}")

    # Define a custom color scale: red (low) -> yellow (medium) -> green (high)
    # This creates a more intuitive color scale for completeness
    custom_colorscale = [
        [0.0, "rgba(178, 24, 43, 1)"],  # Dark red for very low completeness
        [0.25, "rgba(239, 138, 98, 1)"],  # Light red/orange for low completeness
        [0.5, "rgba(253, 219, 121, 1)"],  # Yellow for medium completeness
        [0.75, "rgba(173, 221, 142, 1)"],  # Light green for good completeness
        [1.0, "rgba(49, 163, 84, 1)"],  # Dark green for excellent completeness
    ]

    # Create the map - note the adjusted size_max and explicit zoom level
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="completeness_pct",
        size="completeness_pct",
        color_continuous_scale=custom_colorscale,
        range_color=[0, 100],
        size_max=30,  # Smaller markers
        hover_name="location",
        hover_data={
            "completeness_pct": ":.1f",
            "lat": False,
            "lon": False,
            "sensor_id": True,
        },
        title="Sensor Locations and Data Completeness",
        labels={"completeness_pct": "Data Completeness (%)"},
    )

    fig.update_traces(
        marker=dict(opacity=0.8),
        selector=dict(mode="markers"),
    )

    # Update layout with explicit zoom level
    fig.update_layout(
        height=700,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Completeness (%)",
            ticksuffix="%",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
        ),
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            style="carto-positron",
            zoom=zoom_level,  # Explicitly set zoom level
            bounds=dict(west=min_lon, east=max_lon, south=min_lat, north=max_lat),
        ),
    )

    return fig
