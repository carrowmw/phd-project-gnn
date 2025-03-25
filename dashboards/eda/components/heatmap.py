import numpy as np
import pandas as pd
import plotly.express as px


# Create an interactive data availability heatmap
def interactive_data_availability(time_series_dict):
    """Create an interactive heatmap showing data availability across sensors over time"""
    # Get all unique timestamps from all sensors
    all_timestamps = set()
    for sensor_id, series in time_series_dict.items():
        all_timestamps.update(series.index)

    all_timestamps = sorted(all_timestamps)

    # Create a DataFrame with all timestamps and fill with NaN
    data_matrix = pd.DataFrame(index=all_timestamps)

    # For each sensor, add a column to the DataFrame
    for sensor_id, series in time_series_dict.items():
        data_matrix[sensor_id] = np.nan
        # Only fill in data that exists
        data_matrix.loc[series.index, sensor_id] = 1

    # Resample to a lower resolution for better visualization if too many datapoints
    if len(all_timestamps) > 1000:
        data_matrix = data_matrix.resample("1h").mean()

    # Convert to long format for plotly
    data_long = data_matrix.reset_index().melt(
        id_vars="index", var_name="sensor_id", value_name="has_data"
    )

    # Create the heatmap with plotly
    fig = px.density_heatmap(
        data_long,
        x="index",
        y="sensor_id",
        z="has_data",
        color_continuous_scale=[
            [0, "rgba(255,255,255,0)"],  # Transparent for NaN
            [0.5, "rgba(222,235,247,1)"],  # Light blue
            [1, "rgba(49,130,189,1)"],  # Dark blue
        ],
        title="Data Availability Across Sensors (Interactive)",
        labels={
            "index": "Date",
            "sensor_id": "Sensor ID",
            "has_data": "Data Available",
        },
    )

    # Update layout
    fig.update_layout(
        height=800,
        xaxis_title="Date",
        yaxis_title="Sensor ID",
        title_x=0.5,
        coloraxis_showscale=False,
    )

    return fig
