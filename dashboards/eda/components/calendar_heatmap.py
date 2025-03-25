import plotly.graph_objects as go
import pandas as pd
import numpy as np
import calendar
from datetime import datetime


def create_calendar_heatmap(time_series_dict, sensor_id=None):
    """
    Create a calendar heatmap showing data patterns by day of week and hour.

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping sensor IDs to their time series data
    sensor_id : str, optional
        Specific sensor to visualize. If None, uses the sensor with most data.

    Returns:
    --------
    plotly.graph_objects.Figure
        Calendar heatmap figure
    """
    # If no sensor_id provided, use the one with most data points
    if sensor_id is None:
        sensor_data_counts = {
            sensor_id: len(series) for sensor_id, series in time_series_dict.items()
        }
        sensor_id = max(sensor_data_counts, key=sensor_data_counts.get)

    # Get the time series for this sensor
    series = time_series_dict.get(sensor_id)
    if series is None or len(series) == 0:
        return go.Figure().update_layout(
            title=f"No data available for sensor {sensor_id}"
        )

    # Extract hour and day of week
    df = pd.DataFrame(
        {
            "value": series.values,
            "hour": series.index.hour,
            "day_of_week": series.index.dayofweek,
            "date": series.index.date,
        }
    )

    # Group by hour and day of week, and calculate mean value
    pivot_data = df.pivot_table(
        values="value", index="day_of_week", columns="hour", aggfunc="mean"
    ).fillna(0)

    # Days in order (0=Monday in pandas)
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=list(range(24)),  # 24 hours
            y=days,
            colorscale="YlOrRd",
            zmin=0,
            zmax=max(1, pivot_data.values.max()),  # Ensure non-zero upper bound
            hoverongaps=False,
            colorbar=dict(title="Avg Traffic Count", titleside="right"),
            hovertemplate="Hour: %{x}<br>Day: %{y}<br>Avg Value: %{z:.1f}<extra></extra>",
        )
    )

    # Calculate data coverage percentage
    total_slots = 24 * 7
    filled_slots = np.count_nonzero(pivot_data.values)
    coverage_pct = filled_slots / total_slots * 100

    # Update layout
    fig.update_layout(
        title=f"Traffic Patterns by Day & Hour - Sensor {sensor_id} (Data Coverage: {coverage_pct:.1f}%)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}:00" for h in range(24)],
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig
