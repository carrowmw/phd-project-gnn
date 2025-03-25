import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sensors_comparison(time_series_dict, n_sensors=5, days_back=30):
    """
    Create an interactive comparison of top sensors' traffic patterns.

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping sensor IDs to their time series data
    n_sensors : int
        Number of top sensors to compare
    days_back : int
        Number of days to look back for the comparison

    Returns:
    --------
    plotly.graph_objects.Figure
        Sensors comparison figure
    """
    # Find sensors with most data points
    sensor_data_counts = {
        sensor_id: len(series) for sensor_id, series in time_series_dict.items()
    }
    top_sensors = sorted(sensor_data_counts, key=sensor_data_counts.get, reverse=True)[
        :n_sensors
    ]

    # Get the current date (or max date in the data)
    all_dates = []
    for series in time_series_dict.values():
        if len(series) > 0:
            all_dates.extend(series.index.tolist())

    if not all_dates:
        return go.Figure().update_layout(title="No data available")

    current_date = max(all_dates).date()
    start_date = current_date - timedelta(days=days_back)

    # Create figure
    fig = go.Figure()

    # Add a trace for each sensor
    for sensor_id in top_sensors:
        series = time_series_dict[sensor_id]

        # Filter to the selected date range
        mask = (series.index.date >= start_date) & (series.index.date <= current_date)
        filtered_series = series[mask]

        # Skip if no data in the range
        if len(filtered_series) == 0:
            continue

        # Resample to hourly data for smoother visualization
        hourly_data = filtered_series.resample("1H").mean()

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=hourly_data.index,
                y=hourly_data.values,
                mode="lines",
                name=f"Sensor {sensor_id}",
                hovertemplate="%{x}<br>Value: %{y:.1f}<extra>Sensor "
                + sensor_id
                + "</extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Top {n_sensors} Sensors: Traffic Comparison (Last {days_back} Days)",
        xaxis_title="Date",
        yaxis_title="Traffic Count",
        height=600,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )

    return fig
