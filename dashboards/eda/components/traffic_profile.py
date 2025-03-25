import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


def create_time_of_day_profiles(time_series_dict, top_n=5):
    """
    Create profiles of traffic patterns by time of day, comparing weekday vs weekend.

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping sensor IDs to their time series data
    top_n : int
        Number of top sensors to include (with most data points)

    Returns:
    --------
    plotly.graph_objects.Figure
        Time of day profiles figure
    """
    # Find sensors with most data points
    sensor_data_counts = {
        sensor_id: len(series) for sensor_id, series in time_series_dict.items()
    }
    top_sensors = sorted(sensor_data_counts, key=sensor_data_counts.get, reverse=True)[
        :top_n
    ]

    # Create figure with subplots - one row per sensor
    fig = make_subplots(
        rows=len(top_sensors),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Sensor {sensor_id}" for sensor_id in top_sensors],
        vertical_spacing=0.05,
    )

    # Colors for weekday vs weekend
    colors = {"Weekday": "rgb(31, 119, 180)", "Weekend": "rgb(255, 127, 14)"}

    for i, sensor_id in enumerate(top_sensors):
        # Get the time series for this sensor
        series = time_series_dict[sensor_id]

        # Create DataFrame with time components
        df = pd.DataFrame(
            {
                "value": series.values,
                "hour": series.index.hour,
                "is_weekend": series.index.dayofweek >= 5,  # 5=Sat, 6=Sun
            }
        )

        # Group by hour and weekday/weekend
        weekday_data = df[~df["is_weekend"]].groupby("hour")["value"].mean()
        weekend_data = df[df["is_weekend"]].groupby("hour")["value"].mean()

        # Add weekday line
        fig.add_trace(
            go.Scatter(
                x=list(range(24)),
                y=weekday_data.reindex(range(24)).fillna(0).values,
                mode="lines+markers",
                name="Weekday" if i == 0 else None,  # Only show in legend once
                line=dict(color=colors["Weekday"], width=2),
                marker=dict(size=6),
                showlegend=(i == 0),
                legendgroup="Weekday",
            ),
            row=i + 1,
            col=1,
        )

        # Add weekend line
        fig.add_trace(
            go.Scatter(
                x=list(range(24)),
                y=weekend_data.reindex(range(24)).fillna(0).values,
                mode="lines+markers",
                name="Weekend" if i == 0 else None,  # Only show in legend once
                line=dict(color=colors["Weekend"], width=2, dash="dot"),
                marker=dict(size=6),
                showlegend=(i == 0),
                legendgroup="Weekend",
            ),
            row=i + 1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=250 * len(top_sensors),
        title_text="Time of Day Traffic Profiles: Weekday vs Weekend",
        xaxis_title="Hour of Day",
        yaxis_title="Average Traffic Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    # Update all x-axes to show hours in 24-hour format
    for i in range(len(top_sensors)):
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(0, 24, 2)),  # Show every 2 hours
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            row=i + 1,
            col=1,
        )

        # Add y-axis title only to the first subplot
        if i == 0:
            fig.update_yaxes(title_text="Avg Traffic Count", row=i + 1, col=1)

    return fig
