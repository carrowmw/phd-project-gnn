import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_completeness_trend(time_series_dict):
    """
    Create a visualization showing how data completeness trends over time.

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping sensor IDs to their time series data

    Returns:
    --------
    plotly.graph_objects.Figure
        Completeness trend figure
    """
    # Get the time range from all sensors
    all_timestamps = set()
    for sensor_id, series in time_series_dict.items():
        all_timestamps.update(series.index)

    if not all_timestamps:
        return go.Figure().update_layout(title="No data available")

    # Get the min and max date to establish the full range
    min_date = min(all_timestamps).date()
    max_date = max(all_timestamps).date()

    # Calculate expected readings per day (assuming 15-min intervals = 96 per day)
    expected_per_day = 96

    # Create date range
    date_range = pd.date_range(min_date, max_date, freq="D")

    # Initialize a DataFrame to store completeness by date
    completeness_df = pd.DataFrame(index=date_range)

    # Process each sensor
    for sensor_id, series in time_series_dict.items():
        # Count readings per day
        daily_counts = series.groupby(series.index.date).size()

        # Calculate completeness percentage
        completeness = daily_counts / expected_per_day * 100

        # Add to the DataFrame
        completeness_df[sensor_id] = completeness

    # Calculate overall completeness (average across all sensors)
    completeness_df["overall_avg"] = completeness_df.mean(axis=1)

    # Calculate the 7-day rolling average for smoothing
    completeness_df["rolling_avg"] = (
        completeness_df["overall_avg"].rolling(window=7, min_periods=1).mean()
    )

    # Create the figure
    fig = go.Figure()

    # Add the individual sensor completeness as light traces
    for sensor_id in time_series_dict.keys():
        fig.add_trace(
            go.Scatter(
                x=completeness_df.index,
                y=completeness_df[sensor_id],
                mode="lines",
                line=dict(width=0.5, color="rgba(180,180,180,0.3)"),
                name=sensor_id,
                showlegend=False,
            )
        )

    # Add the overall average
    fig.add_trace(
        go.Scatter(
            x=completeness_df.index,
            y=completeness_df["overall_avg"],
            mode="lines",
            line=dict(width=1, color="rgba(31, 119, 180, 0.8)"),
            name="Daily Average",
        )
    )

    # Add the rolling average
    fig.add_trace(
        go.Scatter(
            x=completeness_df.index,
            y=completeness_df["rolling_avg"],
            mode="lines",
            line=dict(width=3, color="rgb(31, 119, 180)"),
            name="7-Day Rolling Average",
        )
    )

    # Update layout
    fig.update_layout(
        title="Data Completeness Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Completeness (%)",
        yaxis=dict(
            range=[0, 105], tickvals=[0, 25, 50, 75, 100]  # 0-100% with a little margin
        ),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    # Add a horizontal reference line at 100%
    fig.add_shape(
        type="line",
        x0=min_date,
        x1=max_date,
        y0=100,
        y1=100,
        line=dict(
            color="rgba(0,100,0,0.5)",
            width=1,
            dash="dash",
        ),
    )

    # Add annotation for 100% line
    fig.add_annotation(
        x=min_date + (max_date - min_date) * 0.02,  # Slightly offset from left edge
        y=100,
        text="100% Complete",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="rgba(0,100,0,0.7)"),
    )

    return fig
