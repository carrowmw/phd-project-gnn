# dashboards/gnn_diagnostics/components/missing_data_analyzer.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.data_utils import analyze_missing_data, get_data_for_sensor


def create_missing_data_analysis(experiment_data, sensor_id=None):
    """
    Create a visualization of missing data patterns

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data
    sensor_id : str, optional
        ID of the sensor to analyze

    Returns:
    --------
    plotly.graph_objects.Figure
        Missing data analysis visualization
    """
    # Create a figure with multiple subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Missing Data Pattern Over Time",
            "Missing Data by Hour of Day",
            "Missing Data Statistics",
            "Missing Data by Day of Week"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Check if sensor_id is provided
    if not sensor_id:
        # No sensor_id provided
        fig.add_annotation(
            text="No sensor selected",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig

    # Get missing value from config
    missing_value = -999.0
    if "config" in experiment_data and "data" in experiment_data["config"] and "general" in experiment_data["config"]["data"]:
        missing_value = experiment_data["config"]["data"]["general"].get("missing_value", -999.0)

    # Get data for this sensor
    sensor_data = get_data_for_sensor(experiment_data, sensor_id)

    # Analyze missing data
    missing_data_analysis = analyze_missing_data(experiment_data, sensor_id)

    # Get raw data series
    time_series = None

    # First try to get from raw_data
    if "raw_data" in sensor_data:
        time_series = sensor_data["raw_data"]
    # Then try validation_series
    elif "validation_series" in sensor_data:
        time_series = sensor_data["validation_series"]

    # Check if we have time series data
    if time_series is None or (hasattr(time_series, "empty") and time_series.empty):
        # No time series data
        fig.add_annotation(
            text=f"No time series data found for sensor {sensor_id}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig

    # Convert to pandas Series if it's not already
    if not isinstance(time_series, pd.Series):
        if hasattr(time_series, "values") and hasattr(time_series, "index"):
            time_series = pd.Series(time_series.values, index=time_series.index)
        else:
            # Try to convert to Series
            try:
                time_series = pd.Series(time_series)
            except Exception as e:
                # Failed to convert
                fig.add_annotation(
                    text=f"Error converting time series data: {str(e)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
                return fig

    # Create a mask for missing values
    missing_mask = time_series == missing_value

    # Color the missing time steps red in the time series
    df = pd.DataFrame({
        'timestamp': time_series.index,
        'value': time_series.values,
        'is_missing': missing_mask
    })

    # Plot pattern of missing data over time
    # Group by day and calculate missing percentage
    df['date'] = df['timestamp'].dt.date
    missing_by_date = df.groupby('date').apply(lambda x: (x['is_missing'].sum() / len(x)) * 100)
    missing_dates = missing_by_date.index

    # Plot missing data pattern
    fig.add_trace(
        go.Bar(
            x=missing_dates,
            y=missing_by_date.values,
            marker_color='red',
            name='Missing Data %',
            hovertemplate='Date: %{x}<br>Missing: %{y:.1f}%<extra></extra>'
        ),
        row=1,
        col=1,
    )

    # Plot missing data by hour of day
    df['hour'] = df['timestamp'].dt.hour
    missing_by_hour = df.groupby('hour').apply(lambda x: (x['is_missing'].sum() / len(x)) * 100)
    hours = missing_by_hour.index

    fig.add_trace(
        go.Bar(
            x=hours,
            y=missing_by_hour.values,
            marker_color='orange',
            name='Missing by Hour',
            hovertemplate='Hour: %{x}<br>Missing: %{y:.1f}%<extra></extra>'
        ),
        row=1,
        col=2,
    )

    # Plot missing data by day of week
    df['day_of_week'] = df['timestamp'].dt.day_name()
    # Order days of week correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    missing_by_day = df.groupby('day_of_week').apply(lambda x: (x['is_missing'].sum() / len(x)) * 100)
    missing_by_day = missing_by_day.reindex(day_order)

    fig.add_trace(
        go.Bar(
            x=missing_by_day.index,
            y=missing_by_day.values,
            marker_color='purple',
            name='Missing by Day',
            hovertemplate='Day: %{x}<br>Missing: %{y:.1f}%<extra></extra>'
        ),
        row=2,
        col=2,
    )

    # Create statistics table
    total_points = len(df)
    missing_points = df['is_missing'].sum()
    missing_pct = missing_points / total_points * 100 if total_points > 0 else 0

    longest_gap = 0
    current_gap = 0

    for is_missing in df['is_missing']:
        if is_missing:
            current_gap += 1
            longest_gap = max(longest_gap, current_gap)
        else:
            current_gap = 0

    # Identify time periods with most missing data
    df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')
    missing_by_month = df.groupby('year_month').apply(lambda x: (x['is_missing'].sum() / len(x)) * 100)
    worst_month = missing_by_month.idxmax() if not missing_by_month.empty else "N/A"
    worst_month_pct = missing_by_month.max() if not missing_by_month.empty else 0

    # Identify hours with most missing data
    worst_hour = missing_by_hour.idxmax() if not missing_by_hour.empty else "N/A"
    worst_hour_pct = missing_by_hour.max() if not missing_by_hour.empty else 0

    # Create a table
    table_data = [
        ["Total Points", f"{total_points}"],
        ["Missing Points", f"{missing_points} ({missing_pct:.1f}%)"],
        ["Longest Gap", f"{longest_gap} points"],
        ["Worst Month", f"{worst_month} ({worst_month_pct:.1f}%)"],
        ["Worst Hour", f"{worst_hour} ({worst_hour_pct:.1f}%)"],
    ]

    # Add the table trace
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="paleturquoise",
                align="left",
                font=dict(size=12),
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color="lavender",
                align="left",
                font=dict(size=11),
            ),
        ),
        row=2,
        col=1,
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
    fig.update_xaxes(title_text="Day of Week", row=2, col=2)

    fig.update_yaxes(title_text="Missing Data (%)", row=1, col=1)
    fig.update_yaxes(title_text="Missing Data (%)", row=1, col=2)
    fig.update_yaxes(title_text="Missing Data (%)", row=2, col=2)

    # Set y-axis range to 0-100% for all percentage plots
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_yaxes(range=[0, 100], row=2, col=2)

    # Update layout
    sensor_name = sensor_data.get("sensor_name", f"Sensor {sensor_id}")
    fig.update_layout(
        title=f"Missing Data Analysis for {sensor_name} (ID: {sensor_id})",
        height=800,
        showlegend=False,
        margin=dict(t=100, b=100, l=50, r=50),
    )

    return fig