import plotly.graph_objects as go
import pandas as pd
import numpy as np
import calendar
from datetime import datetime


def create_monthly_coverage_matrix(time_series_dict, n_sensors=20):
    """
    Create a matrix showing monthly data coverage for top sensors.

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping sensor IDs to their time series data
    n_sensors : int
        Number of top sensors to include

    Returns:
    --------
    plotly.graph_objects.Figure
        Monthly coverage matrix figure
    """
    # Find sensors with most data points
    sensor_data_counts = {
        sensor_id: len(series) for sensor_id, series in time_series_dict.items()
    }
    top_sensors = sorted(sensor_data_counts, key=sensor_data_counts.get, reverse=True)[
        :n_sensors
    ]

    # Get the time range
    all_dates = []
    for series in time_series_dict.values():
        if len(series) > 0:
            all_dates.extend(series.index.tolist())

    if not all_dates:
        return go.Figure().update_layout(title="No data available")

    min_date = min(all_dates).date()
    max_date = max(all_dates).date()

    # Create month-year list
    unique_months = set()
    current_date = pd.Timestamp(min_date)
    while current_date <= pd.Timestamp(max_date):
        unique_months.add((current_date.year, current_date.month))
        current_date += pd.DateOffset(months=1)

    month_labels = [f"{calendar.month_name[m]} {y}" for y, m in sorted(unique_months)]
    month_keys = sorted(unique_months)

    # Expected readings per day and per month
    readings_per_day = 96  # 15-minute intervals

    # Initialize coverage matrix
    coverage_matrix = np.zeros((len(top_sensors), len(month_keys)))

    # Calculate coverage for each sensor and month
    for i, sensor_id in enumerate(top_sensors):
        series = time_series_dict[sensor_id]

        # Group by year and month
        ym_counts = series.groupby([series.index.year, series.index.month]).size()

        # Calculate days in each month-year for expected total readings
        for j, (year, month) in enumerate(month_keys):
            # Calculate days in this month
            if (year, month) in ym_counts.index:
                # Get number of days in this month
                days_in_month = calendar.monthrange(year, month)[1]
                expected_readings = days_in_month * readings_per_day

                # Calculate coverage percentage
                coverage_matrix[i, j] = min(
                    100, ym_counts[(year, month)] / expected_readings * 100
                )

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=coverage_matrix,
            x=month_labels,
            y=[f"Sensor {sensor_id}" for sensor_id in top_sensors],
            colorscale=[
                [0, "rgb(247, 247, 247)"],  # Very light gray for 0%
                [0.2, "rgb(224, 224, 255)"],  # Very light blue for 20%
                [0.5, "rgb(150, 150, 255)"],  # Light blue for 50%
                [0.8, "rgb(67, 67, 255)"],  # Medium blue for 80%
                [1, "rgb(0, 0, 180)"],  # Dark blue for 100%
            ],
            zmin=0,
            zmax=100,
            colorbar=dict(
                title="Coverage %",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0%", "25%", "50%", "75%", "100%"],
            ),
            hovertemplate="Sensor: %{y}<br>Month: %{x}<br>Coverage: %{z:.1f}%<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title="Monthly Data Coverage Matrix (Top Sensors)",
        xaxis_title="Month",
        yaxis_title="Sensor",
        height=600,
        margin=dict(l=150),  # More space for sensor labels
        xaxis=dict(tickangle=45, type="category"),
    )

    return fig
