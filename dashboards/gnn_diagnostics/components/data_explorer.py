# dashboards/gnn_diagnostics/components/data_explorer.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ..utils.data_utils import get_data_for_sensor


def create_raw_data_explorer(experiment_data, sensor_id=None):
    """
    Create a visualization of raw data for a sensor

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data
    sensor_id : str
        ID of the sensor to visualize

    Returns:
    --------
    plotly.graph_objects.Figure
        Raw data visualization
    """
    if not experiment_data or not sensor_id:
        # Create empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data selected",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    # Get data for this sensor
    sensor_data = get_data_for_sensor(experiment_data, sensor_id)

    # Get sensor name
    sensor_name = sensor_data.get("sensor_name", f"Sensor {sensor_id}")

    # Create a figure with two subplots - raw data and missing data pattern
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Raw Time Series Data", "Missing Data Pattern"],
        vertical_spacing=0.15,
    )

    # Get missing value from config
    missing_value = -999.0
    if "config" in experiment_data and "data" in experiment_data["config"] and "general" in experiment_data["config"]["data"]:
        missing_value = experiment_data["config"]["data"]["general"].get("missing_value", -999.0)

    # Plot raw data if available
    raw_data_available = False

    # First try to get data from raw_data dict
    if "raw_data" in sensor_data:
        raw_data = sensor_data["raw_data"]

        # Convert to series if not already
        if not isinstance(raw_data, pd.Series):
            if hasattr(raw_data, "values") and hasattr(raw_data, "index"):
                raw_data = pd.Series(raw_data.values, index=raw_data.index)
            else:
                # Skip if conversion not possible
                raw_data = None

        if raw_data is not None and not raw_data.empty:
            raw_data_available = True

            # Create a mask for valid values
            valid_mask = raw_data != missing_value
            valid_series = raw_data[valid_mask]
            missing_series = raw_data[~valid_mask]

            # Plot valid data
            if not valid_series.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid_series.index,
                        y=valid_series.values,
                        mode="lines+markers",
                        name="Valid Data",
                        line=dict(color="blue", width=2),
                        marker=dict(size=4, color="blue"),
                    ),
                    row=1,
                    col=1,
                )

            # Plot missing data points
            if not missing_series.empty:
                # Replace missing values with NaN for the plot
                missing_series_plot = missing_series.copy()
                missing_series_plot[missing_series_plot == missing_value] = np.nan

                fig.add_trace(
                    go.Scatter(
                        x=missing_series.index,
                        y=[np.nan] * len(missing_series),
                        mode="markers",
                        name="Missing Data",
                        marker=dict(
                            symbol="x",
                            size=8,
                            color="red",
                            line=dict(width=1, color="darkred"),
                        ),
                    ),
                    row=1,
                    col=1,
                )

            # Create a binary missing data plot
            all_dates = pd.date_range(raw_data.index.min(), raw_data.index.max(), freq="D")
            missing_pattern = np.zeros(len(all_dates))

            for i, date in enumerate(all_dates):
                # Get data for this date
                date_data = raw_data[raw_data.index.date == date.date()]

                if date_data.empty:
                    # No data for this date
                    missing_pattern[i] = 0
                else:
                    # Compute percentage of missing values
                    missing_pct = (date_data == missing_value).mean()
                    missing_pattern[i] = 1 - missing_pct

            # Plot missing data pattern
            fig.add_trace(
                go.Bar(
                    x=all_dates,
                    y=missing_pattern,
                    name="Data Availability",
                    marker_color="blue",
                ),
                row=2,
                col=1,
            )

            # Add a line at 0.5 for reference
            fig.add_shape(
                type="line",
                x0=all_dates[0],
                x1=all_dates[-1],
                y0=0.5,
                y1=0.5,
                line=dict(color="gray", dash="dash"),
                row=2,
                col=1,
            )

            # Add annotations for data stats
            total_points = len(raw_data)
            missing_points = (~valid_mask).sum()
            missing_pct = missing_points / total_points * 100 if total_points > 0 else 0

            stats_text = (
                f"Total Points: {total_points}<br>"
                f"Missing Points: {missing_points} ({missing_pct:.1f}%)<br>"
                f"Date Range: {raw_data.index.min().date()} to {raw_data.index.max().date()}"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                text=stats_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
            )

    # If we still don't have raw data, try validation_series
    if not raw_data_available and "validation_series" in sensor_data:
        validation_series = sensor_data["validation_series"]

        # Convert to series if not already
        if not isinstance(validation_series, pd.Series):
            if hasattr(validation_series, "values") and hasattr(validation_series, "index"):
                validation_series = pd.Series(validation_series.values, index=validation_series.index)
            else:
                # Skip if conversion not possible
                validation_series = None

        if validation_series is not None and not validation_series.empty:
            raw_data_available = True

            # Create a mask for valid values
            valid_mask = validation_series != missing_value
            valid_series = validation_series[valid_mask]
            missing_series = validation_series[~valid_mask]

            # Plot valid data
            if not valid_series.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid_series.index,
                        y=valid_series.values,
                        mode="lines+markers",
                        name="Valid Data (Validation)",
                        line=dict(color="green", width=2),
                        marker=dict(size=4, color="green"),
                    ),
                    row=1,
                    col=1,
                )

            # Plot missing data points
            if not missing_series.empty:
                # Replace missing values with NaN for the plot
                missing_series_plot = missing_series.copy()
                missing_series_plot[missing_series_plot == missing_value] = np.nan

                fig.add_trace(
                    go.Scatter(
                        x=missing_series.index,
                        y=[np.nan] * len(missing_series),
                        mode="markers",
                        name="Missing Data (Validation)",
                        marker=dict(
                            symbol="x",
                            size=8,
                            color="orange",
                            line=dict(width=1, color="darkorange"),
                        ),
                    ),
                    row=1,
                    col=1,
                )

            # Add annotations for data stats
            total_points = len(validation_series)
            missing_points = (~valid_mask).sum()
            missing_pct = missing_points / total_points * 100 if total_points > 0 else 0

            stats_text = (
                f"Total Points (Validation): {total_points}<br>"
                f"Missing Points: {missing_points} ({missing_pct:.1f}%)<br>"
                f"Date Range: {validation_series.index.min().date()} to {validation_series.index.max().date()}"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.85,
                text=stats_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
            )

    # If no raw data or validation series is available
    if not raw_data_available:
        fig.add_annotation(
            text="No raw data available for this sensor",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Update layout
    fig.update_layout(
        title=f"Raw Data Analysis for {sensor_name} (ID: {sensor_id})",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=100, b=100, l=50, r=50),
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Data Availability (0-1)", row=2, col=1, range=[0, 1])

    return fig