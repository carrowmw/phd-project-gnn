# dashboards/gnn_diagnostics/components/feature_distribution.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_feature_distribution(experiment_data, comparison_data=None):
    """
    Create a visualization of feature distribution before and after preprocessing

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data
    comparison_data : dict, optional
        Dictionary containing comparison experiment data

    Returns:
    --------
    plotly.graph_objects.Figure
        Feature distribution visualization
    """
    # Create a figure with three subplots - raw data, standardized data, and distribution comparison
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Raw Data Distribution",
            "Processed Data Distribution",
            "Histogram Comparison",
            "Sensor Statistics"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Get missing value from config
    missing_value = -999.0
    if "config" in experiment_data and "data" in experiment_data["config"] and "general" in experiment_data["config"]["data"]:
        missing_value = experiment_data["config"]["data"]["general"].get("missing_value", -999.0)

    # Get standardization settings
    standardize = True
    if "config" in experiment_data and "data" in experiment_data["config"] and "general" in experiment_data["config"]["data"]:
        standardize = experiment_data["config"]["data"]["general"].get("standardize", True)

    # Collect raw data values
    raw_values = []

    # Try to get from raw_data dict
    if "raw_data" in experiment_data:
        raw_data = experiment_data["raw_data"]

        # Collect all valid values
        for sensor_id, series in raw_data.items():
            if hasattr(series, "values"):
                # It's a pandas Series
                valid_mask = series.values != missing_value
                valid_values = series.values[valid_mask]
                raw_values.extend(valid_values)

    # Collect validation data values
    validation_values = []

    # Try to get from time_series dict
    if "time_series" in experiment_data and "validation" in experiment_data["time_series"]:
        validation_series = experiment_data["time_series"]["validation"]

        # Collect all valid values
        for sensor_id, series in validation_series.items():
            if hasattr(series, "values"):
                # It's a pandas Series
                valid_mask = series.values != missing_value
                valid_values = series.values[valid_mask]
                validation_values.extend(valid_values)

    # Create distributions
    if raw_values:
        # Plot raw data histogram
        fig.add_trace(
            go.Histogram(
                x=raw_values,
                nbinsx=30,
                name="Raw Values",
                marker_color="blue",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        # Plot raw KDE
        try:
            from scipy.stats import gaussian_kde

            # Calculate the KDE
            raw_kde = gaussian_kde(raw_values)
            x_range = np.linspace(min(raw_values), max(raw_values), 1000)
            y_kde = raw_kde(x_range)

            # Plot the KDE
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode="lines",
                    line=dict(color="darkblue", width=2),
                    name="Raw Data KDE",
                ),
                row=1,
                col=1,
            )
        except:
            # Skip KDE if it fails
            pass

        # Add stats annotation
        raw_mean = np.mean(raw_values)
        raw_std = np.std(raw_values)
        raw_min = np.min(raw_values)
        raw_max = np.max(raw_values)

        raw_stats_text = (
            f"Raw Data Statistics:<br>"
            f"Mean: {raw_mean:.2f}<br>"
            f"Std Dev: {raw_std:.2f}<br>"
            f"Min: {raw_min:.2f}<br>"
            f"Max: {raw_max:.2f}<br>"
            f"Range: {raw_max - raw_min:.2f}<br>"
            f"Total Points: {len(raw_values)}"
        )

        fig.add_annotation(
            xref="x1",
            yref="y1",
            x=raw_max - (raw_max - raw_min) * 0.1,
            y=0.9 * fig.data[0].y.max() if fig.data and hasattr(fig.data[0], 'y') else 0.9,
            text=raw_stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
        )
    else:
        # No raw data
        fig.add_annotation(
            text="No raw data available",
            xref="x1",
            yref="y1",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Plot validation/standardized data
    if validation_values:
        # Plot validation data histogram
        fig.add_trace(
            go.Histogram(
                x=validation_values,
                nbinsx=30,
                name="Processed Values",
                marker_color="green",
                opacity=0.7,
            ),
            row=1,
            col=2,
        )

        # Plot validation KDE
        try:
            from scipy.stats import gaussian_kde

            # Calculate the KDE
            val_kde = gaussian_kde(validation_values)
            x_range = np.linspace(min(validation_values), max(validation_values), 1000)
            y_kde = val_kde(x_range)

            # Plot the KDE
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode="lines",
                    line=dict(color="darkgreen", width=2),
                    name="Processed Data KDE",
                ),
                row=1,
                col=2,
            )
        except:
            # Skip KDE if it fails
            pass

        # Add stats annotation
        val_mean = np.mean(validation_values)
        val_std = np.std(validation_values)
        val_min = np.min(validation_values)
        val_max = np.max(validation_values)

        val_stats_text = (
            f"Processed Data Statistics:<br>"
            f"Mean: {val_mean:.2f}<br>"
            f"Std Dev: {val_std:.2f}<br>"
            f"Min: {val_min:.2f}<br>"
            f"Max: {val_max:.2f}<br>"
            f"Range: {val_max - val_min:.2f}<br>"
            f"Total Points: {len(validation_values)}"
        )

        fig.add_annotation(
            xref="x2",
            yref="y2",
            x=val_max - (val_max - val_min) * 0.1,
            y=0.9 * fig.data[-1].y.max() if fig.data and hasattr(fig.data[-1], 'y') else 0.9,
            text=val_stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
        )
    else:
        # No validation data
        fig.add_annotation(
            text="No processed data available",
            xref="x2",
            yref="y2",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Create a histogram comparison
    if raw_values and validation_values:
        # Normalize both datasets for comparison
        raw_norm = np.array(raw_values)
        val_norm = np.array(validation_values)

        # Plot normalized histograms
        fig.add_trace(
            go.Histogram(
                x=raw_norm,
                nbinsx=30,
                name="Raw Values",
                marker_color="blue",
                opacity=0.5,
                histnorm="probability density",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=val_norm,
                nbinsx=30,
                name="Processed Values",
                marker_color="green",
                opacity=0.5,
                histnorm="probability density",
            ),
            row=2,
            col=1,
        )

        # Add annotation for standardization
        if standardize:
            # Add text about standardization
            standardization_text = (
                f"Data is standardized in this experiment.<br>"
                f"Expected processed mean ≈ 0 and std dev ≈ 1"
            )

            fig.add_annotation(
                xref="x3",
                yref="y3",
                x=0.5,
                y=0.9,
                text=standardization_text,
                showarrow=False,
                align="center",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
            )
    else:
        # Can't create comparison
        fig.add_annotation(
            text="Cannot create comparison - missing data",
            xref="x3",
            yref="y3",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Create a sensor statistics table
    if "predictions_df" in experiment_data:
        predictions_df = experiment_data["predictions_df"]

        # Group by sensor and calculate statistics
        if "node_id" in predictions_df.columns:
            sensor_stats = []

            for sensor_id in predictions_df["node_id"].unique():
                sensor_data = predictions_df[predictions_df["node_id"] == sensor_id]

                # Create mask for valid values
                valid_mask = (sensor_data["prediction"] != missing_value) & (sensor_data["actual"] != missing_value)
                valid_data = sensor_data[valid_mask]

                # Calculate error metrics if we have valid data
                if not valid_data.empty:
                    sensor_name = sensor_data["sensor_name"].iloc[0] if "sensor_name" in sensor_data.columns else f"Sensor {sensor_id}"
                    valid_count = len(valid_data)
                    total_count = len(sensor_data)
                    valid_pct = valid_count / total_count * 100 if total_count > 0 else 0

                    # Calculate errors
                    if "error" not in valid_data.columns:
                        valid_data["error"] = valid_data["prediction"] - valid_data["actual"]

                    mse = (valid_data["error"] ** 2).mean()
                    mae = abs(valid_data["error"]).mean()

                    sensor_stats.append({
                        "sensor_id": sensor_id,
                        "sensor_name": sensor_name,
                        "valid_count": valid_count,
                        "total_count": total_count,
                        "valid_pct": valid_pct,
                        "mse": mse,
                        "mae": mae,
                    })

            # Sort by MSE
            sensor_stats.sort(key=lambda x: x["mse"], reverse=True)

            # Create a table with sensor statistics
            if sensor_stats:
                # Take only top 10 sensors
                top_sensors = sensor_stats[:10]

                sensor_names = [s["sensor_name"] for s in top_sensors]
                mse_values = [s["mse"] for s in top_sensors]
                mae_values = [s["mae"] for s in top_sensors]
                valid_pcts = [s["valid_pct"] for s in top_sensors]

                # Create a table trace
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=["Sensor", "MSE", "MAE", "Valid %"],
                            fill_color="paleturquoise",
                            align="left",
                            font=dict(size=12),
                        ),
                        cells=dict(
                            values=[
                                sensor_names,
                                [f"{mse:.6f}" for mse in mse_values],
                                [f"{mae:.6f}" for mae in mae_values],
                                [f"{pct:.1f}%" for pct in valid_pcts],
                            ],
                            fill_color="lavender",
                            align="left",
                            font=dict(size=11),
                        ),
                    ),
                    row=2,
                    col=2,
                )

                # Add title for the table
                fig.update_layout(
                    annotations=[{
                        "text": "Top 10 Sensors by MSE",
                        "x": 0.75,
                        "y": 0.31,
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 14},
                    }] + list(fig.layout.annotations)
                )
            else:
                # No sensor statistics
                fig.add_annotation(
                    text="No sensor statistics available",
                    xref="x4",
                    yref="y4",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
    else:
        # No predictions data
        fig.add_annotation(
            text="No predictions data available",
            xref="x4",
            yref="y4",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Update layout
    fig.update_layout(
        title="Feature Distribution Analysis",
        height=800,
        barmode="overlay",
        margin=dict(t=100, b=0, l=50, r=50),
    )

    # Update axes
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Value", row=2, col=1)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=2, col=1)

    return fig