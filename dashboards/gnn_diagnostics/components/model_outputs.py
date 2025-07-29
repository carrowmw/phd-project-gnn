# dashboards/gnn_diagnostics/components/model_outputs.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils.data_utils import get_data_for_sensor


def create_prediction_comparison(experiment_data, comparison_data=None, sensor_id=None):
    """
    Create a visualization comparing predictions to ground truth

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data
    comparison_data : dict, optional
        Dictionary containing comparison experiment data
    sensor_id : str
        ID of the sensor to visualize

    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization of predictions vs ground truth
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

    # Create a figure
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Prediction vs Ground Truth", "Prediction Error"],
        vertical_spacing=0.15,
    )

    # Get missing value from config
    missing_value = -999.0
    if "config" in experiment_data and "data" in experiment_data["config"] and "general" in experiment_data["config"]["data"]:
        missing_value = experiment_data["config"]["data"]["general"].get("missing_value", -999.0)

    # Plot predictions if available
    if "predictions" in sensor_data:
        predictions_df = sensor_data["predictions"]

        # Convert timestamp to datetime if it's a string
        if "timestamp" in predictions_df.columns and isinstance(predictions_df["timestamp"].iloc[0], str):
            predictions_df["timestamp"] = pd.to_datetime(predictions_df["timestamp"])

        # Create a mask for valid values
        valid_mask = (predictions_df["prediction"] != missing_value) & (predictions_df["actual"] != missing_value)
        valid_df = predictions_df[valid_mask]

        # Get sensor name
        sensor_name = sensor_data.get("sensor_name", f"Sensor {sensor_id}")

        if not valid_df.empty:
            # Plot predicted values
            fig.add_trace(
                go.Scatter(
                    x=valid_df["timestamp"],
                    y=valid_df["prediction"],
                    mode="lines+markers",
                    name=f"{experiment_data['experiment_name']} Prediction",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6, color="blue"),
                ),
                row=1,
                col=1,
            )

            # Plot actual values
            fig.add_trace(
                go.Scatter(
                    x=valid_df["timestamp"],
                    y=valid_df["actual"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="green", width=2),
                    marker=dict(size=6, color="green"),
                ),
                row=1,
                col=1,
            )

            # Plot error
            valid_df["error"] = valid_df["prediction"] - valid_df["actual"]

            fig.add_trace(
                go.Scatter(
                    x=valid_df["timestamp"],
                    y=valid_df["error"],
                    mode="lines+markers",
                    name="Error",
                    line=dict(color="red", width=2),
                    marker=dict(size=6, color="red"),
                ),
                row=2,
                col=1,
            )

            # Add horizontal line at zero for error plot
            fig.add_shape(
                type="line",
                x0=valid_df["timestamp"].min(),
                x1=valid_df["timestamp"].max(),
                y0=0,
                y1=0,
                line=dict(color="black", dash="dash"),
                row=2,
                col=1,
            )

            # Add comparison if available
            if comparison_data and sensor_id:
                comparison_sensor_data = get_data_for_sensor(comparison_data, sensor_id)

                if "predictions" in comparison_sensor_data:
                    comp_pred_df = comparison_sensor_data["predictions"]

                    # Convert timestamp to datetime if it's a string
                    if "timestamp" in comp_pred_df.columns and isinstance(comp_pred_df["timestamp"].iloc[0], str):
                        comp_pred_df["timestamp"] = pd.to_datetime(comp_pred_df["timestamp"])

                    # Create a mask for valid values
                    comp_valid_mask = (comp_pred_df["prediction"] != missing_value) & (comp_pred_df["actual"] != missing_value)
                    comp_valid_df = comp_pred_df[comp_valid_mask]

                    if not comp_valid_df.empty:
                        # Plot comparison predicted values
                        fig.add_trace(
                            go.Scatter(
                                x=comp_valid_df["timestamp"],
                                y=comp_valid_df["prediction"],
                                mode="lines+markers",
                                name=f"{comparison_data['experiment_name']} Prediction",
                                line=dict(color="orange", width=2),
                                marker=dict(size=6, color="orange"),
                            ),
                            row=1,
                            col=1,
                        )

                        # Plot comparison error
                        comp_valid_df["error"] = comp_valid_df["prediction"] - comp_valid_df["actual"]

                        fig.add_trace(
                            go.Scatter(
                                x=comp_valid_df["timestamp"],
                                y=comp_valid_df["error"],
                                mode="lines+markers",
                                name=f"{comparison_data['experiment_name']} Error",
                                line=dict(color="purple", width=2),
                                marker=dict(size=6, color="purple"),
                            ),
                            row=2,
                            col=1,
                        )

            # Compute statistics for the errors
            mse = (valid_df["error"] ** 2).mean()
            mae = valid_df["error"].abs().mean()
            rmse = np.sqrt(mse)

            # Add error statistics as an annotation
            stats_text = (
                f"MSE: {mse:.6f}<br>"
                f"MAE: {mae:.6f}<br>"
                f"RMSE: {rmse:.6f}<br>"
                f"Valid Points: {len(valid_df)}/{len(predictions_df)} ({len(valid_df)/len(predictions_df)*100:.1f}%)"
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
        else:
            # No valid data points
            fig.add_annotation(
                text="No valid data points found for this sensor",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )

            # Check if there are any invalid points and explain why
            if not predictions_df.empty:
                # Count missing actuals and predictions
                missing_actuals = (predictions_df["actual"] == missing_value).sum()
                missing_preds = (predictions_df["prediction"] == missing_value).sum()

                explanation = (
                    f"Found {len(predictions_df)} data points, but all are invalid.<br>"
                    f"Missing actual values: {missing_actuals}<br>"
                    f"Missing predicted values: {missing_preds}"
                )

                fig.add_annotation(
                    text=explanation,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.4,
                    showarrow=False,
                    font=dict(size=12),
                )
    else:
        # No predictions found
        fig.add_annotation(
            text=f"No predictions found for sensor {sensor_id}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Add raw data if available
    if "raw_data" in sensor_data and not sensor_data["raw_data"].empty:
        raw_data = sensor_data["raw_data"]

        # Convert to a pandas Series if it's not already
        if not isinstance(raw_data, pd.Series):
            if hasattr(raw_data, "values"):
                raw_data = pd.Series(raw_data.values, index=raw_data.index)
            else:
                # Try to convert to series
                try:
                    raw_data = pd.Series(raw_data)
                except:
                    # If conversion fails, skip this part
                    pass

        if isinstance(raw_data, pd.Series):
            # Create a mask for valid values
            valid_mask = raw_data != missing_value
            valid_series = raw_data[valid_mask]

            if not valid_series.empty:
                # Plot raw data on the top subplot
                fig.add_trace(
                    go.Scatter(
                        x=valid_series.index,
                        y=valid_series.values,
                        mode="lines",
                        name="Raw Data",
                        line=dict(color="gray", width=1, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

    # Update layout
    fig.update_layout(
        title=f"Prediction Analysis for {sensor_name} (ID: {sensor_id})",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=100, b=100, l=50, r=50),
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Error (Prediction - Actual)", row=2, col=1)

    return fig