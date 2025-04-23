# gnn_package/src/visualization/prediction_plots.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.dates import DateFormatter
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import os
from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map


def plot_predictions_with_validation(
    predictions_dict: Dict[str, Any],
    data: Dict[str, Any],
    node_ids: List[str],
    config: Any,
    max_plots: int = 6,
) -> plt.Figure:
    """
    Plot predictions alongside actual data for validation.

    Parameters:
    -----------
    predictions_dict : dict
        Dictionary returned by predict_with_model
    data : dict
        Dict containing time series data
    node_ids : list
        List of node IDs
    config : ExperimentConfig
        Configuration object
    max_plots : int
        Maximum number of sensors to plot

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Get validation window from config
    validation_window = config.data.general.horizon

    # Get name-to-id mapping
    name_id_map = get_sensor_name_id_map(config=config)
    name_id_map = {v: k for k, v in name_id_map.items()}

    # Get predictions array and node indices
    pred_array = predictions_dict["predictions"]
    node_indices = predictions_dict["node_indices"]
    valid_nodes = [node_ids[idx] for idx in node_indices]

    # Create a figure
    n_nodes = min(len(valid_nodes), max_plots)  # Limit to max_plots nodes
    fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 3 * n_nodes))
    if n_nodes == 1:
        axes = [axes]

    # Get time series dictionary from data
    time_series_dict = data["time_series"]["validation"]

    for i, node_id in enumerate(valid_nodes[:n_nodes]):
        ax = axes[i]

        # Get full historical data
        if node_id not in time_series_dict:
            print(f"No historical data found for node {node_id}")
            continue

        historical = time_series_dict[node_id]

        # Split historical data into 'input' and 'validation' parts
        input_data = historical[:-validation_window]
        validation_data = historical[-validation_window:]

        # Get prediction for this node
        node_position = np.where(node_indices == node_ids.index(node_id))[0]
        if len(node_position) == 0:
            print(f"Cannot find node {node_id} in prediction data")
            continue

        node_idx = node_position[0]
        pred = pred_array[0, node_idx, :, 0]  # [batch=0, node, time, feature=0]

        # Get the last timestamp from input data
        last_input_time = input_data.index[-1]

        # Create time indices for prediction that align with validation data
        pred_times = validation_data.index

        # Plot
        ax.plot(
            input_data.index,
            input_data.values,
            label="Input Data",
            color="blue",
            linewidth=1.5,
        )
        ax.plot(
            validation_data.index,
            validation_data.values,
            label="Actual Values",
            color="green",
            linewidth=2,
        )
        ax.plot(pred_times, pred, "r--", label="Predictions", linewidth=2)

        # Highlight the last input point for visual clarity
        ax.scatter(
            [last_input_time],
            [input_data.values[-1]],
            color="darkblue",
            s=50,
            zorder=5,
            label="Last Input Point",
        )

        # Add sensor name to title if available
        sensor_name = name_id_map.get(node_id, node_id)
        ax.set_title(f"Model Validation: {sensor_name} (ID: {node_id})")
        ax.set_ylabel("Traffic Count")

        # Add a grid for better readability
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add a legend
        ax.legend(loc="best", framealpha=0.9)

        # Format x-axis as time
        ax.tick_params(axis="x", rotation=45)

        # Add a vertical line to separate input data and validation period
        ax.axvline(x=last_input_time, color="gray", linestyle="--", alpha=0.8)

        # Calculate and display metrics if validation data exists
        if len(validation_data) > 0:
            mse = ((pred - validation_data.values) ** 2).mean()
            mae = abs(pred - validation_data.values).mean()
            ax.text(
                0.02,
                0.95,
                f"MSE: {mse:.4f}\nMAE: {mae:.4f}",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
            )

    plt.tight_layout()
    return fig


def plot_sensors_grid(
    predictions_df: pd.DataFrame,
    plots_per_row: int = 5,
    figsize: Tuple[int, int] = (20, 25),
) -> plt.Figure:
    """
    Create a grid of plots showing prediction vs actual values for all sensors.

    Parameters:
    -----------
    predictions_df : pandas DataFrame
        DataFrame containing the prediction results with columns:
        'node_id', 'sensor_name', 'timestamp', 'prediction', 'actual', 'horizon'
    plots_per_row : int
        Number of plots to show in each row
    figsize : tuple
        Size of the overall figure (width, height)

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the grid of plots
    """
    # Get unique sensors
    unique_sensors = predictions_df["node_id"].unique()
    num_sensors = len(unique_sensors)

    # Calculate grid dimensions
    num_rows = int(np.ceil(num_sensors / plots_per_row))

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=figsize)
    axes = axes.flatten()  # Flatten to make indexing easier

    # Set overall title
    fig.suptitle(f"Predictions vs Actual Values for {num_sensors} Sensors", fontsize=16)

    # Format for dates
    date_formatter = DateFormatter("%H:%M")

    # First pass: determine global min and max values for consistent y-axis scaling
    global_min = float("inf")
    global_max = float("-inf")

    for sensor_id in unique_sensors:
        # Get data for this sensor
        sensor_data = predictions_df[predictions_df["node_id"] == sensor_id]

        # Check if we have data
        if len(sensor_data) > 0:
            # Get min and max values for both predictions and actuals
            predictions_min = sensor_data["prediction"].min()
            predictions_max = sensor_data["prediction"].max()
            actuals_min = sensor_data["actual"].min()
            actuals_max = sensor_data["actual"].max()

            # Update global min and max
            global_min = min(global_min, predictions_min)
            global_max = max(global_max, predictions_max)

    # Add a small buffer to the limits (5% padding)
    y_range = global_max - global_min
    global_min = global_min - 0.05 * y_range if y_range > 0 else global_min - 1
    global_max = global_max + 0.05 * y_range if y_range > 0 else global_max + 1

    # Loop through each sensor and create a plot
    for i, sensor_id in enumerate(unique_sensors):
        if i >= len(axes):  # Safety check
            break

        # Get data for this sensor
        sensor_data = predictions_df[predictions_df["node_id"] == sensor_id]

        # Check if we have data
        if len(sensor_data) > 0:
            # Get sensor name
            sensor_name = sensor_data["sensor_name"].iloc[0]

            # Sort by timestamp to ensure correct plot order
            sensor_data = sensor_data.sort_values("timestamp")

            # Get x and y values
            timestamps = sensor_data["timestamp"]
            predictions = sensor_data["prediction"]
            actuals = sensor_data["actual"]

            # Plot
            ax = axes[i]
            ax.plot(timestamps, predictions, "r-", label="Prediction", linewidth=2)
            ax.plot(timestamps, actuals, "b-", label="Actual", linewidth=2)

            # Format plot
            ax.set_title(f"{sensor_name.split('Ncl')[-1]}", fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.xaxis.set_major_formatter(date_formatter)

            # Apply consistent y-axis limits to all plots
            ax.set_ylim(global_min, global_max)

            # Only show legend for the first plot
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

            # Add grid for better readability
            ax.grid(True, linestyle="--", alpha=0.6)

            # Calculate and show error metrics
            mse = ((predictions - actuals) ** 2).mean()
            mae = (predictions - actuals).abs().mean()
            ax.text(
                0.02,
                0.95,
                f"MAE: {mae:.1f}",
                transform=ax.transAxes,
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7),
            )
        else:
            # No data case
            ax.text(
                0.5,
                0.5,
                f"No data for {sensor_id}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle

    return fig


def plot_error_distribution(
    predictions_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create plots showing the error distribution and patterns.

    Parameters:
    -----------
    predictions_df : pandas DataFrame
        DataFrame containing prediction results
    figsize : tuple
        Size of the figure

    Returns:
    --------
    matplotlib.figure.Figure
        The figure with error distribution plots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Error histogram
    axes[0, 0].hist(
        predictions_df["error"], bins=30, color="skyblue", edgecolor="black"
    )
    axes[0, 0].set_title("Prediction Error Distribution")
    axes[0, 0].set_xlabel("Error")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, linestyle="--", alpha=0.7)

    # Plot 2: Error by prediction horizon
    horizon_errors = predictions_df.groupby("horizon")["abs_error"].mean()
    axes[0, 1].bar(horizon_errors.index, horizon_errors.values, color="lightgreen")
    axes[0, 1].set_title("Mean Absolute Error by Prediction Horizon")
    axes[0, 1].set_xlabel("Horizon (steps ahead)")
    axes[0, 1].set_ylabel("Mean Absolute Error")
    axes[0, 1].grid(True, linestyle="--", alpha=0.7)

    # Plot 3: Scatter plot of predicted vs actual
    axes[1, 0].scatter(
        predictions_df["actual"],
        predictions_df["prediction"],
        alpha=0.5,
        s=10,
        color="blue",
    )

    # Add perfect prediction line
    max_val = max(predictions_df["actual"].max(), predictions_df["prediction"].max())
    min_val = min(predictions_df["actual"].min(), predictions_df["prediction"].min())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--")

    axes[1, 0].set_title("Predicted vs Actual Values")
    axes[1, 0].set_xlabel("Actual Values")
    axes[1, 0].set_ylabel("Predicted Values")
    axes[1, 0].grid(True, linestyle="--", alpha=0.7)

    # Plot 4: Top 10 sensors by error
    sensor_errors = (
        predictions_df.groupby(["sensor_name"])["abs_error"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    sensor_errors.plot(kind="barh", ax=axes[1, 1], color="salmon")
    axes[1, 1].set_title("Top 10 Sensors by Mean Absolute Error")
    axes[1, 1].set_xlabel("Mean Absolute Error")
    axes[1, 1].set_ylabel("Sensor Name")

    plt.tight_layout()
    return fig


def save_visualization_pack(
    predictions_df: pd.DataFrame,
    results_dict: Dict[str, Any],
    output_dir: Union[str, Path],
    timestamp: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate and save a comprehensive set of visualizations.

    Parameters:
    -----------
    predictions_df : pandas DataFrame
        DataFrame with prediction results
    results_dict : dict
        Dictionary with prediction results and data
    output_dir : str or Path
        Directory to save visualizations
    timestamp : str, optional
        Timestamp string for filenames

    Returns:
    --------
    dict
        Dictionary with paths to all saved visualizations
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Use current timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dictionary to store all visualization paths
    viz_paths = {}

    # 1. Create and save sensors grid plot
    try:
        grid_fig = plot_sensors_grid(predictions_df)
        grid_path = output_dir / f"sensors_grid_{timestamp}.png"
        grid_fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(grid_fig)
        viz_paths["grid_plot"] = str(grid_path)
    except Exception as e:
        print(f"Error creating sensors grid plot: {e}")

    # 2. Create and save error distribution plots
    try:
        error_fig = plot_error_distribution(predictions_df)
        error_path = output_dir / f"error_analysis_{timestamp}.png"
        error_fig.savefig(error_path, dpi=150, bbox_inches="tight")
        plt.close(error_fig)
        viz_paths["error_analysis"] = str(error_path)
    except Exception as e:
        print(f"Error creating error distribution plot: {e}")

    # 3. Create and save detailed validation plot
    try:
        if all(k in results_dict for k in ["predictions", "data", "node_ids"]):
            validation_fig = plot_predictions_with_validation(
                results_dict["predictions"],
                results_dict["data"],
                results_dict["node_ids"],
                results_dict.get("config"),
            )
            validation_path = output_dir / f"validation_plot_{timestamp}.png"
            validation_fig.savefig(validation_path, dpi=150, bbox_inches="tight")
            plt.close(validation_fig)
            viz_paths["validation_plot"] = str(validation_path)
    except Exception as e:
        print(f"Error creating validation plot: {e}")

    return viz_paths
