# gnn_package/src/models/prediction.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from pathlib import Path

from private_uoapi import (
    LSConfig,
    LSAuth,
    LightsailWrapper,
    DateRangeParams,
    convert_to_dataframe,
)

from gnn_package.src import training
from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map
from gnn_package.src.models.stgnn import create_stgnn_model
from gnn_package.config import ExperimentConfig, get_config

logger = logging.getLogger(__name__)


def load_model(model_path, config):
    """
    Load a trained STGNN model using parameters from config.

    Parameters:
    -----------
    model_path : str
        Path to the saved model
    config : ExperimentConfig
        Configuration object to use for model parameters

    Returns:
    --------
    Loaded model
    """
    logger.info(f"Loading model using configuration from {config.config_path}")

    # Create model with parameters from config
    model = create_stgnn_model(config=config)

    # Load state dict and set to eval mode
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Log model configuration
    logger.info(f"Loaded model with parameters:")
    logger.info(f"  hidden_dim: {config.model.hidden_dim}")
    logger.info(f"  num_layers: {config.model.num_layers}")
    logger.info(f"  num_gc_layers: {getattr(config.model, 'num_gc_layers', 2)}")
    logger.info(f"  horizon: {config.data.horizon}")

    return model


async def fetch_recent_data_for_validation(config):
    """
    Fetch recent data for validation using the configuration.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration object containing all parameters

    Returns:
    --------
    dict
        Dictionary containing processed data for validation
    """
    logger.info(f"Using configuration from {config.config_path}")

    # Initialize API client
    api_config = LSConfig()
    auth = LSAuth(api_config)
    client = LightsailWrapper(api_config, auth)

    # Get sensor name to ID mapping
    name_id_map = get_sensor_name_id_map(config=config)
    id_to_name_map = {v: k for k, v in name_id_map.items()}

    # Use all available node IDs by default
    node_ids = list(name_id_map.values())
    logger.info(f"Using all available {len(node_ids)} nodes")

    # Get parameters from config
    days_back = getattr(config.data, "days_back", 2)
    window_size = config.data.window_size
    horizon = config.data.horizon

    logger.info(
        f"Fetching data for {days_back} days with window_size={window_size}, horizon={horizon}"
    )

    # Determine date range for API request
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Create date range parameters
    date_range_params = DateRangeParams(
        start_date=start_date,
        end_date=end_date,
        max_date_range=timedelta(days=days_back + 1),
    )

    # Fetch data from API
    logger.info(f"Querying API for data from {start_date} to {end_date}")
    count_data = await client.get_traffic_data(date_range_params)
    counts_df = convert_to_dataframe(count_data)

    # Create time series dictionary
    time_series_dict = {}
    successful_nodes = 0

    for node_id in node_ids:
        # Look up location name for this node ID
        location = id_to_name_map.get(node_id)
        if not location:
            logger.warning(f"No location found for node ID {node_id}")
            continue

        # Filter data for this location
        df = counts_df[counts_df["location"] == location]

        if df.empty:
            logger.warning(f"No data found for node {node_id} (location: {location})")
            continue

        # Create time series
        series = pd.Series(df["value"].values, index=df["dt"])

        # Remove duplicates
        series = series[~series.index.duplicated(keep="first")]

        # Store in dictionary
        time_series_dict[node_id] = series
        successful_nodes += 1

    logger.info(
        f"Successfully fetched data for {successful_nodes}/{len(node_ids)} nodes"
    )

    # If no data was fetched, return early
    if not time_series_dict:
        logger.warning("No valid data fetched from API")
        return {
            "windows": {},
            "masks": {},
            "time_series": {},
        }

    # For each sensor, hold out the last 'horizon' points for validation
    validation_dict = {}
    input_dict = {}

    for node_id, series in time_series_dict.items():
        if len(series) > horizon:
            # Keep full series for validation purposes
            validation_dict[node_id] = series
            # Use shortened series for prediction input
            input_dict[node_id] = series[:-horizon]
        else:
            logger.warning(
                f"Not enough data points for node {node_id}, needs at least {horizon+1} points"
            )
            # Still include it but with the same data, may not be able to validate effectively
            validation_dict[node_id] = series
            input_dict[node_id] = series

    # Process data using the config
    input_data_loaders = training.preprocess_data(
        data=input_dict,
        config=config,
    )

    # Return a dict with the processed data
    return {
        "data_loaders": input_data_loaders,
        "time_series": validation_dict,
        "input_series": input_dict,
    }


def plot_predictions_with_validation(predictions_dict, data, node_ids, config):
    """
    Plot predictions alongside actual data for validation

    Parameters:
    -----------
    predictions_dict : dict
        Dictionary returned by predict_with_model
    data : dict
        Dict containing time series data from fetch_recent_data
    node_ids : list
        List of node IDs
    config : ExperimentConfig
        Configuration object

    Returns:
    --------
    matplotlib figure
    """
    # Get validation window from config
    validation_window = config.data.horizon

    # Get name-to-id mapping
    name_id_map = get_sensor_name_id_map(config=config)
    name_id_map = {v: k for k, v in name_id_map.items()}

    # Get predictions array and node indices
    pred_array = predictions_dict["predictions"]
    node_indices = predictions_dict["node_indices"]
    valid_nodes = [node_ids[idx] for idx in node_indices]

    # Create a figure
    n_nodes = min(len(valid_nodes), 6)  # Limit to 6 nodes
    fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 3 * n_nodes))
    if n_nodes == 1:
        axes = [axes]

    # Get time series dictionary from data
    time_series_dict = data["time_series"]

    for i, node_id in enumerate(valid_nodes[:n_nodes]):
        ax = axes[i]

        # Get full historical data
        if node_id not in time_series_dict:
            logger.warning(f"No historical data found for node {node_id}")
            continue

        historical = time_series_dict[node_id]

        # Split historical data into 'input' and 'validation' parts
        input_data = historical[:-validation_window]
        validation_data = historical[-validation_window:]

        # Get prediction for this node
        node_position = np.where(node_indices == node_ids.index(node_id))[0]
        if len(node_position) == 0:
            logger.warning(f"Cannot find node {node_id} in prediction data")
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


async def predict_all_sensors_with_validation(
    model_path, config, output_file=None, plot=True
):
    """
    Make predictions for all available sensors and validate against actual data

    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    config : ExperimentConfig
        Configuration object
    output_file : str, optional
        Path to save the prediction results
    plot : bool
        Whether to create and show validation plots

    Returns:
    --------
    dict
        Dictionary containing predictions, actual values, and evaluation metrics
    """
    logger.info(f"Using configuration from {config.config_path}")

    # Load model with the config
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path=model_path, config=config)
    logger.info(f"Model loaded successfully")

    # Fetch and preprocess recent data for validation
    logger.info(f"Fetching and preprocessing recent data for validation")
    data = await fetch_recent_data_for_validation(config=config)

    # Extract preprocessed data
    data_loaders = data["data_loaders"]
    time_series = data["time_series"]  # Full time series

    # The adjacency matrix and node_ids are consistent with the training data
    adj_matrix = data_loaders["adj_matrix"]
    node_ids = data_loaders["node_ids"]

    logger.info(f"Preprocessed data for {len(node_ids)} nodes")

    # Make predictions using the validation dataloader
    predictions = predict_with_model(model, data_loaders["val_loader"], config=config)

    # Format results
    results_df = format_predictions_with_validation(
        predictions, time_series, node_ids, config=config
    )

    # Save to file if requested
    if output_file and not results_df.empty:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")

    # Plot if requested
    if plot and not results_df.empty:
        # Create a data structure compatible with plot_predictions
        plot_data = {"time_series": time_series}
        fig = plot_predictions_with_validation(
            predictions, plot_data, node_ids, config=config
        )
        plt.show()

        # Save the plot if output file is specified
        if output_file:
            plot_filename = (
                output_file.replace(".csv", ".png")
                if output_file.endswith(".csv")
                else f"{output_file}_plot.png"
            )
            fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {plot_filename}")

    return {
        "predictions": predictions,
        "dataframe": results_df,
        "data": data,
        "node_ids": node_ids,
    }


def predict_with_model(model, dataloader, config):
    """
    Make predictions using a trained model and a dataloader.

    Parameters:
    -----------
    model : STGNN
        The trained model
    dataloader : DataLoader
        DataLoader containing the data to predict on
    config : ExperimentConfig
        Configuration object

    Returns:
    --------
    dict
        Dictionary containing predictions and metadata
    """
    # Get device from config if specified, otherwise use best available
    device_name = getattr(config.training, "device", None)
    if device_name:
        device = torch.device(device_name)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Get a single batch from the dataloader
    batch = next(iter(dataloader))

    # Move data to device
    x = batch["x"].to(device)
    x_mask = batch["x_mask"].to(device)
    adj = batch["adj"].to(device)

    logger.info(
        f"Input shapes - x: {x.shape}, x_mask: {x_mask.shape}, adj: {adj.shape}"
    )

    # Make prediction
    with torch.no_grad():
        predictions = model(x, adj, x_mask)

    # Convert to numpy for easier handling
    predictions_np = predictions.cpu().numpy()

    return {
        "predictions": predictions_np,
        "input_data": {
            "x": x.cpu().numpy(),
            "x_mask": x_mask.cpu().numpy(),
        },
        "node_indices": batch["node_indices"].numpy(),
    }


def format_predictions_with_validation(
    predictions_dict, time_series_dict, node_ids, config
):
    """
    Format model predictions into a pandas DataFrame and include actual values for comparison.

    Parameters:
    -----------
    predictions_dict : dict
        Dictionary returned by predict_with_model
    time_series_dict : dict
        Dictionary mapping node IDs to their original time series data
    node_ids : list
        List of node IDs in the order they appear in the predictions
    config : ExperimentConfig
        Configuration object

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing formatted predictions and actual values
    """
    # Get prediction array and node indices
    predictions = predictions_dict["predictions"]
    node_indices = predictions_dict["node_indices"]
    horizon = config.data.horizon

    # Get name-to-ID mapping for sensor names
    name_id_map = get_sensor_name_id_map(config=config)
    id_to_name_map = {v: k for k, v in name_id_map.items()}

    # Create rows for the DataFrame
    rows = []

    # For each node that has predictions
    for i, node_idx in enumerate(node_indices):
        node_id = node_ids[node_idx]

        # Skip if this node doesn't have time series data
        if node_id not in time_series_dict:
            continue

        # Get the time series for this node
        series = time_series_dict[node_id]

        # Skip if not enough data
        if len(series) <= horizon:
            logger.warning(f"Not enough data for node {node_id} to validate")
            continue

        # Get the validation data
        validation_data = series.iloc[-horizon:]

        # Extract predictions for this node
        node_preds = predictions[0, i, :, 0]

        # Create a row for each prediction horizon
        for h, pred_value in enumerate(node_preds):
            if h < len(validation_data):
                # Get the corresponding actual value and timestamp
                actual_time = validation_data.index[h]
                actual_value = validation_data.iloc[h]

                # Create the row
                row = {
                    "node_id": node_id,
                    "sensor_name": id_to_name_map.get(node_id, "Unknown"),
                    "timestamp": actual_time,
                    "prediction": float(pred_value),
                    "actual": float(actual_value),
                    "error": float(pred_value - actual_value),
                    "abs_error": float(abs(pred_value - actual_value)),
                    "horizon": h + 1,  # 1-based horizon index
                }
                rows.append(row)

    # Create DataFrame
    if rows:
        df = pd.DataFrame(rows)
        # Calculate overall metrics
        mse = (df["error"] ** 2).mean()
        mae = df["abs_error"].mean()
        logger.info(f"Overall MSE: {mse:.4f}, MAE: {mae:.4f}")
        return df
    else:
        logger.warning("No predictions could be validated against actual data")
        return pd.DataFrame(
            columns=[
                "node_id",
                "sensor_name",
                "timestamp",
                "prediction",
                "actual",
                "error",
                "abs_error",
                "horizon",
            ]
        )
