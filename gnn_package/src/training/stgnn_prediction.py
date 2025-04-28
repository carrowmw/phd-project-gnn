# Updated src/training/stgnn_prediction.py
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map
from gnn_package.src.utils.model_io import load_model as load_model_io
from gnn_package.src.utils.metrics import (
    calculate_error_metrics,
    format_prediction_results,
    calculate_metrics_by_horizon,
)
from gnn_package.config import create_prediction_config
from gnn_package.src.data.processors import DataProcessorFactory, ProcessorMode
from gnn_package.src.data.data_sources import APIDataSource
from gnn_package.src.utils.data_utils import validate_data_package

logger = logging.getLogger(__name__)


async def fetch_recent_data_for_validation(config):
    """
    Fetch recent data for validation using the prediction data processor.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration object containing all parameters

    Returns:
    --------
    dict
        Dictionary containing processed data for validation
    """
    try:
        # Create an API data source for the real-time data
        data_source = APIDataSource()

        # Create processor explicitly for prediction mode
        processor = DataProcessorFactory.create_processor(
            mode=ProcessorMode.PREDICTION,
            config=config,
            data_source=data_source,
        )

        # Process data using prediction-specific logic
        result = await processor.process_data()

        if result is None:
            raise ValueError("Data processing returned no results. Check logs for details.")

        return result
    except Exception as e:
        logger.error(f"Error fetching or processing data: {str(e)}")
        raise


def predict_with_model(model, data_package, config):
    """
    Make predictions using a trained model and a dataloader.

    Parameters:
    -----------
    model : STGNN
        The trained model
    data_package : dict
        Complete data package containing data loaders, graph data, and metadata
    config : ExperimentConfig
        Configuration object

    Returns:
    --------
    dict
        Dictionary containing predictions and metadata
    """
    # Validate data package
    validate_data_package(
        data_package, required_components=["val_loader"], mode="prediction"
    )

    # Extract components for use
    val_loader = data_package["data_loaders"]["val_loader"]

    # Get device
    from gnn_package.src.utils.device_utils import get_device_from_config

    device = get_device_from_config(config)

    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Get a single batch from the dataloader
    batch = next(iter(val_loader))

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
    predictions_dict, data_package, node_ids, config
):
    """
    Format model predictions into a pandas DataFrame and include actual values for comparison.

    Parameters:
    -----------
    predictions_dict : dict
        Dictionary returned by predict_with_model
    data_package : dict
        Complete data package containing time series data
    node_ids : list
        List of node IDs in the order they appear in the predictions
    config : ExperimentConfig
        Configuration object

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing formatted predictions and actual values
    """
    # Validate data package
    validate_data_package(data_package, required_components=["time_series"])

    # Extract time series data
    time_series_dict = data_package["time_series"]["validation"]

    # Get prediction array and node indices
    predictions = predictions_dict["predictions"]
    node_indices = predictions_dict["node_indices"]

    # Get window size and horizon from config
    window_size = config.data.general.window_size
    horizon = config.data.general.horizon

    # Get name-to-ID mapping for sensor names
    name_id_map = get_sensor_name_id_map(config=config)
    id_to_name_map = {v: k for k, v in name_id_map.items()}

    # Use the centralized formatting function
    return format_prediction_results(
        predictions=predictions,
        time_series_dict=time_series_dict,
        node_ids=node_ids,
        node_indices=node_indices,
        window_size=window_size,
        horizon=horizon,
        id_to_name_map=id_to_name_map,
    )


async def predict_all_sensors_with_validation(
    model_path, config=None, output_file=None, plot=True
):
    """
    Make predictions for all available sensors and validate against actual data.

    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model file
    config : ExperimentConfig, optional
        Configuration object. If None, attempts to find config in model directory
        or falls back to global config.
    output_file : str, optional
        Path to save the prediction results
    plot : bool
        Whether to create and show validation plots

    Returns:
    --------
    dict
        Dictionary containing predictions, actual values, and evaluation metrics
    """
    # Load model with appropriate configuration
    prediction_config = create_prediction_config()

    # Use the centralized model loading function
    model, _ = load_model_io(
        model_path=model_path, model_type="stgnn", config=prediction_config
    )

    logger.info(f"Model loaded successfully from: {model_path}")

    # Fetch and preprocess recent data for validation
    logger.info(f"Fetching and preprocessing recent data for validation")
    data_package = await fetch_recent_data_for_validation(config=prediction_config)

    # Extract components from the standardized structure
    node_ids = data_package["graph_data"]["node_ids"]
    time_series = data_package["time_series"]["validation"]

    logger.info(f"Preprocessed data for {len(node_ids)} nodes")

    # Make predictions using the validation dataloader
    predictions_dict = predict_with_model(model, data_package, config=prediction_config)

    # Format results using the centralized function
    results_df = format_predictions_with_validation(
        predictions_dict=predictions_dict,
        data_package=data_package,
        node_ids=node_ids,
        config=prediction_config,
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
            predictions_dict, plot_data, node_ids, config=prediction_config
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

    # Calculate overall metrics
    metrics = {}
    if not results_df.empty and "error" in results_df.columns:
        metrics = {
            "mse": (results_df["error"] ** 2).mean(),
            "mae": results_df["abs_error"].mean(),
        }
        logger.info(f"Overall MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")

    return {
        "predictions": predictions_dict,
        "dataframe": results_df,
        "data": data_package,
        "node_ids": node_ids,
        "metrics": metrics,
    }
