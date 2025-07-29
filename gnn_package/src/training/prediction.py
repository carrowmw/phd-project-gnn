# src/training/prediction.py
import os
import logging
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime

from gnn_package.config import ExperimentConfig, get_config
from gnn_package.src.models.factory import create_model
from gnn_package.src.utils.model_io import load_model
from gnn_package.src.utils.metrics import (
    calculate_error_metrics,
    format_prediction_results,
    calculate_metrics_by_horizon,
    calculate_metrics_by_sensor,
)

logger = logging.getLogger(__name__)

async def fetch_data_for_prediction(config=None):
    """Fetch data for prediction, with appropriate processing."""
    from gnn_package.src.data.processors import DataProcessorFactory, ProcessorMode
    from gnn_package.src.data.data_sources import APIDataSource

    if config is None:
        config = get_config()

    # Create processor for prediction mode
    data_source = APIDataSource()
    processor = DataProcessorFactory.create_processor(
        mode=ProcessorMode.PREDICTION,
        config=config,
        data_source=data_source,
    )

    # Process the data
    data_package = await processor.process_data()

    return data_package

def predict_with_model(model, data_package, config=None):
    """
    Generate predictions using a model and formatted data package.

    Parameters:
    -----------
    model : nn.Module
        Trained model
    data_package : Dict[str, Any]
        Data package containing validation loader
    config : ExperimentConfig, optional
        Configuration object

    Returns:
    --------
    Dict[str, Any]
        Prediction results
    """
    if config is None:
        config = get_config()

    # Get device
    from gnn_package.src.utils.device_utils import get_device_from_config
    device = get_device_from_config(config)

    # Validate data package
    if "data_loaders" not in data_package or "val_loader" not in data_package["data_loaders"]:
        raise ValueError("Data package must contain val_loader")

    val_loader = data_package["data_loaders"]["val_loader"]

    # Prepare model for prediction
    model.to(device)
    model.eval()

    # Get a batch from the dataloader
    batch = next(iter(val_loader))

    # Move data to device
    x = batch["x"].to(device)
    x_mask = batch["x_mask"].to(device)
    adj = batch["adj"].to(device)

    # Make prediction
    with torch.no_grad():
        predictions = model(x, adj, x_mask)

    # Convert to numpy
    predictions_np = predictions.cpu().numpy()

    return {
        "predictions": predictions_np,
        "input_data": {
            "x": x.cpu().numpy(),
            "x_mask": x_mask.cpu().numpy(),
        },
        "node_indices": batch["node_indices"].numpy(),
    }

def format_predictions(predictions_dict, data_package, config=None):
    """Format prediction results into a DataFrame with actual values."""
    if config is None:
        config = get_config()

    # Extract components
    time_series_dict = data_package["time_series"]["validation"]
    node_ids = data_package["graph_data"]["node_ids"]
    node_indices = predictions_dict["node_indices"]
    predictions = predictions_dict["predictions"]

    # Get missing value from config
    missing_value = config.data.general.missing_value

    # Create sensor name mapping if possible
    id_to_name_map = None
    try:
        from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map
        name_id_map = get_sensor_name_id_map(config=config)
        id_to_name_map = {v: k for k, v in name_id_map.items()}
    except Exception as e:
        logger.warning(f"Could not load sensor name mapping: {e}")

    results_df = format_prediction_results(
        predictions=predictions,
        time_series_dict=time_series_dict,
        node_ids=node_ids,
        node_indices=node_indices,
        window_size=config.data.general.window_size,
        horizon=config.data.general.horizon,
        id_to_name_map=id_to_name_map,
        missing_value=missing_value
    )

    return results_df

async def predict_and_evaluate(
    model_path,
    output_dir=None,
    config=None,
    visualize=True,
):
    """
    Run prediction and evaluation with a saved model.

    Parameters:
    -----------
    model_path : str or Path
        Path to saved model
    output_dir : str or Path, optional
        Directory to save outputs
    config : ExperimentConfig, optional
        Configuration object
    visualize : bool
        Whether to create visualizations

    Returns:
    --------
    Dict[str, Any]
        Prediction results and metrics
    """
    # Setup
    if config is None:
        config = get_config(is_prediction_mode=True)

    # Get missing value from config for calculations
    missing_value = config.data.general.missing_value

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"predictions/{timestamp}")
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, metadata = load_model(
        model_path=model_path,
        config=config,
        is_prediction_mode=True
    )

    # Fetch prediction data
    data_package = await fetch_data_for_prediction(config)

    # Generate predictions
    predictions_dict = predict_with_model(model, data_package, config)

    # Format predictions
    predictions_df = format_predictions(predictions_dict, data_package, config)

    # Calculate metrics - only on valid (non-missing) values
    metrics = {}
    if "error" in predictions_df.columns:
        # Create a valid data mask (exclude missing values and NaN errors)
        valid_mask = predictions_df["error"].notna() & (predictions_df["prediction"] != missing_value) & (predictions_df["actual"] != missing_value)
        valid_df = predictions_df[valid_mask]

        if len(valid_df) > 0:
            # Use our enhanced metrics calculation
            metrics = calculate_error_metrics(
                predictions=valid_df["prediction"].values,
                targets=valid_df["actual"].values,
                missing_value=missing_value
            )

            # Add standard metadata
            metrics["valid_points"] = len(valid_df)
            metrics["total_points"] = len(predictions_df)
            metrics["missing_points"] = len(predictions_df) - len(valid_df)

            # Add detailed metrics
            metrics["by_horizon"] = calculate_metrics_by_horizon(
                predictions_df, missing_value=missing_value).to_dict()
            metrics["by_sensor"] = calculate_metrics_by_sensor(
                predictions_df, top_n=10, missing_value=missing_value).to_dict()
        else:
            logger.warning("No valid data points for metric calculation")
            metrics = {
                "mse": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "poisson_deviance": 0.0,
                "mape": 0.0,
                "wape": 0.0,
                "rmsle": 0.0,
                "valid_points": 0,
                "total_points": len(predictions_df),
                "missing_points": len(predictions_df)
            }

    # Save predictions
    predictions_path = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # Generate visualizations if requested
    if visualize:
        from gnn_package.src.visualization.visualization_utils import VisualizationManager
        viz_manager = VisualizationManager()
        viz_paths = viz_manager.save_visualization_pack(
            predictions_df,
            output_dir,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    else:
        viz_paths = {}

    # In the predict_and_evaluate function, find the summary report generation code:
    summary_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Prediction Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Date/Time: {datetime.now()}\n")
        f.write(f"Model: {model_path}\n\n")

        f.write(f"Total predictions: {len(predictions_df)}\n")
        f.write(f"Total sensors: {predictions_df['node_id'].nunique()}\n")
        if "valid_points" in metrics:
            f.write(f"Valid data points: {metrics['valid_points']} ({metrics['valid_points']/metrics['total_points']*100:.1f}%)\n")
            f.write(f"Missing data points: {metrics['missing_points']} ({metrics['missing_points']/metrics['total_points']*100:.1f}%)\n\n")

        if metrics:
            f.write(f"Overall metrics (excluding missing values {missing_value}):\n")
            f.write(f"  MSE: {metrics['mse']:.4f}\n")
            f.write(f"  MAE: {metrics['mae']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}\n")

            # Add count-specific metrics
            f.write(f"  Poisson Deviance: {metrics['poisson_deviance']:.4f}\n")
            f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
            f.write(f"  WAPE: {metrics['wape']:.2f}%\n")
            f.write(f"  RMSLE: {metrics['rmsle']:.4f}\n\n")

            # Add horizon metrics
            f.write("Metrics by prediction horizon:\n")
            if "by_horizon" in metrics:
                horizon_df = pd.DataFrame(metrics["by_horizon"])
                f.write(horizon_df.to_string() + "\n\n")

            # Add sensor metrics
            f.write("Top 10 sensors by error:\n")
            if "by_sensor" in metrics:
                sensor_df = pd.DataFrame(metrics["by_sensor"])
                f.write(sensor_df.to_string() + "\n")

    return {
        "predictions": predictions_dict,
        "dataframe": predictions_df,
        "metrics": metrics,
        "output_dir": str(output_dir),
        "visualization_paths": viz_paths,
    }