# src/utils/metrics.py
import numpy as np
import torch
import pandas as pd
from typing import Dict, Union, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_error_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    reduction: str = "mean",
) -> Dict[str, float]:
    """
    Calculate standard error metrics with proper handling of masked values.

    Parameters:
    -----------
    predictions : numpy.ndarray or torch.Tensor
        Predicted values
    targets : numpy.ndarray or torch.Tensor
        Target values
    masks : numpy.ndarray or torch.Tensor, optional
        Binary masks for valid values (1 = valid, 0 = invalid)
    reduction : str
        How to reduce metrics ('mean', 'sum', 'none')

    Returns:
    --------
    Dict[str, float]
        Dictionary with calculated metrics (MSE, MAE, RMSE)
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if masks is not None and isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    # Create default mask if none provided
    if masks is None:
        masks = np.ones_like(predictions)

    # Ensure shapes match
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}"
        )
    if predictions.shape != masks.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape}, masks {masks.shape}"
        )

    # Calculate squared errors
    squared_errors = ((predictions - targets) ** 2) * masks
    absolute_errors = np.abs(predictions - targets) * masks

    # Apply reduction
    if reduction == "mean":
        # Normalize by sum of masks to only account for valid points
        mask_sum = np.sum(masks)
        if mask_sum == 0:
            logger.warning("No valid points in mask, returning zero metrics")
            return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}

        mse = np.sum(squared_errors) / mask_sum
        mae = np.sum(absolute_errors) / mask_sum
        rmse = np.sqrt(mse)
    elif reduction == "sum":
        mse = np.sum(squared_errors)
        mae = np.sum(absolute_errors)
        rmse = np.sqrt(mse)
    elif reduction == "none":
        # Return unreduced arrays
        return {"squared_errors": squared_errors, "absolute_errors": absolute_errors}
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return {"mse": float(mse), "mae": float(mae), "rmse": float(rmse)}


def calculate_metrics_by_horizon(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate error metrics grouped by prediction horizon.

    Parameters:
    -----------
    predictions_df : pandas.DataFrame
        DataFrame with predictions and actual values
        Must contain columns: 'prediction', 'actual', 'horizon'

    Returns:
    --------
    pandas.DataFrame
        DataFrame with metrics by horizon
    """
    if not all(
        col in predictions_df.columns for col in ["prediction", "actual", "horizon"]
    ):
        missing = [
            col
            for col in ["prediction", "actual", "horizon"]
            if col not in predictions_df.columns
        ]
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Calculate errors
    predictions_df = predictions_df.copy()
    predictions_df["error"] = predictions_df["prediction"] - predictions_df["actual"]
    predictions_df["abs_error"] = predictions_df["error"].abs()
    predictions_df["squared_error"] = predictions_df["error"] ** 2

    # Group by horizon and calculate metrics
    metrics_by_horizon = (
        predictions_df.groupby("horizon")
        .agg({"abs_error": "mean", "squared_error": "mean", "prediction": "count"})
        .rename(
            columns={"abs_error": "mae", "squared_error": "mse", "prediction": "count"}
        )
    )

    # Add RMSE
    metrics_by_horizon["rmse"] = np.sqrt(metrics_by_horizon["mse"])

    return metrics_by_horizon


def calculate_metrics_by_sensor(
    predictions_df: pd.DataFrame, top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate error metrics grouped by sensor.

    Parameters:
    -----------
    predictions_df : pandas.DataFrame
        DataFrame with predictions and actual values
        Must contain columns: 'prediction', 'actual', 'node_id' or 'sensor_name'
    top_n : int, optional
        If provided, return only top N sensors by error

    Returns:
    --------
    pandas.DataFrame
        DataFrame with metrics by sensor
    """
    # Determine grouping column
    if "sensor_name" in predictions_df.columns:
        grouping_col = "sensor_name"
    elif "node_id" in predictions_df.columns:
        grouping_col = "node_id"
    else:
        raise ValueError(
            "DataFrame must contain either 'sensor_name' or 'node_id' column"
        )

    # Calculate errors
    predictions_df = predictions_df.copy()
    predictions_df["error"] = predictions_df["prediction"] - predictions_df["actual"]
    predictions_df["abs_error"] = predictions_df["error"].abs()
    predictions_df["squared_error"] = predictions_df["error"] ** 2

    # Group by sensor and calculate metrics
    metrics_by_sensor = (
        predictions_df.groupby(grouping_col)
        .agg({"abs_error": "mean", "squared_error": "mean", "prediction": "count"})
        .rename(
            columns={"abs_error": "mae", "squared_error": "mse", "prediction": "count"}
        )
    )

    # Add RMSE
    metrics_by_sensor["rmse"] = np.sqrt(metrics_by_sensor["mse"])

    # Sort by MAE
    metrics_by_sensor = metrics_by_sensor.sort_values("mae", ascending=False)

    # Return top N if requested
    if top_n is not None:
        return metrics_by_sensor.head(top_n)

    return metrics_by_sensor


def format_prediction_results(
    predictions: np.ndarray,
    time_series_dict: Dict[str, pd.Series],
    node_ids: List[str],
    node_indices: np.ndarray,
    window_size: int,
    horizon: int,
    id_to_name_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Format model predictions into a standardized DataFrame.

    Parameters:
    -----------
    predictions : numpy.ndarray
        Model predictions with shape [batch, num_nodes, horizon, features]
    time_series_dict : Dict[str, pd.Series]
        Dictionary mapping node IDs to their time series data
    node_ids : List[str]
        List of all node IDs
    node_indices : numpy.ndarray
        Indices of nodes in the predictions
    window_size : int
        Size of input windows
    horizon : int
        Prediction horizon
    id_to_name_map : Dict[str, str], optional
        Mapping from node IDs to readable names

    Returns:
    --------
    pandas.DataFrame
        Formatted prediction results
    """
    rows = []

    # Get valid nodes
    valid_nodes = [node_ids[idx] for idx in node_indices]

    # Process each node
    for i, node_id in enumerate(valid_nodes):
        # Skip if no time series data available
        if node_id not in time_series_dict:
            logger.warning(f"No time series data for node {node_id}")
            continue

        # Get the time series
        series = time_series_dict[node_id]

        # Split into input and validation parts
        validation_data = series[-horizon:] if len(series) >= horizon else series

        # Get node position in predictions
        node_idx = node_indices[i]

        # Get predictions for this node
        node_preds = predictions[0, i, :, 0]  # [batch=0, node, time, feature=0]

        # Get prediction timestamps (use validation data timestamps if available)
        if len(validation_data) > 0:
            timestamps = validation_data.index
        else:
            # If no validation data, use the last timestamp + increments
            last_timestamp = series.index[-1] if len(series) > 0 else pd.Timestamp.now()
            freq = pd.infer_freq(series.index) if len(series) > 1 else "15min"
            if freq is None:
                freq = "15min"  # Default if frequency can't be inferred
            timestamps = pd.date_range(
                start=last_timestamp, periods=horizon, freq=freq
            )[1:]

        # Create rows for each prediction horizon
        for h, pred_value in enumerate(node_preds):
            if h < len(validation_data):
                # We have actual data for validation
                actual_time = timestamps[h]
                actual_value = validation_data.iloc[h]

                # Create a row with prediction and actual value
                row = {
                    "node_id": node_id,
                    "sensor_name": (
                        id_to_name_map.get(node_id, str(node_id))
                        if id_to_name_map
                        else str(node_id)
                    ),
                    "timestamp": actual_time,
                    "prediction": float(pred_value),
                    "actual": float(actual_value),
                    "error": float(pred_value - actual_value),
                    "abs_error": float(abs(pred_value - actual_value)),
                    "horizon": h + 1,  # 1-based horizon index
                }
            else:
                # No actual data available, just store prediction
                # Get timestamp by extrapolation if needed
                if h < len(timestamps):
                    pred_time = timestamps[h]
                else:
                    # Extrapolate timestamps if needed
                    freq = pd.infer_freq(timestamps) if len(timestamps) > 1 else "15min"
                    if freq is None:
                        freq = "15min"
                    pred_time = timestamps[-1] + pd.Timedelta(freq) * (
                        h - len(timestamps) + 1
                    )

                row = {
                    "node_id": node_id,
                    "sensor_name": (
                        id_to_name_map.get(node_id, str(node_id))
                        if id_to_name_map
                        else str(node_id)
                    ),
                    "timestamp": pred_time,
                    "prediction": float(pred_value),
                    "actual": None,
                    "error": None,
                    "abs_error": None,
                    "horizon": h + 1,
                }

            rows.append(row)

    # Create DataFrame
    if rows:
        return pd.DataFrame(rows)
    else:
        # Return empty DataFrame with expected columns
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
