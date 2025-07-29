# src/utils/metrics.py
import numpy as np
import torch
import pandas as pd
from typing import Dict, Union, Tuple, List, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_error_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    missing_value: float = None,
    reduction: str = "mean",
) -> Dict[str, float]:
    """
    Calculate error metrics appropriate for count data.

    Parameters:
    -----------
    predictions : numpy.ndarray or torch.Tensor
        Predicted values
    targets : numpy.ndarray or torch.Tensor
        Target values
    masks : numpy.ndarray or torch.Tensor, optional
        Binary masks for valid values (1 = valid, 0 = invalid)
    missing_value : float
        Value to treat as missing/invalid in targets and predictions
    reduction : str
        How to reduce metrics ('mean', 'sum', 'none')

    Returns:
    --------
    Dict[str, float]
        Dictionary with calculated metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if masks is not None and isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    # Create missing value mask (1 = valid, 0 = missing)
    missing_mask_pred = (predictions != missing_value).astype(float)
    missing_mask_targ = (targets != missing_value).astype(float)

    # Combine masks - a point is valid only if both prediction and target are valid
    valid_mask = missing_mask_pred * missing_mask_targ

    # Apply user-provided mask if available
    if masks is not None:
        valid_mask = valid_mask * masks

    # Ensure targets are non-negative (required for Poisson and other count metrics)
    valid_indices = valid_mask > 0
    valid_predictions = predictions[valid_indices]
    valid_targets = targets[valid_indices]
    valid_targets = np.maximum(valid_targets, 0)

    # Skip if no valid points
    if len(valid_targets) == 0:
        logger.warning("No valid points after masking, returning zero metrics")
        return {
            "mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "poisson_deviance": 0.0,
            "mape": 0.0,
            "wape": 0.0,
            "rmsle": 0.0
        }

    # Calculate standard metrics
    mse = np.mean((valid_predictions - valid_targets) ** 2)
    mae = np.mean(np.abs(valid_predictions - valid_targets))
    rmse = np.sqrt(mse)

    # Count-specific metrics
    # 1. Poisson Deviance (2*(y*log(y/mu) - (y-mu)))
    eps = 1e-8  # To avoid division by zero or log(0)
    poisson_deviance = 2 * np.mean(
        valid_targets * np.log((valid_targets + eps) / (valid_predictions + eps)) -
        (valid_targets - valid_predictions)
    )

    # 2. Mean Absolute Percentage Error
    mape = np.mean(np.abs((valid_predictions - valid_targets) / (valid_targets + eps))) * 100

    # 3. Weighted Absolute Percentage Error (handles zero values better than MAPE)
    wape = np.sum(np.abs(valid_predictions - valid_targets)) / np.sum(valid_targets + eps) * 100

    # 4. Root Mean Squared Logarithmic Error (common for count data)
    rmsle = np.sqrt(np.mean((np.log1p(valid_predictions) - np.log1p(valid_targets)) ** 2))

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "poisson_deviance": float(poisson_deviance),
        "mape": float(mape),
        "wape": float(wape),
        "rmsle": float(rmsle)
    }

def calculate_metrics_by_horizon(
    predictions_df: pd.DataFrame,
    missing_value: float = None
) -> pd.DataFrame:
    """
    Calculate error metrics grouped by prediction horizon.

    Parameters:
    -----------
    predictions_df : pandas.DataFrame
        DataFrame with predictions and actual values
        Must contain columns: 'prediction', 'actual', 'horizon'
    missing_value : float
        Value to treat as missing in predictions and actuals

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

    # Create a valid data mask (exclude missing values)
    valid_mask = (predictions_df["prediction"] != missing_value) & (predictions_df["actual"] != missing_value)

    # Use only valid data points
    valid_df = predictions_df[valid_mask].copy()

    # Calculate errors
    valid_df["error"] = valid_df["prediction"] - valid_df["actual"]
    valid_df["abs_error"] = valid_df["error"].abs()
    valid_df["squared_error"] = valid_df["error"] ** 2

    # Add count-specific error calculations
    eps = 1e-8  # For numerical stability
    valid_df["poisson_dev"] = 2 * (
        valid_df["actual"] * np.log((valid_df["actual"] + eps) / (valid_df["prediction"] + eps)) -
        (valid_df["actual"] - valid_df["prediction"])
    )
    valid_df["log_squared_error"] = (np.log1p(valid_df["prediction"]) - np.log1p(valid_df["actual"])) ** 2

    # Group by horizon and calculate metrics
    metrics_by_horizon = (
        valid_df.groupby("horizon")
        .agg({
            "abs_error": "mean",
            "squared_error": "mean",
            "poisson_dev": "mean",
            "log_squared_error": "mean",
            "prediction": "count"
        })
        .rename(
            columns={
                "abs_error": "mae",
                "squared_error": "mse",
                "poisson_dev": "poisson_deviance",
                "log_squared_error": "msle",
                "prediction": "count"
            }
        )
    )

    # Add RMSE and RMSLE
    metrics_by_horizon["rmse"] = np.sqrt(metrics_by_horizon["mse"])
    metrics_by_horizon["rmsle"] = np.sqrt(metrics_by_horizon["msle"])

    return metrics_by_horizon


def calculate_metrics_by_sensor(
    predictions_df: pd.DataFrame,
    top_n: Optional[int] = None,
    missing_value: float = None
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
    missing_value : float
        Value to treat as missing in predictions and actuals

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

    # Create a valid data mask (exclude missing values)
    valid_mask = (predictions_df["prediction"] != missing_value) & (predictions_df["actual"] != missing_value)

    # Use only valid data points
    valid_df = predictions_df[valid_mask].copy()

    # Calculate errors
    valid_df["error"] = valid_df["prediction"] - valid_df["actual"]
    valid_df["abs_error"] = valid_df["error"].abs()
    valid_df["squared_error"] = valid_df["error"] ** 2

    # Add count-specific error calculations
    eps = 1e-8  # For numerical stability
    valid_df["poisson_dev"] = 2 * (
        valid_df["actual"] * np.log((valid_df["actual"] + eps) / (valid_df["prediction"] + eps)) -
        (valid_df["actual"] - valid_df["prediction"])
    )
    valid_df["log_squared_error"] = (np.log1p(valid_df["prediction"]) - np.log1p(valid_df["actual"])) ** 2

    # Group by sensor and calculate metrics
    metrics_by_sensor = (
        valid_df.groupby(grouping_col)
        .agg({
            "abs_error": "mean",
            "squared_error": "mean",
            "poisson_dev": "mean",
            "log_squared_error": "mean",
            "prediction": "count"
        })
        .rename(
            columns={
                "abs_error": "mae",
                "squared_error": "mse",
                "poisson_dev": "poisson_deviance",
                "log_squared_error": "msle",
                "prediction": "count"
            }
        )
    )

    # Add RMSE and RMSLE
    metrics_by_sensor["rmse"] = np.sqrt(metrics_by_sensor["mse"])
    metrics_by_sensor["rmsle"] = np.sqrt(metrics_by_sensor["msle"])

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
    missing_value: float = None
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
    missing_value : float
        Value to treat as missing in predictions and actuals

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
                    "horizon": h + 1,  # 1-based horizon index
                }

                # Only calculate error if neither value is a missing value
                if actual_value != missing_value and pred_value != missing_value:
                    row["error"] = float(pred_value - actual_value)
                    row["abs_error"] = float(abs(pred_value - actual_value))
                else:
                    row["error"] = None
                    row["abs_error"] = None
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