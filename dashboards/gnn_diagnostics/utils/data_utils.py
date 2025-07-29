# dashboards/gnn_diagnostics/utils/data_utils.py

import os
import json
import pickle
import numpy as np
import pandas as pd
import yaml
import re
from pathlib import Path
import torch


def load_experiment_data(experiment_dir):
    """
    Load all relevant data from an experiment directory

    Parameters:
    -----------
    experiment_dir : str or Path
        Path to the experiment directory

    Returns:
    --------
    dict
        Dictionary containing experiment data
    """
    experiment_dir = Path(experiment_dir)
    result = {
        "experiment_dir": str(experiment_dir),
        "experiment_name": experiment_dir.name,
    }

    # Load configuration
    config_path = experiment_dir / "config.yml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                result["config"] = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")

    # Load prediction results
    prediction_dir = experiment_dir / "prediction"
    if prediction_dir.exists():
        # Find prediction CSV files
        prediction_files = list(prediction_dir.glob("predictions_*.csv"))
        if prediction_files:
            # Use the latest prediction file
            prediction_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest_prediction = prediction_files[0]
            try:
                result["predictions_df"] = pd.read_csv(latest_prediction)
                result["predictions_file"] = str(latest_prediction)
            except Exception as e:
                print(f"Error loading predictions: {e}")

        # Find summary files
        summary_files = list(prediction_dir.glob("summary_*.txt"))
        if summary_files:
            # Use the latest summary file
            summary_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest_summary = summary_files[0]
            try:
                with open(latest_summary, "r") as f:
                    result["summary_text"] = f.read()
                result["summary_file"] = str(latest_summary)
            except Exception as e:
                print(f"Error loading summary: {e}")

        # Find visualization files
        error_dist_files = list(prediction_dir.glob("error_distribution_*.png"))
        sensors_grid_files = list(prediction_dir.glob("sensors_grid_*.png"))

        if error_dist_files:
            error_dist_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            result["error_dist_file"] = str(error_dist_files[0])

        if sensors_grid_files:
            sensors_grid_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            result["sensors_grid_file"] = str(sensors_grid_files[0])

    # Load training information
    performance_path = experiment_dir / "performance.json"
    if performance_path.exists():
        try:
            with open(performance_path, "r") as f:
                result["performance"] = json.load(f)
        except Exception as e:
            print(f"Error loading performance data: {e}")

    # Check for training curve PNG
    training_curve_path = experiment_dir / "training_curve.png"
    if training_curve_path.exists():
        result["training_curve_img"] = str(training_curve_path)

    # Load runtime statistics
    runtime_stats_path = experiment_dir / "runtime_stats.json"
    if runtime_stats_path.exists():
        try:
            with open(runtime_stats_path, "r") as f:
                result["runtime_stats"] = json.load(f)
        except Exception as e:
            print(f"Error loading runtime stats: {e}")

    # Load training curves
    try:
        # Check for training curve pickle
        pickled_data_paths = list(experiment_dir.glob("*.pkl"))
        for pkl_path in pickled_data_paths:
            if "data_loaders" in pkl_path.name:
                # This is likely the data loaders, not what we want
                continue

            try:
                with open(pkl_path, "rb") as f:
                    pkl_data = pickle.load(f)
                    if isinstance(pkl_data, dict):
                        if "train_losses" in pkl_data:
                            result["training_curves"] = pkl_data
                            break
            except Exception as e:
                print(f"Error reading pickle file {pkl_path}: {e}")
    except Exception as e:
        print(f"Error searching for training curves: {e}")

    # Try to load graph data (adjacency matrix)
    try:
        # Check if a model file exists
        model_path = experiment_dir / "model.pth"
        if model_path.exists():
            # Model exists, now try to load preprocessed data
            data_loader_files = list(experiment_dir.glob("data_loaders_*.pkl"))
            if data_loader_files:
                # Use the first data loader file
                data_loader_path = data_loader_files[0]
                try:
                    with open(data_loader_path, "rb") as f:
                        data_loaders = pickle.load(f)

                        # Extract graph data if available
                        if isinstance(data_loaders, dict) and "graph_data" in data_loaders:
                            result["graph_data"] = data_loaders["graph_data"]

                        # Extract time series data if available
                        if isinstance(data_loaders, dict) and "time_series" in data_loaders:
                            result["time_series"] = data_loaders["time_series"]

                except Exception as e:
                    print(f"Error loading data loaders: {e}")

        # Try to load raw data file referenced in data loaders
        if "config" in result and "data" in result["config"]:
            # Look for data path
            data_file = None
            data_config = result["config"]["data"]

            if "file_path" in data_config:
                data_file = data_config["file_path"]

            if data_file and os.path.exists(data_file):
                try:
                    with open(data_file, "rb") as f:
                        result["raw_data"] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading raw data: {e}")
    except Exception as e:
        print(f"Error loading graph data: {e}")

    # Try to extract metrics from summary text
    if "summary_text" in result:
        metrics = extract_metrics_from_summary(result["summary_text"])
        if metrics:
            result["metrics"] = metrics

    return result


def extract_metrics_from_summary(summary_text):
    """
    Extract metrics from a summary text

    Parameters:
    -----------
    summary_text : str
        Summary text containing metrics

    Returns:
    --------
    dict
        Dictionary containing metrics
    """
    metrics = {}

    # Extract MSE, MAE, and RMSE
    mse_match = re.search(r"MSE:\s*([\d\.]+)", summary_text)
    mae_match = re.search(r"MAE:\s*([\d\.]+)", summary_text)
    rmse_match = re.search(r"RMSE:\s*([\d\.]+)", summary_text)

    if mse_match:
        metrics["mse"] = float(mse_match.group(1))
    if mae_match:
        metrics["mae"] = float(mae_match.group(1))
    if rmse_match:
        metrics["rmse"] = float(rmse_match.group(1))

    # Extract valid data points info
    valid_points_match = re.search(r"Valid data points:\s*(\d+)\s*\(([^)]+)%\)", summary_text)
    if valid_points_match:
        metrics["valid_points"] = int(valid_points_match.group(1))
        metrics["valid_percentage"] = float(valid_points_match.group(2))

    # Extract total points
    total_points_match = re.search(r"Total predictions:\s*(\d+)", summary_text)
    if total_points_match:
        metrics["total_points"] = int(total_points_match.group(1))

    # Extract total sensors
    total_sensors_match = re.search(r"Total sensors:\s*(\d+)", summary_text)
    if total_sensors_match:
        metrics["total_sensors"] = int(total_sensors_match.group(1))

    return metrics


def extract_metrics(experiment_data):
    """
    Extract metrics from experiment data

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data

    Returns:
    --------
    dict
        Dictionary containing metrics
    """
    metrics = {}

    # Check if metrics are already extracted
    if "metrics" in experiment_data:
        return experiment_data["metrics"]

    # Try to extract from predictions dataframe
    if "predictions_df" in experiment_data:
        df = experiment_data["predictions_df"]

        # Get missing value from config
        missing_value = -999.0
        if "config" in experiment_data and "data" in experiment_data["config"] and "general" in experiment_data["config"]["data"]:
            missing_value = experiment_data["config"]["data"]["general"].get("missing_value", -999.0)

        # Create a mask for valid values
        valid_mask = (df["prediction"] != missing_value) & (df["actual"] != missing_value)
        valid_df = df[valid_mask]

        if len(valid_df) > 0:
            metrics["mse"] = ((valid_df["prediction"] - valid_df["actual"]) ** 2).mean()
            metrics["mae"] = (valid_df["prediction"] - valid_df["actual"]).abs().mean()
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["valid_points"] = len(valid_df)
            metrics["total_points"] = len(df)

    return metrics


def get_data_for_sensor(experiment_data, sensor_id):
    """
    Extract all data for a specific sensor

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data
    sensor_id : str
        Sensor ID to extract data for

    Returns:
    --------
    dict
        Dictionary containing sensor data
    """
    result = {
        "sensor_id": sensor_id,
    }

    # Extract predictions for this sensor
    if "predictions_df" in experiment_data:
        df = experiment_data["predictions_df"]
        sensor_preds = df[df["node_id"] == sensor_id]

        if not sensor_preds.empty:
            result["predictions"] = sensor_preds

            # Get sensor name if available
            if "sensor_name" in sensor_preds.columns:
                result["sensor_name"] = sensor_preds["sensor_name"].iloc[0]

    # Extract raw data for this sensor if available
    if "raw_data" in experiment_data and sensor_id in experiment_data["raw_data"]:
        result["raw_data"] = experiment_data["raw_data"][sensor_id]

    # Extract preprocessing stats if available
    if "time_series" in experiment_data and "validation" in experiment_data["time_series"]:
        time_series = experiment_data["time_series"]["validation"]
        if sensor_id in time_series:
            result["validation_series"] = time_series[sensor_id]

    return result


def analyze_graph_connectivity(experiment_data):
    """
    Analyze graph connectivity from experiment data

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data

    Returns:
    --------
    dict
        Dictionary containing graph connectivity analysis
    """
    result = {}

    if "graph_data" not in experiment_data or "adj_matrix" not in experiment_data["graph_data"]:
        return {"error": "No adjacency matrix found"}

    try:
        adj_matrix = experiment_data["graph_data"]["adj_matrix"]
        node_ids = experiment_data["graph_data"]["node_ids"]

        # Convert to numpy if needed
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        # Basic statistics
        result["num_nodes"] = adj_matrix.shape[0]
        result["num_edges"] = np.sum(adj_matrix > 0) / 2  # Undirected graph, so divide by 2
        result["density"] = result["num_edges"] / (result["num_nodes"] * (result["num_nodes"] - 1) / 2)
        result["avg_degree"] = np.sum(adj_matrix > 0, axis=1).mean()
        result["min_weight"] = np.min(adj_matrix[adj_matrix > 0]) if np.any(adj_matrix > 0) else 0
        result["max_weight"] = np.max(adj_matrix)
        result["isolated_nodes"] = np.sum(np.sum(adj_matrix > 0, axis=1) == 0)

        # Weight distribution
        weights = adj_matrix[adj_matrix > 0].flatten()
        result["weight_mean"] = np.mean(weights) if len(weights) > 0 else 0
        result["weight_std"] = np.std(weights) if len(weights) > 0 else 0
        result["weight_quartiles"] = np.percentile(weights, [25, 50, 75]) if len(weights) > 0 else [0, 0, 0]

        # Node degrees
        degrees = np.sum(adj_matrix > 0, axis=1)
        result["degree_mean"] = np.mean(degrees)
        result["degree_std"] = np.std(degrees)
        result["degree_quartiles"] = np.percentile(degrees, [25, 50, 75])
        result["degree_max"] = np.max(degrees)
        result["degree_min"] = np.min(degrees)

        # Connected components analysis
        import networkx as nx
        G = nx.Graph()
        for i in range(len(node_ids)):
            for j in range(i+1, len(node_ids)):
                if adj_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=float(adj_matrix[i, j]))

        components = list(nx.connected_components(G))
        result["num_components"] = len(components)
        if components:
            result["largest_component_size"] = len(max(components, key=len))
            result["smallest_component_size"] = len(min(components, key=len))
            result["component_sizes"] = [len(c) for c in components]

        return result

    except Exception as e:
        print(f"Error analyzing graph connectivity: {e}")
        return {"error": str(e)}


def analyze_missing_data(experiment_data, sensor_id=None):
    """
    Analyze missing data patterns in the experiment

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data
    sensor_id : str, optional
        Specific sensor to analyze

    Returns:
    --------
    dict
        Dictionary containing missing data analysis
    """
    result = {}

    # Get missing value from config
    missing_value = -999.0
    if "config" in experiment_data and "data" in experiment_data["config"] and "general" in experiment_data["config"]["data"]:
        missing_value = experiment_data["config"]["data"]["general"].get("missing_value", -999.0)

    # Analyze raw data if available
    if "raw_data" in experiment_data:
        raw_data = experiment_data["raw_data"]

        if sensor_id is not None:
            # Analyze specific sensor
            if sensor_id in raw_data:
                series = raw_data[sensor_id]
                total_points = len(series)
                missing_points = (series == missing_value).sum()
                missing_pct = missing_points / total_points * 100 if total_points > 0 else 0

                result["raw_data"] = {
                    "total_points": total_points,
                    "missing_points": missing_points,
                    "missing_pct": missing_pct,
                }
        else:
            # Analyze all sensors
            sensors = {}
            for sensor, series in raw_data.items():
                total_points = len(series)
                if hasattr(series, "values"):
                    missing_points = np.sum(series.values == missing_value)
                else:
                    missing_points = np.sum(np.array(series) == missing_value)
                missing_pct = missing_points / total_points * 100 if total_points > 0 else 0

                sensors[sensor] = {
                    "total_points": total_points,
                    "missing_points": missing_points,
                    "missing_pct": missing_pct,
                }

            result["raw_data_by_sensor"] = sensors

            # Overall statistics
            total_all = sum(s["total_points"] for s in sensors.values())
            missing_all = sum(s["missing_points"] for s in sensors.values())
            missing_pct_all = missing_all / total_all * 100 if total_all > 0 else 0

            result["raw_data_overall"] = {
                "total_points": total_all,
                "missing_points": missing_all,
                "missing_pct": missing_pct_all,
            }

    # Analyze prediction data
    if "predictions_df" in experiment_data:
        df = experiment_data["predictions_df"]

        if sensor_id is not None:
            # Filter for specific sensor
            df = df[df["node_id"] == sensor_id]

        # Analyze predictions
        total_points = len(df)
        missing_actual = df["actual"].isna().sum() + (df["actual"] == missing_value).sum()
        missing_pred = df["prediction"].isna().sum() + (df["prediction"] == missing_value).sum()

        missing_actual_pct = missing_actual / total_points * 100 if total_points > 0 else 0
        missing_pred_pct = missing_pred / total_points * 100 if total_points > 0 else 0

        result["predictions"] = {
            "total_points": total_points,
            "missing_actual": missing_actual,
            "missing_actual_pct": missing_actual_pct,
            "missing_pred": missing_pred,
            "missing_pred_pct": missing_pred_pct,
        }

        if sensor_id is None:
            # Analyze by sensor
            sensors = {}
            for sensor in df["node_id"].unique():
                sensor_df = df[df["node_id"] == sensor]
                total = len(sensor_df)
                missing_actual_sensor = sensor_df["actual"].isna().sum() + (sensor_df["actual"] == missing_value).sum()
                missing_pred_sensor = sensor_df["prediction"].isna().sum() + (sensor_df["prediction"] == missing_value).sum()

                missing_actual_pct_sensor = missing_actual_sensor / total * 100 if total > 0 else 0
                missing_pred_pct_sensor = missing_pred_sensor / total * 100 if total > 0 else 0

                sensors[sensor] = {
                    "total_points": total,
                    "missing_actual": missing_actual_sensor,
                    "missing_actual_pct": missing_actual_pct_sensor,
                    "missing_pred": missing_pred_sensor,
                    "missing_pred_pct": missing_pred_pct_sensor,
                }

            result["predictions_by_sensor"] = sensors

    return result