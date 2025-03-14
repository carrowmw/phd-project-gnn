# gnn_package/src/models/predict.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from gnn_package.src.preprocessing import (
    load_graph_data,
    compute_adjacency_matrix,
    SensorDataFetcher,
    TimeSeriesPreprocessor,
    get_sensor_name_id_map,
)
from gnn_package.src.models.stgnn import create_stgnn_model


def load_model(
    model_path, input_dim=1, hidden_dim=64, output_dim=1, horizon=6, num_layers=2
):
    """
    Load a trained STGNN model

    Parameters:
    -----------
    model_path : str
        Path to the saved model
    input_dim, hidden_dim, output_dim, horizon, num_layers : model parameters

    Returns:
    --------
    Loaded model
    """
    model = create_stgnn_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        horizon=horizon,
        num_layers=num_layers,
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def fetch_recent_data(node_ids, days_back=1, window_size=24):
    """
    Fetch recent data for prediction

    Parameters:
    -----------
    node_ids : list
        List of node IDs to fetch data for
    days_back : int
        Number of days of historical data to fetch
    window_size : int
        Size of the input window for the model

    Returns:
    --------
    Dict containing preprocessed data for prediction
    """
    # Fetch sensor data
    fetcher = SensorDataFetcher()
    response = fetcher.get_sensor_data_batch(node_ids, days_back=days_back)

    print(f"Fetched data for {len(response.data)} nodes")
    print(
        f"API Stats: {response.stats.successful_calls} successful, {response.stats.failed_calls} failed"
    )

    # Filter out empty results
    time_series_dict = {
        node_id: data for node_id, data in response.data.items() if data is not None
    }
    print(f"Nodes with valid data: {len(time_series_dict)}/{len(node_ids)}")

    # Create windows for time series
    processor = TimeSeriesPreprocessor(
        window_size=window_size,
        stride=1,
        gap_threshold=pd.Timedelta(minutes=15),
        missing_value=-1.0,
    )

    X_by_sensor, masks_by_sensor, metadata_by_sensor = processor.create_windows(
        time_series_dict
    )

    # Get the most recent window for each sensor
    latest_windows = {}
    latest_masks = {}
    time_indices = {}

    for node_id in X_by_sensor:
        if len(X_by_sensor[node_id]) > 0:
            # Get the last window
            latest_windows[node_id] = X_by_sensor[node_id][-1:]
            latest_masks[node_id] = masks_by_sensor[node_id][-1:]

            # Get time index information for this window
            metadata = metadata_by_sensor[node_id][-1]
            time_indices[node_id] = metadata

    return {
        "windows": latest_windows,
        "masks": latest_masks,
        "metadata": time_indices,
        "time_series": time_series_dict,
    }


def make_prediction(model, data, adj_matrix, node_ids, device=None):
    """
    Make predictions with the trained model

    Parameters:
    -----------
    model : STGNN
        Trained model
    data : dict
        Dict containing preprocessed data as returned by fetch_recent_data
    adj_matrix : numpy.ndarray
        Adjacency matrix for the graph
    node_ids : list
        List of node IDs
    device : torch.device
        Device to use for inference

    Returns:
    --------
    Dict containing predictions and related information
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Prepare input data
    all_inputs = []
    all_masks = []
    valid_nodes = []

    for i, node_id in enumerate(node_ids):
        if node_id in data["windows"]:
            x = torch.FloatTensor(data["windows"][node_id])
            mask = torch.FloatTensor(data["masks"][node_id])

            all_inputs.append(x)
            all_masks.append(mask)
            valid_nodes.append(node_id)

    if not all_inputs:
        raise ValueError("No valid input data for prediction")

    # Concatenate all inputs
    x = torch.cat(all_inputs, dim=0)
    mask = torch.cat(all_masks, dim=0)

    # Convert to tensors and move to device
    x = x.to(device)
    mask = mask.to(device)
    adj = torch.FloatTensor(adj_matrix).to(device)

    # Add batch dimension if needed
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
        mask = mask.unsqueeze(0)

    # For STGNN model, reshape input to [batch, nodes, time, features]
    if len(x.shape) == 3:
        x = x.unsqueeze(-1)  # Add feature dimension

    # Make prediction
    with torch.no_grad():
        predictions = model(x, adj, mask)

    # Move predictions to CPU
    predictions = predictions.cpu().numpy()

    return {"predictions": predictions, "valid_nodes": valid_nodes}


def plot_predictions(predictions, data, node_ids, name_id_map=None):
    """
    Plot predictions alongside historical data

    Parameters:
    -----------
    predictions : dict
        Dict containing predictions from make_prediction
    data : dict
        Dict containing time series data from fetch_recent_data
    node_ids : list
        List of node IDs
    name_id_map : dict, optional
        Mapping from node IDs to sensor names

    Returns:
    --------
    matplotlib figure
    """
    if name_id_map is None:
        name_id_map = get_sensor_name_id_map()
        # Reverse the mapping to go from id to name
        name_id_map = {v: k for k, v in name_id_map.items()}

    pred_array = predictions["predictions"]
    valid_nodes = predictions["valid_nodes"]

    # Create a figure
    n_nodes = min(len(valid_nodes), 6)  # Limit to 6 nodes
    fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 3 * n_nodes))
    if n_nodes == 1:
        axes = [axes]

    for i, node_id in enumerate(valid_nodes[:n_nodes]):
        ax = axes[i]

        # Get historical data
        historical = data["time_series"][node_id]

        # Get prediction for this node
        node_idx = valid_nodes.index(node_id)
        pred = pred_array[node_idx, 0, :, 0]  # [batch=0, node_idx, time, feature=0]

        # Get the last timestamp from historical data
        last_time = historical.index[-1]

        # Create time indices for prediction
        pred_times = [
            last_time + timedelta(minutes=15 * (i + 1)) for i in range(len(pred))
        ]

        # Plot
        ax.plot(historical.index, historical.values, label="Historical", color="blue")
        ax.plot(pred_times, pred, "r--", label="Prediction", linewidth=2)

        # Add sensor name to title if available
        sensor_name = name_id_map.get(node_id, node_id)
        ax.set_title(f"Sensor: {sensor_name} (ID: {node_id})")
        ax.set_ylabel("Traffic Count")
        ax.legend()

        # Format x-axis as time
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def predict_all_sensors(model_path, graph_prefix, output_file=None, plot=True):
    """
    Make predictions for all available sensors

    Parameters:
    -----------
    model_path : str
        Path to the saved model
    graph_prefix : str
        Prefix for the graph data files
    output_file : str, optional
        Path to save predictions
    plot : bool
        Whether to generate and display plots

    Returns:
    --------
    Dict containing predictions and related information
    """
    # Load graph data
    adj_matrix, node_ids, metadata = load_graph_data(
        prefix=graph_prefix, return_df=False
    )

    # Compute graph weights
    weighted_adj = compute_adjacency_matrix(adj_matrix, sigma_squared=0.1, epsilon=0.5)

    # Load model
    model = load_model(model_path)

    # Fetch recent data
    data = fetch_recent_data(node_ids, days_back=1, window_size=model.horizon)

    # Get valid nodes for prediction
    valid_nodes = list(data["windows"].keys())

    if not valid_nodes:
        print("No valid data for prediction")
        return None

    # Create filtered adjacency matrix
    valid_indices = [node_ids.index(nid) for nid in valid_nodes if nid in node_ids]
    filtered_adj = weighted_adj[valid_indices, :][:, valid_indices]
    filtered_nodes = [node_ids[idx] for idx in valid_indices]

    # Make predictions
    results = make_prediction(model, data, filtered_adj, filtered_nodes)

    # Get mapping from ID to name
    id_to_name = get_sensor_name_id_map()
    id_to_name = {v: k for k, v in id_to_name.items()}

    # Format predictions
    preds = results["predictions"]
    pred_nodes = results["valid_nodes"]

    # Create results dataframe
    last_times = {
        node_id: data["time_series"][node_id].index[-1] for node_id in pred_nodes
    }

    # Convert predictions to dataframe
    all_rows = []

    for i, node_id in enumerate(pred_nodes):
        node_preds = preds[i, 0, :, 0]  # [node_idx, batch=0, time, feature=0]
        last_time = last_times[node_id]

        for t, value in enumerate(node_preds):
            future_time = last_time + timedelta(minutes=15 * (t + 1))

            row = {
                "node_id": node_id,
                "sensor_name": id_to_name.get(node_id, ""),
                "timestamp": future_time,
                "prediction": value,
                "horizon": t + 1,
            }
            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)

    # Save to file if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    # Plot if requested
    if plot:
        fig = plot_predictions(results, data, filtered_nodes, id_to_name)
        plt.show()

    return {"predictions": results, "dataframe": results_df, "data": data}


if __name__ == "__main__":
    # Example usage
    predictions = predict_all_sensors(
        model_path="stgnn_model.pth",
        graph_prefix="traffic_graph",
        output_file="predictions.csv",
        plot=True,
    )

    # Print summary
    if predictions:
        df = predictions["dataframe"]
        print(f"Generated {len(df)} predictions for {df['node_id'].nunique()} sensors")

        # Show prediction ranges
        print("\nPrediction summary stats:")
        print(df.groupby("horizon")["prediction"].describe())
