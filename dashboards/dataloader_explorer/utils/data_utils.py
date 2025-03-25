# dashboards/dataloader_explorer/utils/data_utils.py

import os
import pickle
import numpy as np
import pandas as pd
import torch


def load_data_loaders(pickle_path):
    """
    Load data loaders from a pickle file

    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file containing data loaders

    Returns:
    --------
    dict
        Dictionary containing train_loader, val_loader, and other data
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")

    with open(pickle_path, "rb") as f:
        data_loaders = pickle.load(f)

    return data_loaders


def get_batch_from_loader(data_loader, batch_index=0):
    """
    Get a specific batch from a data loader

    Parameters:
    -----------
    data_loader : torch.utils.data.DataLoader
        DataLoader to get batch from
    batch_index : int, optional
        Index of the batch to retrieve

    Returns:
    --------
    dict
        Batch data
    """
    # Create an iterator from the data loader
    iterator = iter(data_loader)

    # Skip to the desired batch
    batch = None
    for i in range(batch_index + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            raise IndexError(f"Batch index {batch_index} is out of range")

    return batch


def get_node_data(batch, batch_idx, node_idx):
    """
    Extract data for a specific node from a batch

    Parameters:
    -----------
    batch : dict
        Batch data
    batch_idx : int
        Index within the batch
    node_idx : int
        Index of the node

    Returns:
    --------
    dict
        Node data with keys: input_data, input_mask, target_data, target_mask, node_id
    """
    # Validate indices
    batch_size, num_nodes = batch["x"].shape[0], batch["x"].shape[1]

    if batch_idx >= batch_size:
        raise ValueError(f"batch_idx {batch_idx} is out of range (max {batch_size-1})")

    if node_idx >= num_nodes:
        raise ValueError(f"node_idx {node_idx} is out of range (max {num_nodes-1})")

    # Get the actual node ID
    node_id = batch["node_indices"][node_idx].item()

    # Get data for this node
    x_data = batch["x"][batch_idx, node_idx, :, 0].cpu().numpy()
    x_mask = batch["x_mask"][batch_idx, node_idx, :, 0].cpu().numpy()
    y_data = batch["y"][batch_idx, node_idx, :, 0].cpu().numpy()
    y_mask = batch["y_mask"][batch_idx, node_idx, :, 0].cpu().numpy()

    # Get dimension information
    seq_len = x_data.shape[0]
    horizon = y_data.shape[0]

    return {
        "input_data": x_data,
        "input_mask": x_mask,
        "target_data": y_data,
        "target_mask": y_mask,
        "node_id": node_id,
        "seq_len": seq_len,
        "horizon": horizon,
    }


def get_dataloader_stats(data_loader):
    """
    Compute summary statistics for a data loader

    Parameters:
    -----------
    data_loader : torch.utils.data.DataLoader
        DataLoader to analyze

    Returns:
    --------
    dict
        Statistics about the data loader
    """
    try:
        # Get a sample batch
        batch = get_batch_from_loader(data_loader)

        # Extract shapes and dimensions
        x = batch["x"]
        x_mask = batch["x_mask"]
        y = batch["y"]
        y_mask = batch["y_mask"]
        adj = batch["adj"]
        node_indices = batch["node_indices"]

        # Basic statistics
        batch_size, num_nodes, seq_len, features = x.shape
        _, _, horizon, _ = y.shape

        # Compute missing data percentage
        input_missing_pct = 100 - (x_mask.sum().item() / x_mask.numel() * 100)
        target_missing_pct = 100 - (y_mask.sum().item() / y_mask.numel() * 100)

        # Compute basic data statistics
        x_valid = x[x_mask.bool()]
        y_valid = y[y_mask.bool()]

        x_stats = {
            "mean": x_valid.mean().item() if len(x_valid) > 0 else None,
            "std": x_valid.std().item() if len(x_valid) > 0 else None,
            "min": x_valid.min().item() if len(x_valid) > 0 else None,
            "max": x_valid.max().item() if len(x_valid) > 0 else None,
        }

        y_stats = {
            "mean": y_valid.mean().item() if len(y_valid) > 0 else None,
            "std": y_valid.std().item() if len(y_valid) > 0 else None,
            "min": y_valid.min().item() if len(y_valid) > 0 else None,
            "max": y_valid.max().item() if len(y_valid) > 0 else None,
        }

        # Adjacency matrix statistics
        adj_numpy = adj.cpu().numpy()
        adj_stats = {
            "min": float(np.min(adj_numpy)),
            "max": float(np.max(adj_numpy)),
            "mean": float(np.mean(adj_numpy)),
            "sparsity": float(
                100 - (np.count_nonzero(adj_numpy) / adj_numpy.size * 100)
            ),
        }

        # Collect and return all statistics
        return {
            "batch_size": batch_size,
            "num_nodes": num_nodes,
            "seq_len": seq_len,
            "horizon": horizon,
            "features": features,
            "input_missing_pct": input_missing_pct,
            "target_missing_pct": target_missing_pct,
            "unique_node_ids": node_indices.cpu().numpy().tolist(),
            "x_stats": x_stats,
            "y_stats": y_stats,
            "adj_stats": adj_stats,
        }

    except Exception as e:
        return {"error": str(e)}


def compute_node_correlations(batch, batch_idx=0):
    """
    Compute correlations between nodes in a batch

    Parameters:
    -----------
    batch : dict
        Batch data
    batch_idx : int
        Index within the batch

    Returns:
    --------
    tuple
        (correlation_matrix, node_ids)
    """
    # Extract batch components
    x = batch["x"]  # [batch_size, num_nodes, seq_len, features]
    x_mask = batch["x_mask"]
    node_indices = batch["node_indices"]

    # Get dimensions
    batch_size, num_nodes, seq_len, _ = x.shape

    # Validate batch_idx
    if batch_idx >= batch_size:
        raise ValueError(f"batch_idx {batch_idx} is out of range (max {batch_size-1})")

    # Extract data for all nodes in this batch
    all_node_data = []
    node_ids = []

    for node_idx in range(num_nodes):
        # Get the node ID
        node_id = node_indices[node_idx].item()
        node_ids.append(node_id)

        # Get node data and mask
        node_data = x[batch_idx, node_idx, :, 0].cpu().numpy()
        node_mask = x_mask[batch_idx, node_idx, :, 0].cpu().numpy()

        # Apply mask (replace missing values with NaN)
        masked_data = np.where(node_mask > 0, node_data, np.nan)
        all_node_data.append(masked_data)

    # Convert to DataFrame
    df = pd.DataFrame(
        np.array(all_node_data).T, columns=[f"Node {node_id}" for node_id in node_ids]
    )

    # Compute correlation matrix (using pairwise complete observations)
    corr_matrix = df.corr(method="pearson", min_periods=3)

    return corr_matrix, node_ids
