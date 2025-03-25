"""
Utility functions for the tensor flow dashboard with improved data loading.
Includes data loading, preprocessing helpers, and more.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from gnn_package import preprocessing


def load_sample_data(graph_prefix="25022025_test", sample_size=8, use_real_data=True):
    """
    Load data for the tensor flow dashboard.
    Attempts to load real data first, falls back to synthetic if needed.

    Now loads more nodes (8 by default) for better visualizations.

    Parameters:
    -----------
    graph_prefix : str
        Prefix for the graph data files
    sample_size : int
        Number of sensor nodes to include in visualization (increased from 5 to 8)
    use_real_data : bool
        Whether to try loading real data first (True) or go straight to synthetic (False)

    Returns:
    --------
    dict
        Dictionary containing adjacency matrix, node IDs, and time series data
    """
    if use_real_data:
        data = load_real_data(graph_prefix, sample_size)
        if data is not None:
            return data

    # Fall back to synthetic data if real data loading failed or was skipped
    return load_synthetic_data(sample_size=sample_size)


def load_real_data(
    graph_prefix="25022025_test", sample_size=8, data_file="test_data_1mnth.pkl"
):
    """
    Load real data from the dashboards/data folder with improved node selection

    Parameters:
    -----------
    graph_prefix : str
        Prefix for the graph data files
    sample_size : int
        Number of sensor nodes to include in visualization
    data_file : str
        Name of the time series data file

    Returns:
    --------
    dict
        Dictionary containing adjacency matrix, node IDs, and time series data
    """
    print(f"Loading graph data with prefix '{graph_prefix}'...")
    try:
        # Load adjacency matrix and node IDs
        adj_matrix, node_ids, metadata = preprocessing.load_graph_data(
            prefix=graph_prefix, return_df=False
        )

        print(
            f"Loaded adjacency matrix with shape {adj_matrix.shape} containing {len(node_ids)} nodes"
        )

        # Find potential data files
        potential_paths = [
            Path("dashboards/data") / data_file,
            Path("data") / data_file,
            Path("../data") / data_file,
            Path("dashboards/data/test_data_1mnth.pkl"),
            Path("dashboards/data/test_data.pkl"),
            Path(data_file) if data_file.startswith("/") else None,
        ]

        data_path = None
        for path in potential_paths:
            if path and path.exists():
                data_path = path
                print(f"Found data at {data_path}")
                break

        if not data_path:
            print(f"Warning: Could not find data file at any of these locations:")
            for path in potential_paths:
                if path:
                    print(f"  - {path}")
            return None

        # Load the pickle file
        print(f"Loading time series data from {data_path}")
        with open(data_path, "rb") as f:
            time_series_dict = pickle.load(f)

        print(f"Loaded time series data with {len(time_series_dict)} sensors")

        # First determine which nodes have data
        nodes_with_data = [
            node_id
            for node_id in node_ids
            if node_id in time_series_dict and time_series_dict[node_id] is not None
        ]

        print(f"Found {len(nodes_with_data)} nodes with valid time series data")

        if len(nodes_with_data) < 2:
            print(f"Warning: Not enough nodes with data (found {len(nodes_with_data)})")
            return None

        # For better visualizations, select nodes with the best data quality
        if len(nodes_with_data) > sample_size:
            # Calculate data quality for each node
            data_quality = {}
            for node_id in nodes_with_data:
                series = time_series_dict[node_id]
                if series is not None:
                    # Calculate percentage of non-NaN values and total data points
                    total_points = len(series)
                    valid_points = series.count()  # Count non-NaN values
                    quality_score = (
                        (valid_points / total_points) * np.log10(total_points)
                        if total_points > 0
                        else 0
                    )
                    data_quality[node_id] = quality_score

            # Select top nodes by data quality
            selected_nodes = sorted(
                data_quality.items(), key=lambda x: x[1], reverse=True
            )[:sample_size]
            selected_node_ids = [node_id for node_id, _ in selected_nodes]

            print(
                f"Selected {len(selected_node_ids)} nodes with best data quality from {len(nodes_with_data)} available nodes"
            )
        else:
            selected_node_ids = nodes_with_data
            print(f"Using all {len(selected_node_ids)} available nodes with data")

        # Create filtered adjacency matrix for selected nodes
        valid_indices = [
            node_ids.index(node_id)
            for node_id in selected_node_ids
            if node_id in node_ids
        ]
        valid_adj_matrix = adj_matrix[valid_indices][:, valid_indices]

        # Create filtered time series dictionary
        valid_time_series = {
            node_id: time_series_dict[node_id] for node_id in selected_node_ids
        }

        print(f"Successfully prepared data for {len(selected_node_ids)} nodes")
        print(f"Adjacency matrix shape: {valid_adj_matrix.shape}")

        return {
            "adj_matrix": valid_adj_matrix,
            "node_ids": selected_node_ids,
            "time_series_dict": valid_time_series,
        }

    except Exception as e:
        print(f"Error loading real data: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_synthetic_data(adj_matrix=None, node_ids=None, sample_size=8):
    """
    Generate synthetic data as a fallback, now with more nodes

    Parameters:
    -----------
    adj_matrix : np.ndarray, optional
        Pre-existing adjacency matrix
    node_ids : list, optional
        Pre-existing node IDs
    sample_size : int
        Number of sensor nodes to include if creating new synthetic data

    Returns:
    --------
    dict
        Dictionary containing adjacency matrix, node IDs, and synthetic time series data
    """
    print(f"Generating synthetic time series data with {sample_size} nodes...")

    # Create synthetic adjacency matrix and node IDs if not provided
    if adj_matrix is None or node_ids is None:
        # Create random node IDs
        node_ids = [f"1{str(i).zfill(4)}" for i in range(sample_size)]

        # Create random adjacency matrix
        adj_matrix = (
            np.random.rand(sample_size, sample_size) * 1000
        )  # Distances in meters
        adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(adj_matrix, 0)  # Zero diagonal

    # Create synthetic time series data
    time_series_dict = {}
    now = datetime.now()
    date_range = pd.date_range(start=now - timedelta(days=7), end=now, freq="15min")

    for node_id in node_ids:
        # Create a synthetic time series with some missing values and patterns
        base_pattern = np.sin(np.linspace(0, 14 * np.pi, len(date_range))) * 50 + 50

        # Add some random noise
        noise = np.random.normal(0, 5, len(date_range))

        # Add daily pattern
        hour_effect = np.array(
            [
                max(
                    0, 20 * np.sin((hour - 6) * np.pi / 12)
                )  # Peak at noon, low at midnight
                for hour in [d.hour for d in date_range]
            ]
        )

        values = base_pattern + noise + hour_effect

        # Introduce some NaN values to simulate missing data
        # Create patterns of missing data rather than random missing values
        mask = np.ones(len(date_range), dtype=bool)

        # Scenario 1: Missing data at night (midnight to 5am)
        night_hours = np.array([d.hour >= 0 and d.hour < 5 for d in date_range])
        night_missing = (
            np.random.random(len(date_range)) < 0.7
        )  # 70% chance of missing at night
        mask = mask & ~(night_hours & night_missing)

        # Scenario 2: Random data outages (blocks of consecutive missing values)
        num_outages = np.random.randint(3, 7)  # 3-6 outages
        for _ in range(num_outages):
            outage_start = np.random.randint(
                0, len(date_range) - 48
            )  # Ensure room for outage
            outage_length = np.random.randint(
                12, 48
            )  # 3-12 hour outage (at 15-min intervals)
            mask[outage_start : outage_start + outage_length] = False

        # Apply mask
        values[~mask] = np.nan

        time_series_dict[node_id] = pd.Series(values, index=date_range)

    print(
        f"Generated synthetic data for {len(node_ids)} nodes over {len(date_range)} time points"
    )
    return {
        "adj_matrix": adj_matrix,
        "node_ids": node_ids,
        "time_series_dict": time_series_dict,
    }


def get_most_complete_nodes(time_series_dict, num_nodes=8):
    """
    Find the nodes with the most complete data

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping node IDs to time series
    num_nodes : int
        Number of nodes to return

    Returns:
    --------
    list
        List of node IDs with the most complete data
    """
    # Calculate completeness for each node
    completeness = {}
    for node_id, series in time_series_dict.items():
        if series is not None:
            # Calculate percentage of non-NaN values
            completeness[node_id] = (~pd.isna(series)).mean()

    # Sort by completeness and return top N
    sorted_nodes = sorted(completeness.items(), key=lambda x: x[1], reverse=True)
    return [node_id for node_id, _ in sorted_nodes[:num_nodes]]
