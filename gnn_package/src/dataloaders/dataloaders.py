# gnn_package/src/preprocessing/dataloaders.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SpatioTemporalDataset(Dataset):
    def __init__(
        self,
        X_by_sensor,
        masks_by_sensor,
        adj_matrix,
        node_ids,
        window_size,
        horizon,
    ):
        """
        Parameters:
        ----------
        X_by_sensor : Dict[str, np.ndarray]
            Dictionary containing the input data for each sensor.
        masks_by_sensor : Dict[str, np.ndarray]
            Dictionary containing the masks for each sensor.
        adj_matrix : np.ndarray
            Adjacency matrix of the graph.
        node_ids : List[str]
            List of node IDs.
        window_size : int
            Size of the input window.
        horizon : int
            Number of time steps to predict ahead.
        """
        self.X_by_sensor = X_by_sensor
        self.masks_by_sensor = masks_by_sensor
        self.adj_matrix = torch.FloatTensor(adj_matrix)
        self.node_ids = node_ids
        self.window_size = window_size
        self.horizon = horizon

        # Create flattened index mapping (node_id, window_idx)
        self.sample_indices = []
        for node_id in self.node_ids:
            if node_id in X_by_sensor:
                windows = X_by_sensor[node_id]
                for window_idx in range(len(windows)):
                    self.sample_indices.append((node_id, window_idx))

        print(
            f"Created dataset with {len(self.sample_indices)} total samples across {len(node_ids)} nodes"
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        # Get the node_id and window_idx for this sample
        node_id, window_idx = self.sample_indices[idx]

        # Get node index in adjacency matrix
        node_idx = self.node_ids.index(node_id)

        # Get input window (history) and target window (future)
        x_window = self.X_by_sensor[node_id][
            window_idx, : self.window_size - self.horizon
        ]
        x_mask = self.masks_by_sensor[node_id][
            window_idx, : self.window_size - self.horizon
        ]

        y_window = self.X_by_sensor[node_id][window_idx, -self.horizon :]
        y_mask = self.masks_by_sensor[node_id][window_idx, -self.horizon :]

        return {
            "x": torch.FloatTensor(x_window),
            "x_mask": torch.FloatTensor(x_mask),
            "y": torch.FloatTensor(y_window),
            "y_mask": torch.FloatTensor(y_mask),
            "node_idx": node_idx,
            "adj": self.adj_matrix,
        }


def collate_fn(batch):
    """
    Custom collate function that creates batches of multiple time windows,
    each containing data for all nodes present in the batch.

    Parameters:
    ----------
    batch : List[Dict]
        List of samples from the dataset.

    Returns:
    -------
    Dict
        x : Tensor [batch_size, num_nodes, seq_len, 1]
        x_mask : Tensor [batch_size, num_nodes, seq_len, 1]
        y : Tensor [batch_size, num_nodes, horizon, 1]
        y_mask : Tensor [batch_size, num_nodes, horizon, 1]
        node_indices : Tensor [num_nodes]
        adj : Tensor [num_nodes, num_nodes]
    """
    # Get all unique node indices in this batch
    all_node_indices = sorted(list(set(item["node_idx"] for item in batch)))
    node_idx_map = {idx: i for i, idx in enumerate(all_node_indices)}

    # Get window dimensions
    seq_len = len(batch[0]["x"])
    horizon = len(batch[0]["y"])

    # Group samples by window_idx
    # We'll use the relative position in the batch to create window groups
    # This way, we'll create multiple windows in a batch
    max_windows_per_batch = 32  # Maximum windows in a batch
    window_groups = {}

    for i, item in enumerate(batch):
        # Assign a window group based on position in batch
        window_group = i % max_windows_per_batch
        if window_group not in window_groups:
            window_groups[window_group] = []
        window_groups[window_group].append(item)

    # Calculate batch dimensions
    batch_size = len(window_groups)
    num_nodes = len(all_node_indices)

    print(f"Creating batch with dimensions: {batch_size} windows, {num_nodes} nodes")

    # Initialize tensors with proper dimensions
    x = torch.full((batch_size, num_nodes, seq_len, 1), -1.0)
    x_mask = torch.zeros((batch_size, num_nodes, seq_len, 1))
    y = torch.full((batch_size, num_nodes, horizon, 1), -1.0)
    y_mask = torch.zeros((batch_size, num_nodes, horizon, 1))

    # Fill tensors
    for batch_idx, items in enumerate(window_groups.values()):
        for item in items:
            node_pos = node_idx_map[item["node_idx"]]

            # Add data and masks (add feature dimension)
            x[batch_idx, node_pos, :, 0] = item["x"]
            x_mask[batch_idx, node_pos, :, 0] = item["x_mask"]
            y[batch_idx, node_pos, :, 0] = item["y"]
            y_mask[batch_idx, node_pos, :, 0] = item["y_mask"]

    # Extract adjacency for these specific nodes
    adj = batch[0]["adj"]
    batch_adj = adj[all_node_indices][:, all_node_indices]

    return {
        "x": x,
        "x_mask": x_mask,
        "y": y,
        "y_mask": y_mask,
        "node_indices": torch.tensor(all_node_indices),
        "adj": batch_adj,
    }


def create_dataloader(
    X_by_sensor,
    masks_by_sensor,
    adj_matrix,
    node_ids,
    window_size,
    horizon,
    batch_size,
    shuffle,
):
    """
    Create a DataLoader that can handle varying numbers of windows per sensor.
    """
    dataset = SpatioTemporalDataset(
        X_by_sensor, masks_by_sensor, adj_matrix, node_ids, window_size, horizon
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    return dataloader
