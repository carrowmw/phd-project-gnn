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

        # Determine the number of windows available for each node
        self.window_counts = {
            node_id: len(windows) for node_id, windows in X_by_sensor.items()
        }
        self.min_windows = min(self.window_counts.values()) if self.window_counts else 0

        # Create a mapping of window indices
        self.window_indices = list(range(self.min_windows))

        # # Create index mapping for faster access
        # self.sample_indices = []
        # for node_id, windows in X_by_sensor.items():
        #     node_idx = node_ids.index(node_id)
        #     for window_idx in range(len(windows)):
        #         self.sample_indices.append((node_id, node_idx, window_idx))

    def __len__(self):
        return (
            self.min_windows
        )  # Each batch will contian all nodes for specific window index

    def __getitem__(self, window_idx):
        """
        Returns data for all nodes at a specific window index.
        """
        # Get data for all nodes at this window index
        x_windows = []
        x_masks = []
        y_windows = []
        y_masks = []
        node_indices = []

        for i, node_id in enumerate(self.node_ids):
            if node_id in self.X_by_sensor:
                # Get input window (history)
                x_window = self.X_by_sensor[node_id][
                    window_idx, : self.window_size - self.horizon
                ]
                x_mask = self.masks_by_sensor[node_id][
                    window_idx, : self.window_size - self.horizon
                ]

                # Get target window (future)
                y_window = self.X_by_sensor[node_id][window_idx, -self.horizon :]
                y_mask = self.masks_by_sensor[node_id][window_idx, -self.horizon :]

                # Add to lists
                x_windows.append(x_window)
                x_masks.append(x_mask)
                y_windows.append(y_window)
                y_masks.append(y_mask)
                node_indices.append(i)

        # Stack all node data
        x = torch.FloatTensor(np.array(x_windows))
        x_mask = torch.FloatTensor(np.array(x_masks))
        y = torch.FloatTensor(np.array(y_windows))
        y_mask = torch.FloatTensor(np.array(y_masks))

        return {
            "x": x,
            "x_mask": x_mask,
            "y": y,
            "y_mask": y_mask,
            "node_idx": node_indices,
            "adj": self.adj_matrix,
        }


def collate_fn(batch):
    """
    Custom collate function to handle graph data.
    """
    # With the modified dataset, each batch item already contains all nodes
    # We just need to add the batch dimension and ensure 4D shape

    # Stack batch items
    x = torch.stack([item["x"] for item in batch])  # [batch, num_nodes, seq_len]
    x_mask = torch.stack([item["x_mask"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    y_mask = torch.stack([item["y_mask"] for item in batch])

    print(f"DEBUG: Raw x shape: {x.shape}")
    print(f"DEBUG: Raw x_mask shape: {x_mask.shape}")

    # Add feature dimension if needed
    if len(x.shape) == 3:  # [batch, num_nodes, seq_len]
        x = x.unsqueeze(-1)  # [batch, num_nodes, seq_len, 1]
    if len(x_mask.shape) == 3:
        x_mask = x_mask.unsqueeze(-1)
    if len(y.shape) == 3:
        y = y.unsqueeze(-1)
    if len(y_mask.shape) == 3:
        y_mask = y_mask.unsqueeze(-1)

    # Make sure mask has the same shape as input
    if x_mask.shape != x.shape:
        print(
            f"WARNING: Mask shape {x_mask.shape} does not match input shape {x.shape}"
        )
        # Reshape mask to match input shape
        x_mask = x_mask[:, :, : x.shape[2], :]

    # print(f"DEBUG: Processed x shape: {x.shape}")
    # print(f"DEBUG: Processed x_mask shape: {x_mask.shape}")
    # print(f"DEBUG: Processed y shape: {y.shape}")
    # print(f"DEBUG: Processed y_mask shape: {y_mask.shape}")

    # All samples have the same adjacency matrix
    adj = batch[0]["adj"]

    print(f"DEBUG: Batch shape: {x.shape}")  # Should be [batch, num_nodes, seq_len, 1]

    return {
        "x": x,
        "x_mask": x_mask,
        "y": y,
        "y_mask": y_mask,
        "adj": adj,
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
    Create a DataLoader for the SpatioTemporalDataset.
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
