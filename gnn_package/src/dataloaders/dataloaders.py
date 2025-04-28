# gnn_package/src/preprocessing/dataloaders.py

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Set up logging
logger = logging.getLogger(__name__)

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
        """Return the number of windows (time steps)."""
        # Find the sensor with the minimum number of windows
        min_windows = min(len(windows) for windows in self.X_by_sensor.values())
        return min_windows

    # Works with the original TimeSeriesPreprocessor segmented windows
    # def __getitem__(self, idx):
    #     # Get the node_id and window_idx for this sample
    #     node_id, window_idx = self.sample_indices[idx]

    #     # Get node index in adjacency matrix
    #     node_idx = self.node_ids.index(node_id)

    #     # Get input window (history) and target window (future)
    #     x_window = self.X_by_sensor[node_id][
    #         window_idx, : self.window_size - self.horizon
    #     ]
    #     x_mask = self.masks_by_sensor[node_id][
    #         window_idx, : self.window_size - self.horizon
    #     ]

    #     y_window = self.X_by_sensor[node_id][window_idx, -self.horizon :]
    #     y_mask = self.masks_by_sensor[node_id][window_idx, -self.horizon :]

    #     return {
    #         "x": torch.FloatTensor(x_window),
    #         "x_mask": torch.FloatTensor(x_mask),
    #         "y": torch.FloatTensor(y_window),
    #         "y_mask": torch.FloatTensor(y_mask),
    #         "node_idx": node_idx,
    #         "adj": self.adj_matrix,
    #     }

    def __getitem__(self, idx):
        """
        Get data for window index idx across all sensors.

        Returns all sensors' data for this window to represent a system snapshot.
        """
        # idx now represents a window index, not a (node_id, window_idx) pair
        window_idx = idx

        # Create tensors for all nodes at this window idx
        x_windows = []
        x_masks = []
        y_windows = []
        y_masks = []
        node_indices = []

        for i, node_id in enumerate(self.node_ids):
            if node_id in self.X_by_sensor and window_idx < len(
                self.X_by_sensor[node_id]
            ):
                # Get input window and masks
                x_window = self.X_by_sensor[node_id][
                    window_idx, : self.window_size - self.horizon
                ]
                x_mask = self.masks_by_sensor[node_id][
                    window_idx, : self.window_size - self.horizon
                ]

                # Get target window and masks
                y_window = self.X_by_sensor[node_id][window_idx, -self.horizon :]
                y_mask = self.masks_by_sensor[node_id][window_idx, -self.horizon :]

                x_windows.append(torch.FloatTensor(x_window))
                x_masks.append(torch.FloatTensor(x_mask))
                y_windows.append(torch.FloatTensor(y_window))
                y_masks.append(torch.FloatTensor(y_mask))
                node_indices.append(i)

        # Stack into tensors [num_nodes, seq_len]
        x = torch.stack(x_windows)
        x_mask = torch.stack(x_masks)
        y = torch.stack(y_windows)
        y_mask = torch.stack(y_masks)

        return {
            "x": x,
            "x_mask": x_mask,
            "y": y,
            "y_mask": y_mask,
            "node_indices": torch.tensor(node_indices),
            "adj": self.adj_matrix,
        }


def collate_fn(batch):
    """
    Custom collate function for batching system snapshots.
    Each item in the batch already contains all sensors for a specific time window.
    """
    # Extract tensors from batch
    x = torch.stack([item["x"] for item in batch])
    x_mask = torch.stack([item["x_mask"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    y_mask = torch.stack([item["y_mask"] for item in batch])

    # Use the first item's adjacency matrix and node indices
    adj = batch[0]["adj"]
    node_indices = batch[0]["node_indices"]

    return {
        "x": x,  # [batch_size, num_nodes, seq_len, 1]
        "x_mask": x_mask,  # [batch_size, num_nodes, seq_len, 1]
        "y": y,  # [batch_size, num_nodes, horizon, 1]
        "y_mask": y_mask,  # [batch_size, num_nodes, horizon, 1]
        "node_indices": node_indices,
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
    Create a DataLoader that can handle varying numbers of windows per sensor.
    """
    # Check if we have any data to work with
    if not X_by_sensor or all(len(windows) == 0 for windows in X_by_sensor.values()):
        logger.error("No valid windows to create dataset - check data or date range")
        raise ValueError("No valid windows available to create dataset")

    dataset = SpatioTemporalDataset(
        X_by_sensor, masks_by_sensor, adj_matrix, node_ids, window_size, horizon
    )

    # Prevent creating dataloader with empty dataset
    if len(dataset.sample_indices) == 0:
        logger.error("Dataset has no samples - check date range and data availability")
        raise ValueError("Dataset created with no samples")


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    return dataloader
