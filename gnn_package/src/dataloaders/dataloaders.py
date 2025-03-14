# gnn_package/src/preprocessing/dataloaders.py

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

        # Create index mapping for faster access
        self.sample_indices = []
        for node_id, windows in X_by_sensor.items():
            node_idx = node_ids.index(node_id)
            for window_idx in range(len(windows)):
                self.sample_indices.append((node_id, node_idx, window_idx))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        node_id, window_idx, node_idx = self.sample_indices[idx]

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

        return {
            "x": torch.FloatTensor(x_window),
            "x_mask": torch.FloatTensor(x_mask),
            "y": torch.FloatTensor(y_window),
            "y_mask": torch.FloatTensor(y_mask),
            "node_idx": node_idx,
            "adj": self.adj_matrix,  # static graph (same for all samples)
        }


def collate_fn(batch):
    """
    Custom collate function to handle graph data.
    """

    # Extract items
    x = torch.stack([item["x"] for item in batch])
    x_mask = torch.stack([item["x_mask"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    y_mask = torch.stack([item["y_mask"] for item in batch])
    node_indices = torch.tensor([item["node_idx"] for item in batch])

    # All samples have the same adjacency matrix
    adj = batch[0]["adj"]

    return {
        "x": x,
        "x_mask": x_mask,
        "y": y,
        "y_mask": y_mask,
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
