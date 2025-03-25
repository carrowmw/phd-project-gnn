# gnn_package/src/models/train_stgnn.py

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from tqdm import tqdm, trange

from gnn_package.src import preprocessing
from gnn_package.src.dataloaders import create_dataloader
from gnn_package.src.models.stgnn import create_stgnn_model, STGNNTrainer


def preprocess_data(
    data=None,
    data_file=None,
    graph_prefix="graph",
    window_size=24,
    horizon=6,
    stride=1,
    batch_size=32,
    sigma_squared=0.1,
    epsilon=0.5,
):
    """
    Load and preprocess graph and sensor data for training
    with support for varying window counts per sensor
    """
    print("Loading graph data...")

    adj_matrix, node_ids, metadata = preprocessing.load_graph_data(
        prefix=graph_prefix, return_df=False
    )

    # Compute graph weights using Gaussian kernel
    weighted_adj = preprocessing.compute_adjacency_matrix(
        adj_matrix, sigma_squared=sigma_squared, epsilon=epsilon
    )

    print(
        f"Loaded adjacency matrix of shape {adj_matrix.shape} with {len(node_ids)} nodes"
    )

    # Load sensor data if not provided
    if data is None:
        if data_file is None:
            raise ValueError(
                "Either data or data_file must be provided to load sensor data."
            )

        try:
            data = preprocessing.load_sensor_data(data_file)
        except FileNotFoundError as e:
            print(e)
            print("Please run 'python fetch_sensor_data.py' first to fetch the data.")
            sys.exit(1)

    resampled_data = preprocessing.resample_sensor_data(
        data, freq="15min", fill_value=-1.0
    )

    # Create windows for time series
    print(f"Creating windows with size={window_size}, horizon={horizon}...")
    processor = preprocessing.TimeSeriesPreprocessor(
        window_size=window_size,
        stride=stride,
        gap_threshold=pd.Timedelta(minutes=15),
        missing_value=-1.0,
    )

    X_by_sensor, masks_by_sensor, metadata_by_sensor = processor.create_windows(
        resampled_data
    )

    # Get list of sensors with valid windows
    valid_sensors = list(X_by_sensor.keys())
    print(f"Found {len(valid_sensors)} sensors with valid windows")

    if len(valid_sensors) == 0:
        raise ValueError(
            "No valid windows found! Try increasing days_back or decreasing window_size."
        )

    # Create a smaller adjacency matrix for only the valid sensors
    valid_indices = [node_ids.index(sid) for sid in valid_sensors if sid in node_ids]
    valid_adj = weighted_adj[valid_indices, :][:, valid_indices]
    valid_node_ids = [node_ids[idx] for idx in valid_indices]

    # For each sensor, split its data into train and validation
    X_train_by_sensor = {}
    X_val_by_sensor = {}
    masks_train_by_sensor = {}
    masks_val_by_sensor = {}

    # Add progress bar for splitting data
    for node_id in tqdm(
        valid_sensors, desc="Splitting data into train/validation sets"
    ):
        n_windows = len(X_by_sensor[node_id])
        train_size = int(n_windows * 0.8)

        X_train_by_sensor[node_id] = X_by_sensor[node_id][:train_size]
        X_val_by_sensor[node_id] = X_by_sensor[node_id][train_size:]

        masks_train_by_sensor[node_id] = masks_by_sensor[node_id][:train_size]
        masks_val_by_sensor[node_id] = masks_by_sensor[node_id][train_size:]

    # Calculate total windows
    total_train = sum(len(windows) for windows in X_train_by_sensor.values())
    total_val = sum(len(windows) for windows in X_val_by_sensor.values())
    print(f"Total training windows: {total_train}")
    print(f"Total validation windows: {total_val}")

    print("Creating dataloaders...")
    # Create dataloaders with the updated implementation
    train_loader = create_dataloader(
        X_train_by_sensor,
        masks_train_by_sensor,
        valid_adj,
        valid_node_ids,
        window_size,
        horizon,
        batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        X_val_by_sensor,
        masks_val_by_sensor,
        valid_adj,
        valid_node_ids,
        window_size,
        horizon,
        batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "adj_matrix": valid_adj,
        "node_ids": valid_node_ids,
    }


class TqdmSTGNNTrainer(STGNNTrainer):
    """
    Extension of STGNNTrainer that adds progress bars using tqdm
    """

    def train_epoch(self, dataloader):
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Create progress bar for batches
        pbar = tqdm(dataloader, desc="Training batches", leave=False)

        for batch in pbar:
            # Move data to device
            x = batch["x"].to(self.device)
            x_mask = batch["x_mask"].to(self.device)
            y = batch["y"].to(self.device)
            y_mask = batch["y_mask"].to(self.device)
            adj = batch["adj"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x, adj, x_mask)

            # Compute loss on valid points only
            loss = self.criterion(y_pred, y)
            if y_mask is not None:
                # Count non-zero elements in mask
                mask_sum = y_mask.sum()
                if mask_sum > 0:
                    loss = (loss * y_mask).sum() / mask_sum
                else:
                    loss = torch.tensor(0.0, device=self.device)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1

            # Update progress bar with current batch loss
            pbar.set_postfix({"batch_loss": f"{batch_loss:.6f}"})

        return total_loss / max(1, num_batches)

    def evaluate(self, dataloader):
        """Evaluate the model on a validation or test set with progress bar"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            # Create progress bar for validation batches
            pbar = tqdm(dataloader, desc="Validation batches", leave=False)

            for batch in pbar:
                # Move data to device
                x = batch["x"].to(self.device)
                x_mask = batch["x_mask"].to(self.device)
                y = batch["y"].to(self.device)
                y_mask = batch["y_mask"].to(self.device)
                adj = batch["adj"].to(self.device)

                # Forward pass
                y_pred = self.model(x, adj, x_mask)

                # Compute loss on valid points only
                loss = self.criterion(y_pred, y)
                if y_mask is not None:
                    # Count non-zero elements in mask
                    mask_sum = y_mask.sum()
                    if mask_sum > 0:
                        loss = (loss * y_mask).sum() / mask_sum
                    else:
                        loss = torch.tensor(0.0, device=self.device)

                batch_loss = loss.item()
                total_loss += batch_loss
                num_batches += 1

                # Update progress bar with current batch loss
                pbar.set_postfix({"batch_loss": f"{batch_loss:.6f}"})

        return total_loss / max(1, num_batches)


def train_model(
    data_loaders,
    input_dim=1,
    hidden_dim=64,
    output_dim=1,
    horizon=6,
    learning_rate=0.001,
    weight_decay=1e-5,
    num_epochs=50,
    patience=10,
):
    """
    Train the STGNN model with progress bars

    Parameters:
    -----------
    data_loaders : dict
        Dict containing train_loader and val_loader
    input_dim : int
        Input dimension (number of features per node)
    hidden_dim : int
        Hidden dimension for model
    output_dim : int
        Output dimension (number of features to predict)
    horizon : int
        Number of future time steps to predict
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay for regularization
    num_epochs : int
        Maximum number of epochs to train
    patience : int
        Number of epochs to wait for improvement before early stopping

    Returns:
    --------
    Trained model and training metrics
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = create_stgnn_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        horizon=horizon,
    )

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Mean Squared Error loss
    criterion = torch.nn.MSELoss(reduction="none")

    # Create trainer with tqdm support
    trainer = TqdmSTGNNTrainer(model, optimizer, criterion, device)

    # Training loop with early stopping and overall progress bar
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model = None
    no_improve_count = 0

    # Use trange for overall epoch progress
    epochs_pbar = trange(num_epochs, desc="Training progress")

    for epoch in epochs_pbar:
        # Train
        train_loss = trainer.train_epoch(data_loaders["train_loader"])
        train_losses.append(train_loss)

        # Validate
        val_loss = trainer.evaluate(data_loaders["val_loader"])
        val_losses.append(val_loss)

        # Update progress bar with current metrics
        epochs_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "no_improve": no_improve_count,
            }
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.tight_layout()

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }


def predict_and_evaluate(model, dataloader, device=None):
    """
    Make predictions with the trained model and evaluate performance

    Parameters:
    -----------
    model : STGNN
        Trained model
    dataloader : DataLoader
        Dataloader containing test data
    device : torch.device
        Device to use for inference

    Returns:
    --------
    Dict containing predictions and evaluation metrics
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        # Add progress bar for evaluation
        for batch in tqdm(dataloader, desc="Evaluating model"):
            # Move data to device
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            y = batch["y"].to(device)
            y_mask = batch["y_mask"].to(device)
            adj = batch["adj"].to(device)

            # Forward pass
            y_pred = model(x, adj, x_mask)

            # Move predictions and targets to CPU
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(y_mask.cpu().numpy())

    # Concatenate batches
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Compute metrics on valid points only
    mse = np.mean(((predictions - targets) ** 2) * masks) / np.mean(masks)
    mae = np.mean(np.abs(predictions - targets) * masks) / np.mean(masks)

    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")

    return {
        "predictions": predictions,
        "targets": targets,
        "masks": masks,
        "mse": mse,
        "mae": mae,
    }


def save_model(model, file_path):
    """Save the trained model"""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
