# gnn_package/src/training/stgnn_training.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from gnn_package.src import preprocessing
from gnn_package.src.models.stgnn import create_stgnn_model, STGNNTrainer
from gnn_package.config import get_config
from gnn_package.src.data.processors import DataProcessorFactory, ProcessorMode
from gnn_package.src.data.data_sources import FileDataSource
from gnn_package.src.utils.data_utils import validate_data_package
from gnn_package.src.utils.config_utils import save_model_with_config

# gnn_package/src/training/stgnn_training.py


async def preprocess_data(
    data=None, data_file=None, config=None, mode=None, verbose=True
):
    """Load and preprocess graph and sensor data for training."""
    print("Training.preprocess_data: Starting preprocessing")

    # Get configuration
    if config is None:
        if verbose:
            print("No configuration explicitly provided, using global config...")
        config = get_config(verbose=verbose)

    # Determine mode
    if mode is None:
        # Default to training
        processor_mode = ProcessorMode.TRAINING
    else:
        processor_mode = ProcessorMode(mode)

    print(f"Using processor mode: {processor_mode}")

    # Create data source based on inputs
    if data is not None:
        print("Using provided data")

        # Custom in-memory data source using the provided data
        class CustomDataSource(FileDataSource):
            async def get_data(self, config):
                return data

        data_source = CustomDataSource(None)
    elif data_file is not None:
        print(f"Using data file: {data_file}")
        data_source = FileDataSource(data_file)
    else:
        print("WARNING: No data or data_file provided")
        data_source = None  # Will be created by factory if needed

    # Create processor using factory
    print("Creating data processor...")
    processor = DataProcessorFactory.create_processor(
        mode=processor_mode, config=config, data_source=data_source
    )

    print(f"Processor created: {type(processor).__name__}")

    # Process data according to mode
    try:
        print("Calling processor.process_data()...")
        result = await processor.process_data()
        print(f"Processor returned: {type(result)}")
        if result is None:
            print("WARNING: processor.process_data() returned None!")
        return result
    except Exception as e:
        print(f"ERROR in processor.process_data(): {e}")
        import traceback

        traceback.print_exc()
        raise


class TqdmSTGNNTrainer(STGNNTrainer):
    """
    Extension of STGNNTrainer that adds progress bars using tqdm
    """

    def __init__(self, model, config):
        """
        Initialize the trainer with model and configuration.

        Parameters:
        -----------
        model : STGNN
            The model to train
        config : ExperimentConfig
            Configuration object
        """
        # Call the parent class constructor to handle device, optimizer, and criterion
        super().__init__(model, config)

        # Additional tqdm-specific initialization can go here if needed
        self.log_interval = getattr(config.training, "log_interval", 10)

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
    data_package,
    config=None,
):
    """
    Train the STGNN model with progress bars

    Parameters:
    -----------
    data_loaders : dict
        Dict containing train_loader and val_loader
    config : ExperimentConfig, optional
        Centralized configuration object. If not provided, will use global config.
    **kwargs : dict
        Additional parameters to override config settings

    Returns:
    --------
    dict
        Dictionary containing trained model and training metrics
    """

    # Get configuration
    if config is None:
        config = get_config()

        # Validate data package
    validate_data_package(
        data_package,
        required_components=["train_loader", "val_loader"],
        mode="training",
    )

    # Extract components for use
    train_loader = data_package["data_loaders"]["train_loader"]
    val_loader = data_package["data_loaders"]["val_loader"]

    num_epochs = config.training.num_epochs
    patience = config.training.patience

    # Determine device (use config or auto-detect)
    if config.training.device:
        device = torch.device(config.training.device)
    else:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    print(f"Using device: {device}")

    # Create model
    model = create_stgnn_model(config)

    # Create trainer with tqdm support
    trainer = TqdmSTGNNTrainer(model, config)

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
        train_loss = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)

        # Validate
        val_loss = trainer.evaluate(val_loader)
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

    # Save model and configuration together
    if hasattr(config.paths, "model_save_path") and config.paths.model_save_path:

        model_dir = (
            config.paths.model_save_path / f"{config.experiment.name.replace(' ', '_')}"
        )
        save_model_with_config(model, config, model_dir)
        print(f"Model and configuration saved to {model_dir}")

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


def cross_validate_model(data=None, data_file=None, config=None):
    """
    Train and evaluate model using time-based cross-validation.

    Parameters:
    -----------
    data : dict, optional
        Dictionary mapping sensor IDs to their time series data
    data_file : str, optional
        Path to a pickled file containing sensor data
    config : ExperimentConfig, optional
        Centralized configuration object
    Returns:
    --------
    dict
        Dictionary containing cross-validation results
    """
    # Get configuration
    if config is None:
        config = get_config()

    # Allow override of config parameters with kwargs
    n_splits = config.data.training.n_splits

    # Load sensor data if not provided
    if data is None:
        if data_file is None:
            raise ValueError("Either data or data_file must be provided")
        data = preprocessing.load_sensor_data(data_file)

    # Resample data to consistent frequency
    resampled_data = preprocessing.resample_sensor_data(
        data,
    )

    # Create processor with appropriate settings
    processor = preprocessing.TimeSeriesPreprocessor()

    # Create rolling window splits
    print(f"Creating {n_splits} time-based cross-validation splits...")
    splits = processor.create_rolling_window_splits(
        resampled_data,
        config=config,
    )

    print(f"Generated {len(splits)} valid split(s)")

    # Train and evaluate on each split
    results = []
    for i, split in enumerate(tqdm(splits, desc="Training on CV splits")):
        print(f"\nTraining on split {i+1}/{len(splits)}")

        # Process this split into windows
        split_data_loaders = preprocess_data(
            data=split,  # Pass the split directly
            config=config,
        )

        # Train on this split
        split_results = train_model(
            data_package=split_data_loaders,
            config=config,
        )

        # Save results for this split
        results.append(
            {
                "split": i,
                "train_losses": split_results["train_losses"],
                "val_losses": split_results["val_losses"],
                "best_val_loss": split_results["best_val_loss"],
            }
        )

    # Calculate overall metrics
    best_val_losses = [r["best_val_loss"] for r in results]

    return {
        "results": results,
        "mean_val_loss": np.mean(best_val_losses),
        "std_val_loss": np.std(best_val_losses),
        "min_val_loss": np.min(best_val_losses),
        "max_val_loss": np.max(best_val_losses),
    }


def save_model(model, file_path):
    """Save the trained model"""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def train_model_with_cv(data_loaders, config=None):
    """Train model with cross-validation support."""

    if "train_loaders" in data_loaders:  # Cross-validation mode
        cv_results = []

        for i, (train_loader, val_loader) in enumerate(
            zip(data_loaders["train_loaders"], data_loaders["val_loaders"])
        ):

            print(f"Training on split {i+1}/{len(data_loaders['train_loaders'])}")

            # Create a new model for each split
            model = create_stgnn_model(config=config)

            # Train on this split
            split_results = train_model(
                {
                    "data_loader": {
                        "train_loader": train_loader,
                        "val_loader": val_loader,
                    }
                },
                config=config,
            )

            cv_results.append(split_results)

        # Aggregate results across splits
        avg_val_loss = sum(r["best_val_loss"] for r in cv_results) / len(cv_results)

        return {
            "cv_results": cv_results,
            "avg_val_loss": avg_val_loss,
            "best_model_index": np.argmin([r["best_val_loss"] for r in cv_results]),
        }

    else:  # Standard single split mode
        return train_model(data_loaders, config=config)
