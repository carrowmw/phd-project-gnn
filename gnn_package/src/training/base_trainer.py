# src/training/base_trainer.py
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Union

from gnn_package.config import ExperimentConfig
from gnn_package.src.utils.device_utils import get_device_from_config
from gnn_package.src.utils.exceptions import ValidationError, EarlyStoppingException

logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Base trainer class for all model architectures.
    """

    def __init__(self, model: nn.Module, config: ExperimentConfig):
        """Initialize the trainer with model and configuration."""
        self.config = config

        # Get device
        self.device = get_device_from_config(config)
        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = model.to(self.device)

        # Create optimizer based on config
        learning_rate = config.training.learning_rate
        weight_decay = config.training.weight_decay

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function - MSE with reduction='none' to handle masks
        self.criterion = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none')

        logger.info(f"Trainer initialized with {model.__class__.__name__} model")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")

    def train_epoch(self, dataloader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
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

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def evaluate(self, dataloader) -> float:
        """Evaluate model on validation data and return average loss."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
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
                    mask_sum = y_mask.sum()
                    if mask_sum > 0:
                        loss = (loss * y_mask).sum() / mask_sum
                    else:
                        loss = torch.tensor(0.0, device=self.device)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def train(self, train_loader, val_loader, num_epochs=None, patience=None):
        """Train model with early stopping and return training results."""
        # Use config values if not specified
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        if patience is None:
            patience = self.config.training.patience

        logger.info(f"Training for {num_epochs} epochs with patience {patience}")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model = None
        no_improve_count = 0

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            # Evaluate on validation set
            val_loss = self.evaluate(val_loader)
            val_losses.append(val_loss)

            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train loss: {train_loss:.6f}, "
                      f"Val loss: {val_loss:.6f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_count = 0
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
            else:
                no_improve_count += 1
                logger.info(f"No improvement for {no_improve_count} epochs")

            # Early stopping
            if no_improve_count >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
        else:
            logger.warning("No best model state found - using current model state")

        return {
            "model": self.model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

    def save_model(self, path):
        """Save model state dict to the specified path."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")