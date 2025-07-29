# src/training/trainers.py
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List

from gnn_package.config import ExperimentConfig
from gnn_package.src.utils.device_utils import get_device_from_config
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class TqdmTrainer(BaseTrainer):
    """
    Trainer with tqdm progress bars for better visualizations.
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
        """Evaluate the model with progress bar"""
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

    def train(self, train_loader, val_loader, num_epochs=None, patience=None):
        """Train model with early stopping and fancy progress bar"""
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

        # Progress bar for epochs
        epochs = trange(num_epochs, desc="Training")

        for epoch in epochs:
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            # Evaluate on validation set
            val_loss = self.evaluate(val_loader)
            val_losses.append(val_loss)

            # Update progress bar
            epochs.set_postfix({
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "no_improve": no_improve_count
            })

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_count = 0
                epochs.set_description(f"Training (new best: {best_val_loss:.6f})")
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= patience:
                epochs.set_description(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)

        return {
            "model": self.model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }


class MinimalTrainer(BaseTrainer):
    """
    Trainer without progress bars, suitable for production or scripted runs.
    """

    def train(self, train_loader, val_loader, num_epochs=None, patience=None):
        """Train model with early stopping, with minimal output"""
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

            # Log only every N epochs to reduce verbosity
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)

        logger.info(f"Training complete. Best validation loss: {best_val_loss:.6f}")

        return {
            "model": self.model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }