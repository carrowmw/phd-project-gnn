# src/training/cross_validation.py
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from pathlib import Path
import os

from gnn_package.config import ExperimentConfig, get_config
from gnn_package.src.models.factory import create_model
from .base_trainer import BaseTrainer
from .trainers import TqdmTrainer

logger = logging.getLogger(__name__)

def run_cross_validation(
    data_package: Dict[str, Any],
    config: Optional[ExperimentConfig] = None,
    trainer_class=TqdmTrainer,
    save_dir: Optional[Union[str, Path]] = None,
    save_all_models: bool = False,
) -> Dict[str, Any]:
    """
    Run cross-validation training using the prepared data splits.

    Parameters:
    -----------
    data_package : Dict[str, Any]
        Dictionary containing training/validation data splits
    config : ExperimentConfig, optional
        Configuration object
    trainer_class : class, optional
        Trainer class to use (defaults to TqdmTrainer)
    save_dir : str or Path, optional
        Directory to save models and results
    save_all_models : bool
        Whether to save all fold models or just the best one

    Returns:
    --------
    Dict[str, Any]
        Cross-validation results
    """
    if config is None:
        config = get_config()

    # Check if data_package contains CV splits
    if "splits" not in data_package:
        raise ValueError("Data package must contain 'splits' key for cross-validation")

    splits = data_package["splits"]
    if not splits or not isinstance(splits, list):
        raise ValueError("Invalid splits format in data package")

    # Setup saving directory if provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    # Run training on each split
    cv_results = []
    best_val_loss = float('inf')
    best_model = None
    best_fold = -1

    for fold_idx, split in enumerate(splits):
        logger.info(f"Training fold {fold_idx+1}/{len(splits)}")

        # Get data loaders for this split
        train_loader = split.get("train_loader")
        val_loader = split.get("val_loader")

        if train_loader is None or val_loader is None:
            logger.warning(f"Missing data loaders in fold {fold_idx+1}, skipping")
            continue

        # Create a new model for each fold
        model = create_model(config)

        # Create trainer
        trainer = trainer_class(model, config)

        # Train model
        fold_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )

        # Save fold results
        fold_val_loss = fold_results["best_val_loss"]
        cv_results.append({
            "fold": fold_idx,
            "val_loss": fold_val_loss,
            "train_losses": fold_results["train_losses"],
            "val_losses": fold_results["val_losses"],
        })

        # Save model if requested
        if save_dir is not None:
            if save_all_models:
                fold_path = save_dir / f"model_fold_{fold_idx}.pth"
                trainer.save_model(fold_path)

            # Track best model
            if fold_val_loss < best_val_loss:
                best_val_loss = fold_val_loss
                best_model = model.state_dict().copy()
                best_fold = fold_idx

    # Save best model if we have one
    if save_dir is not None and best_model is not None:
        best_path = save_dir / "best_model.pth"
        torch.save(best_model, best_path)
        logger.info(f"Best model (fold {best_fold}) saved to {best_path}")

    # Calculate average metrics
    val_losses = [result["val_loss"] for result in cv_results]
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    logger.info(f"Cross-validation complete: {len(cv_results)} folds")
    logger.info(f"Mean validation loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
    logger.info(f"Best validation loss: {best_val_loss:.6f} (fold {best_fold})")

    # Return summarized results
    return {
        "fold_results": cv_results,
        "mean_val_loss": mean_val_loss,
        "std_val_loss": std_val_loss,
        "best_val_loss": best_val_loss,
        "best_fold": best_fold
    }