# src/training/experiment_manager.py
import os
import json
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn_package.config import ExperimentConfig, get_config
from gnn_package.src.models.factory import create_model
from gnn_package.src.utils.model_io import save_model
from .trainers import TqdmTrainer
from .cross_validation import run_cross_validation

logger = logging.getLogger(__name__)

async def run_experiment(
    data_package: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[ExperimentConfig] = None,
    use_cross_validation: Optional[bool] = None,
    save_model_checkpoints: bool = True,
    plot_results: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete training experiment with comprehensive output.

    Parameters:
    -----------
    data_package : Dict[str, Any]
        Preprocessed data package with loaders and metadata
    output_dir : str or Path, optional
        Directory to save experiment outputs
    config : ExperimentConfig, optional
        Configuration object
    use_cross_validation : bool, optional
        Whether to use cross-validation (overrides config setting)
    save_model_checkpoints : bool
        Whether to save intermediate model checkpoints
    plot_results : bool
        Whether to generate and save plots

    Returns:
    --------
    Dict[str, Any]
        Experiment results
    """
    # Setup configuration
    if config is None:
        config = get_config()

    # Determine if cross-validation should be used
    if use_cross_validation is None:
        use_cross_validation = config.data.training.use_cross_validation

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/{config.experiment.name.replace(' ', '_')}_{timestamp}")
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.yml"
    config.save(config_path)
    logger.info(f"Configuration saved to {config_path}")

    # Initialize results dictionary
    experiment_results = {
        "experiment_name": config.experiment.name,
        "timestamp": datetime.now().isoformat(),
        "architecture": config.model.architecture,
        "output_dir": str(output_dir),
    }

    # Run training (with or without cross-validation)
    if use_cross_validation:
        logger.info("Running experiment with cross-validation")
        cv_results = run_cross_validation(
            data_package=data_package,
            config=config,
            trainer_class=TqdmTrainer,
            save_dir=output_dir / "cv_models",
            save_all_models=save_model_checkpoints,
        )
        experiment_results["cv_results"] = cv_results

        # Train final model on all data if requested
        if config.data.training.train_final_model:
            logger.info("Training final model on all data")
            train_loader = data_package["data_loaders"]["train_loader"]
            val_loader = data_package["data_loaders"]["val_loader"]

            model = create_model(config)
            trainer = TqdmTrainer(model, config)

            final_results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader
            )

            # Save final model
            model_path = output_dir / "model.pth"
            torch.save(model.state_dict(), model_path)

            experiment_results["final_model"] = {
                "train_losses": final_results["train_losses"],
                "val_losses": final_results["val_losses"],
                "best_val_loss": final_results["best_val_loss"],
                "model_path": str(model_path),
            }
    else:
        logger.info("Running experiment with single train/val split")
        train_loader = data_package["data_loaders"]["train_loader"]
        val_loader = data_package["data_loaders"]["val_loader"]

        model = create_model(config)
        trainer = TqdmTrainer(model, config)

        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )

        # Save model
        model_path = output_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

        experiment_results["training_results"] = {
            "train_losses": training_results["train_losses"],
            "val_losses": training_results["val_losses"],
            "best_val_loss": training_results["best_val_loss"],
            "model_path": str(model_path),
        }

    # Generate and save plots if requested
    if plot_results:
        if use_cross_validation:
            if "final_model" in experiment_results:
                _plot_learning_curves(
                    experiment_results["final_model"]["train_losses"],
                    experiment_results["final_model"]["val_losses"],
                    output_dir / "final_model_learning_curve.png"
                )

            # Plot CV results
            _plot_cv_results(experiment_results["cv_results"], output_dir / "cv_results.png")
        else:
            _plot_learning_curves(
                experiment_results["training_results"]["train_losses"],
                experiment_results["training_results"]["val_losses"],
                output_dir / "learning_curve.png"
            )

    # Save experiment results
    results_path = output_dir / "experiment_results.json"
    _save_jsonable_results(experiment_results, results_path)

    return experiment_results

def _plot_learning_curves(train_losses, val_losses, output_path):
    """Create and save a plot of training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    # Calculate best validation loss
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss)

    # Add marker for best epoch
    plt.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5)
    plt.annotate(f"Best: {best_val_loss:.4f}",
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + 1, best_val_loss * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_cv_results(cv_results, output_path):
    """Create and save a plot of cross-validation results."""
    folds = [r["fold"] for r in cv_results["fold_results"]]
    losses = [r["val_loss"] for r in cv_results["fold_results"]]

    plt.figure(figsize=(10, 6))
    plt.bar(folds, losses)

    # Add mean line
    mean_val_loss = cv_results["mean_val_loss"]
    plt.axhline(y=mean_val_loss, color='r', linestyle='--',
               label=f"Mean: {mean_val_loss:.4f} ± {cv_results['std_val_loss']:.4f}")

    plt.title("Cross-Validation Results")
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def _save_jsonable_results(results, output_path):
    """Save experiment results in JSON format."""
    # Convert non-serializable values
    def process_value(v):
        if isinstance(v, (np.ndarray, np.generic)):
            return v.tolist()
        elif isinstance(v, list):
            return [process_value(i) for i in v]
        elif isinstance(v, dict):
            return {k: process_value(val) for k, val in v.items()}
        elif isinstance(v, (int, float, bool, str, type(None))):
            return v
        else:
            return str(v)

    serializable_results = process_value(results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)