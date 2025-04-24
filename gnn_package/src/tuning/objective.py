# gnn_package/src/tuning/objective.py

import os
import copy
import pickle
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import mlflow
import optuna
import torch
import numpy as np

from gnn_package.config import ExperimentConfig, get_config
from gnn_package import training
from .parameter_space import get_param_space_with_suggestions
from .experiment_manager import log_trial_metrics

logger = logging.getLogger(__name__)


def update_config_with_params(
    config: ExperimentConfig, params: Dict[str, Any]
) -> ExperimentConfig:
    """
    Update configuration object with parameters from the tuning process.

    Parameters:
    -----------
    config : ExperimentConfig
        Original configuration object
    params : Dict[str, Any]
        Parameters to update in the configuration

    Returns:
    --------
    ExperimentConfig
        Updated configuration object
    """
    # Create a deep copy to avoid modifying the original
    updated_config = copy.deepcopy(config)

    # Update each parameter
    for param_name, value in params.items():
        # Split parameter name into sections
        parts = param_name.split(".")

        # Handle nested attributes
        if len(parts) == 2:
            section, attribute = parts
            if hasattr(updated_config, section) and hasattr(
                getattr(updated_config, section), attribute
            ):
                setattr(getattr(updated_config, section), attribute, value)
            else:
                logger.warning(
                    f"Could not update {param_name}: attribute not found in config"
                )
        elif len(parts) == 1:
            # Handle top-level attributes
            if hasattr(updated_config, param_name):
                setattr(updated_config, param_name, value)
            else:
                logger.warning(
                    f"Could not update {param_name}: attribute not found in config"
                )
        else:
            logger.warning(f"Unexpected parameter format: {param_name}")

    return updated_config


def create_objective_function(
    data_file: Union[str, Path],
    param_space: Dict[str, Any],
    experiment_name: str,
    config: Optional[ExperimentConfig] = None,
    n_epochs: Optional[int] = None,
):
    """
    Create an objective function for Optuna to optimize.

    Parameters:
    -----------
    data_file : str or Path
        Path to the data file
    param_space : Dict[str, Any]
        Parameter space definition
    experiment_name : str
        Name of the MLflow experiment
    config : ExperimentConfig, optional
        Base configuration to use (falls back to global config)
    n_epochs : int, optional
        Number of epochs to train (overrides config)

    Returns:
    --------
    Callable
        Objective function that takes an Optuna trial
    """
    if config is None:
        config = get_config()

    # Function to be optimized
    def objective(trial: optuna.trial.Trial) -> float:
        # Get parameter suggestions for this trial
        params = get_param_space_with_suggestions(trial, param_space)

        # Create a new MLflow run for each trial
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            # Log parameters to MLflow
            mlflow.log_params(params)

        # Update configuration with sampled parameters
        trial_config = update_config_with_params(config, params)

        # Override number of epochs if specified
        if n_epochs is not None:
            trial_config.training.num_epochs = n_epochs

        # Set a shorter patience for faster hyperparameter tuning
        # This reduces training time while still identifying promising configurations
        trial_config.training.patience = min(
            trial_config.training.patience,
            max(3, int(trial_config.training.num_epochs * 0.2)),
        )

        try:
            # Preprocess data with the current configuration - THIS IS THE KEY CHANGE
            # We need to run the async function in the event loop
            data_package = asyncio.run(
                training.preprocess_data(
                    data_file=data_file,
                    config=trial_config,
                )
            )

            # Extract standardization stats if available
            standardization_stats = {}
            if (
                "metadata" in data_package
                and "preprocessing_stats" in data_package["metadata"]
            ):
                standardization_stats = data_package["metadata"][
                    "preprocessing_stats"
                ].get("standardization", {})

            # Add to trial attributes for later analysis
            trial.set_user_attr("standardization_stats", standardization_stats)

            # Train model with updated config
            results = training.train_model(
                data_package=data_package,
                config=trial_config,
            )

            # Get validation loss as the optimization target
            best_val_loss = results["best_val_loss"]

            # Calculate additional metrics
            metrics = {
                "best_val_loss": best_val_loss,
                "final_train_loss": results["train_losses"][-1],
                "final_val_loss": results["val_losses"][-1],
                "num_epochs_trained": len(results["train_losses"]),
                "stopped_early": len(results["train_losses"])
                < trial_config.training.num_epochs,
            }

            # Log metrics to MLflow
            log_trial_metrics(metrics, standardization_stats)

            # Report to the trial
            trial.set_user_attr("metrics", metrics)

            # Free up memory
            del data_package
            del results
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None

            return best_val_loss

        except Exception as e:
            # Log failed trial
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            mlflow.log_param("error", str(e))
            raise optuna.exceptions.TrialPruned(f"Trial failed: {str(e)}")

    return objective


async def train_with_best_params(
    data_file: Union[str, Path],
    best_params: Dict[str, Any],
    output_dir: Union[str, Path],
    config: Optional[ExperimentConfig] = None,
) -> Tuple[ExperimentConfig, Dict[str, Any]]:
    """
    Train the model with the best parameters found during tuning.

    Parameters:
    -----------
    data_file : str or Path
        Path to the data file
    best_params : Dict[str, Any]
        Best parameters found during tuning
    output_dir : str or Path
        Directory to save results
    config : ExperimentConfig, optional
        Base configuration to use (falls back to global config)

    Returns:
    --------
    Tuple[ExperimentConfig, Dict[str, Any]]
        Updated configuration and training results
    """
    if config is None:
        config = get_config()

    # Update configuration with best parameters
    best_config = update_config_with_params(config, best_params)

    # Save the best configuration
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    best_config_path = output_dir / "best_config.yml"
    best_config.save(str(best_config_path))

    # Preprocess data with best configuration
    with mlflow.start_run(run_name="best_params_training"):
        mlflow.log_params(best_params)

        data_package = await training.preprocess_data(
            data_file=data_file,
            config=best_config,
        )

        # Extract standardization stats
        standardization_stats = {}
        if (
            "metadata" in data_package
            and "preprocessing_stats" in data_package["metadata"]
        ):
            standardization_stats = data_package["metadata"]["preprocessing_stats"].get(
                "standardization", {}
            )

        mlflow.log_params(
            {
                "standardization_mean": standardization_stats.get("mean", 0),
                "standardization_std": standardization_stats.get("std", 1),
            }
        )

        # Train model with best configuration
        results = training.train_model(
            data_package=data_package,
            config=best_config,
        )

        # Log metrics
        metrics = {
            "best_val_loss": results["best_val_loss"],
            "final_train_loss": results["train_losses"][-1],
            "final_val_loss": results["val_losses"][-1],
            "num_epochs_trained": len(results["train_losses"]),
        }
        mlflow.log_metrics(metrics)

        # Save the best model
        model_path = output_dir / "best_model.pth"
        torch.save(results["model"].state_dict(), str(model_path))
        mlflow.log_artifact(str(model_path))

        # Save loss curve data
        loss_data = {
            "train_losses": results["train_losses"],
            "val_losses": results["val_losses"],
        }
        loss_path = output_dir / "training_curves.pkl"
        with open(loss_path, "wb") as f:
            pickle.dump(loss_data, f)
        mlflow.log_artifact(str(loss_path))

    return best_config, results
