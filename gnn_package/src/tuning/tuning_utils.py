# gnn_package/src/tuning/tuning_utils.py

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import mlflow
import optuna
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gnn_package.config import ExperimentConfig, get_config
from .parameter_space import get_default_param_space, get_focused_param_space
from .experiment_manager import (
    setup_mlflow_experiment,
    log_best_trial_details,
    save_config_from_params,
)
from .objective import create_objective_function, train_with_best_params

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    data_file: Union[str, Path],
    experiment_name: str,
    n_trials: int = 20,
    n_epochs: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[ExperimentConfig] = None,
    param_space: Optional[Dict[str, Any]] = None,
    previous_best_params: Optional[Dict[str, Any]] = None,
    retrain_best: bool = True,
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning with Optuna and MLflow.

    Parameters:
    -----------
    data_file : str or Path
        Path to the data file
    experiment_name : str
        Name of the MLflow experiment
    n_trials : int
        Number of trials to run
    n_epochs : int, optional
        Number of epochs to train (overrides config)
    output_dir : str or Path, optional
        Directory to save results (uses experiment_name if not provided)
    config : ExperimentConfig, optional
        Base configuration to use (falls back to global config)
    param_space : Dict[str, Any], optional
        Parameter space definition (uses default if not provided)
    previous_best_params : Dict[str, Any], optional
        Best parameters from a previous tuning run for focused search
    retrain_best : bool
        Whether to retrain with the best parameters after tuning
    study_name : str, optional
        Name for the Optuna study (uses experiment_name if not provided)

    Returns:
    --------
    Dict[str, Any]
        Best parameters and study information
    """
    # Setup MLflow experiment
    if output_dir is None:
        output_dir = Path(f"results/tuning/{experiment_name}")

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Set up MLflow experiment
    experiment_id = setup_mlflow_experiment(experiment_name, output_dir)

    # Set default config if not provided
    if config is None:
        config = get_config()

    # Set parameter space if not provided
    if param_space is None:
        if previous_best_params is not None:
            param_space = get_focused_param_space(previous_best_params)
        else:
            param_space = get_default_param_space()

    # Set study name if not provided
    if study_name is None:
        study_name = experiment_name

    # Create objective function
    objective = create_objective_function(
        data_file=data_file,
        param_space=param_space,
        experiment_name=experiment_name,
        config=config,
        n_epochs=n_epochs,
    )

    # Create Optuna storage and study
    storage_path = output_dir / f"{study_name}.db"
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path}", engine_kwargs={"connect_args": {"timeout": 30}}
    )

    # Create or load existing study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",  # Minimize validation loss
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Check if study already has trials
    existing_trials = len(study.trials)
    if existing_trials > 0:
        logger.info(f"Loaded existing study with {existing_trials} trials")
        logger.info(f"Best value so far: {study.best_value}")

    # Run the optimization
    with mlflow.start_run(run_name=f"{study_name}_optimization"):
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        study.optimize(
            objective, n_trials=n_trials, timeout=None, show_progress_bar=True
        )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best validation loss: {best_value}")
    logger.info(f"Best parameters: {best_params}")

    # Save best parameters to file
    best_params_path = output_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Generate reports
    log_best_trial_details(study, experiment_name, output_dir)

    # Create visualization plots
    try:
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(str(output_dir / "optimization_history.png"))

        # Parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(str(output_dir / "param_importances.png"))

        # Parallel coordinate plot for parameters
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(str(output_dir / "parallel_coordinate.png"))

        # Slice plot for selected parameters
        for param in study.best_params.keys():
            if len(study.trials) > 10:  # Only if we have enough trials
                fig = optuna.visualization.plot_slice(study, params=[param])
                fig.write_image(str(output_dir / f"slice_{param}.png"))

        logger.info(f"Visualization plots saved to {output_dir}")
    except Exception as e:
        logger.warning(f"Error creating visualization plots: {str(e)}")

    # Retrain with best parameters if requested
    if retrain_best:
        logger.info("Retraining with best parameters...")
        best_model_dir = output_dir / "best_model"
        os.makedirs(best_model_dir, exist_ok=True)

        best_config, best_results = train_with_best_params(
            data_file=data_file,
            best_params=best_params,
            output_dir=best_model_dir,
            config=config,
        )

        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(best_results["train_losses"], label="Training Loss")
        plt.plot(best_results["val_losses"], label="Validation Loss")
        plt.axhline(
            y=best_results["best_val_loss"],
            color="r",
            linestyle="--",
            label=f"Best Val Loss: {best_results['best_val_loss']:.4f}",
        )
        plt.title(f"Training Results with Best Parameters")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            str(best_model_dir / "loss_curves.png"), dpi=300, bbox_inches="tight"
        )

        logger.info(f"Best model results saved to {best_model_dir}")

    return {
        "best_params": best_params,
        "best_value": best_value,
        "study": study,
        "output_dir": str(output_dir),
    }


def get_best_params(
    output_dir: Union[str, Path],
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load the best parameters from a previous tuning run.

    Parameters:
    -----------
    output_dir : str or Path
        Directory with tuning results
    study_name : str, optional
        Name of the Optuna study (uses output_dir.name if not provided)

    Returns:
    --------
    Dict[str, Any]
        Best parameters
    """
    output_dir = Path(output_dir)

    # First try to load from best_params.json
    best_params_path = output_dir / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            return json.load(f)

    # If not found, try to load from Optuna storage
    if study_name is None:
        study_name = output_dir.name

    storage_path = output_dir / f"{study_name}.db"
    if storage_path.exists():
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{storage_path}",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.load_study(study_name=study_name, storage=storage)
        return study.best_params

    raise FileNotFoundError(f"Could not find best parameters in {output_dir}")


def load_tuning_results(
    output_dir: Union[str, Path],
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load results from a previous tuning run.

    Parameters:
    -----------
    output_dir : str or Path
        Directory with tuning results
    study_name : str, optional
        Name of the Optuna study (uses output_dir.name if not provided)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing study information and results
    """
    output_dir = Path(output_dir)

    if study_name is None:
        study_name = output_dir.name

    # Load Optuna study
    storage_path = output_dir / f"{study_name}.db"
    if not storage_path.exists():
        raise FileNotFoundError(f"Optuna storage not found: {storage_path}")

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path}", engine_kwargs={"connect_args": {"timeout": 30}}
    )
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Load trial data
    all_trials_path = output_dir / "all_trials.csv"
    if all_trials_path.exists():
        trials_df = pd.read_csv(all_trials_path)
    else:
        # Create from study
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = {
                    "number": trial.number,
                    "value": trial.value,
                    **trial.params,
                    "duration_seconds": (
                        trial.datetime_complete - trial.datetime_start
                    ).total_seconds(),
                }
                trials_data.append(row)

        trials_df = pd.DataFrame(trials_data) if trials_data else None

    # Load best model results if available
    best_model_dir = output_dir / "best_model"
    best_model_results = None

    if best_model_dir.exists():
        training_curves_path = best_model_dir / "training_curves.pkl"
        if training_curves_path.exists():
            with open(training_curves_path, "rb") as f:
                best_model_results = pickle.load(f)

        best_model_path = best_model_dir / "best_model.pth"
        if best_model_path.exists():
            best_model_results = best_model_results or {}
            best_model_results["model_path"] = str(best_model_path)

    return {
        "study": study,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials_df": trials_df,
        "best_model_results": best_model_results,
        "output_dir": str(output_dir),
    }


def run_multi_stage_tuning(
    data_file: Union[str, Path],
    experiment_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[ExperimentConfig] = None,
    n_trials_stages: List[int] = [20, 10, 5],
    n_epochs_stages: List[Optional[int]] = [10, 20, None],
    data_fraction_stages: List[Optional[float]] = [0.25, 0.5, 1.0],
) -> Dict[str, Any]:
    """
    Run multi-stage hyperparameter tuning with progressively more data and epochs.

    Parameters:
    -----------
    data_file : str or Path
        Path to the data file
    experiment_name : str
        Base name for the MLflow experiment
    output_dir : str or Path, optional
        Directory to save results
    config : ExperimentConfig, optional
        Base configuration to use
    n_trials_stages : List[int]
        Number of trials to run in each stage
    n_epochs_stages : List[int]
        Number of epochs to train in each stage
    data_fraction_stages : List[float]
        Fraction of data to use in each stage

    Returns:
    --------
    Dict[str, Any]
        Best parameters and study information from the final stage
    """
    if output_dir is None:
        output_dir = Path(f"results/tuning/{experiment_name}")

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Ensure all stage parameter lists have the same length
    n_stages = len(n_trials_stages)
    if len(n_epochs_stages) != n_stages:
        raise ValueError("n_epochs_stages must have the same length as n_trials_stages")
    if len(data_fraction_stages) != n_stages:
        raise ValueError(
            "data_fraction_stages must have the same length as n_trials_stages"
        )

    # Initialize best params for the first stage
    previous_best_params = None
    final_results = None

    # Create a list to store results from each stage
    stage_results = []

    # Run each stage
    for i in range(n_stages):
        stage_name = f"stage_{i+1}"
        stage_experiment_name = f"{experiment_name}_{stage_name}"
        stage_output_dir = output_dir / stage_name

        # Prepare data for this stage - this is just a placeholder
        # In a real implementation, you'd sample/preprocess data according to data_fraction_stages[i]
        stage_data_file = data_file  # For now, we use the same data file

        logger.info(f"Starting tuning stage {i+1}/{n_stages}")
        logger.info(f"  Trials: {n_trials_stages[i]}")
        logger.info(f"  Epochs: {n_epochs_stages[i]}")
        logger.info(f"  Data fraction: {data_fraction_stages[i]}")

        # Run this stage of tuning
        results = tune_hyperparameters(
            data_file=stage_data_file,
            experiment_name=stage_experiment_name,
            n_trials=n_trials_stages[i],
            n_epochs=n_epochs_stages[i],
            output_dir=stage_output_dir,
            config=config,
            previous_best_params=previous_best_params,
            retrain_best=(i == n_stages - 1),  # Only retrain on final stage
            study_name=stage_name,
        )

        # Store results for this stage
        stage_results.append(
            {
                "stage": i + 1,
                "experiment_name": stage_experiment_name,
                "best_params": results["best_params"],
                "best_value": results["best_value"],
            }
        )

        # Update best params for next stage
        previous_best_params = results["best_params"]
        final_results = results

    # Save summary of all stages
    stages_summary_path = output_dir / "stages_summary.json"
    with open(stages_summary_path, "w") as f:
        json.dump(stage_results, f, indent=2)

    # Create comparison plot of stages
    plt.figure(figsize=(10, 6))
    stages = [f"Stage {i+1}" for i in range(n_stages)]
    best_values = [stage["best_value"] for stage in stage_results]

    plt.bar(stages, best_values, color="skyblue")
    plt.title("Best Validation Loss by Tuning Stage")
    plt.xlabel("Stage")
    plt.ylabel("Validation Loss")
    plt.xticks(rotation=0)

    for i, value in enumerate(best_values):
        plt.text(i, value, f"{value:.4f}", ha="center", va="bottom")

    plt.savefig(str(output_dir / "stages_comparison.png"), dpi=300, bbox_inches="tight")

    return {
        "stage_results": stage_results,
        "final_results": final_results,
        "output_dir": str(output_dir),
        "best_params": previous_best_params,
    }
