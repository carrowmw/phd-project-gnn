# gnn_package/src/tuning/experiment_manager.py

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import mlflow
import mlflow.pytorch
import optuna
import pandas as pd
from tabulate import tabulate

from gnn_package.config import get_config

from gnn_package.config import ExperimentConfig

logger = logging.getLogger(__name__)


def setup_mlflow_experiment(
    experiment_name: str, output_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Set up MLflow experiment and configure tracking.

    Parameters:
    -----------
    experiment_name : str
        Name of the MLflow experiment
    output_dir : str or Path, optional
        Directory to store MLflow data

    Returns:
    --------
    str
        MLflow experiment ID
    """
    # Configure MLflow tracking - use local directory if not specified
    if output_dir is None:
        output_dir = Path("mlruns")
    else:
        output_dir = Path(output_dir) / "mlruns"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set MLflow tracking URI to a local directory
    tracking_uri = f"file://{output_dir.absolute()}"
    mlflow.set_tracking_uri(tracking_uri)

    # Create or get the experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(
                f"Created new experiment '{experiment_name}' with ID: {experiment_id}"
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing experiment '{experiment_name}' with ID: {experiment_id}"
            )
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {str(e)}")
        # Fallback - create a new experiment with a timestamp to avoid conflicts
        import time

        experiment_name = f"{experiment_name}_{int(time.time())}"
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(
            f"Created fallback experiment '{experiment_name}' with ID: {experiment_id}"
        )

    # Set the active experiment
    mlflow.set_experiment(experiment_name)

    return experiment_id


def log_trial_metrics(
    metrics: Dict[str, Any], standardization_stats: Dict[str, Any]
) -> None:
    """
    Log metrics from a trial to MLflow.

    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary of metrics to log
    standardization_stats : Dict[str, Any]
        Dictionary of standardization statistics from preprocessing
    """
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(name, value)

    # Log standardization statistics separately
    if standardization_stats:
        mlflow.log_param("standardization_mean", standardization_stats.get("mean", 0))
        mlflow.log_param("standardization_std", standardization_stats.get("std", 1))


def save_config_from_params(
    params: Dict[str, Any],
    output_path: Union[str, Path],
    base_config: Optional[ExperimentConfig] = None,
) -> None:
    """
    Create and save a configuration file from tuned parameters.

    Parameters:
    -----------
    params : Dict[str, Any]
        Dictionary of tuned parameters
    output_path : str or Path
        Path to save the configuration file
    base_config : ExperimentConfig, optional
        Base configuration to update with params
    """
    # Get base config
    if base_config is None:
        config = get_config()
    else:
        config = base_config

    # Update config with tuned parameters
    for param_name, value in params.items():
        parts = param_name.split(".")
        if len(parts) == 2:
            section, attribute = parts
            config_dict = config._config_dict
            if section in config_dict and attribute in config_dict[section]:
                config_dict[section][attribute] = value

    # Save updated config
    with open(output_path, "w") as f:
        yaml.dump(config._config_dict, f, default_flow_style=False)


def log_best_trial_details(
    study: optuna.study.Study,
    experiment_name: str,
    output_dir: Union[str, Path],
    include_time_series: bool = True,
) -> None:
    """
    Log details about the best trial and generate summary reports.

    Parameters:
    -----------
    study : optuna.study.Study
        Completed Optuna study
    experiment_name : str
        Name of the experiment
    output_dir : str or Path
        Directory to save results
    include_time_series : bool
        Whether to include training curves in the report
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get best trial
    best_trial = study.best_trial

    # Save best parameters
    best_params = best_trial.params
    best_params_path = output_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Generate detailed report for the best trial
    best_trial_report = {
        "trial_number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "datetime_start": best_trial.datetime_start.isoformat(),
        "datetime_complete": best_trial.datetime_complete.isoformat(),
        "duration_seconds": (
            best_trial.datetime_complete - best_trial.datetime_start
        ).total_seconds(),
    }

    # Add user attributes (if any)
    for key, value in best_trial.user_attrs.items():
        best_trial_report[key] = value

    # Save best trial report
    best_trial_report_path = output_dir / "best_trial_report.json"
    with open(best_trial_report_path, "w") as f:
        json.dump(best_trial_report, f, indent=2)

    # Generate summary of all trials
    trial_data = []
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
            trial_data.append(row)

    # Create DataFrame and sort by performance
    if trial_data:
        trials_df = pd.DataFrame(trial_data)
        trials_df = trials_df.sort_values("value")

        # Save as CSV
        trials_csv_path = output_dir / "all_trials.csv"
        trials_df.to_csv(trials_csv_path, index=False)

        # Create a text report for better readability
        trials_report_path = output_dir / "trials_summary.txt"
        with open(trials_report_path, "w") as f:
            f.write(f"# Hyperparameter Tuning Report: {experiment_name}\n\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Completed trials: {len(trials_df)}\n")
            f.write(f"Best trial: #{best_trial.number}\n")
            f.write(f"Best value: {best_trial.value}\n\n")

            f.write("## Best Parameters\n\n")
            best_params_table = [[param, value] for param, value in best_params.items()]
            f.write(
                tabulate(
                    best_params_table, headers=["Parameter", "Value"], tablefmt="grid"
                )
            )
            f.write("\n\n")

            f.write("## Top 10 Trials\n\n")
            top_10_trials = trials_df.head(10)
            # Select relevant columns for the report
            cols_to_show = (
                ["number", "value"] + list(best_params.keys()) + ["duration_seconds"]
            )
            cols_to_show = [col for col in cols_to_show if col in top_10_trials.columns]
            f.write(
                tabulate(
                    top_10_trials[cols_to_show].values,
                    headers=cols_to_show,
                    tablefmt="grid",
                )
            )
            f.write("\n\n")

            f.write("## Parameter Importance\n\n")
            try:
                importances = optuna.importance.get_param_importances(study)
                importance_table = [
                    [param, importance] for param, importance in importances.items()
                ]
                f.write(
                    tabulate(
                        importance_table,
                        headers=["Parameter", "Importance"],
                        tablefmt="grid",
                    )
                )
            except Exception as e:
                f.write(f"Could not compute parameter importance: {str(e)}\n")

    logger.info(f"Trial reports saved to {output_dir}")

    # Save a copy of the best configuration
    try:
        from gnn_package.config import get_config

        config = get_config()
        best_config_path = output_dir / "best_config.yml"
        save_config_from_params(best_params, best_config_path, config)
        logger.info(f"Best configuration saved to {best_config_path}")
    except Exception as e:
        logger.error(f"Error saving best configuration: {str(e)}")
