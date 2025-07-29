#!/usr/bin/env python
# tune_model.py - Script for hyperparameter tuning using the tuning module

import argparse
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Import tuning module
from gnn_package.src.tuning import (
    tune_hyperparameters,
    run_multi_stage_tuning,
)
from gnn_package.config import get_config, ExperimentConfig
from gnn_package.src.training.preprocessing import prepare_data_for_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tuning.log"),
    ],
)
logger = logging.getLogger("tuning")


async def main():
    """Run hyperparameter tuning"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for GNN traffic prediction"
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data file (e.g., data/raw/timeseries/test_data_1wk.pkl)",
    )

    # Optional arguments
    parser.add_argument("--config", type=str, help="Path to a custom config file")
    parser.add_argument("--output", type=str, help="Directory to save tuning results")
    parser.add_argument("--experiment", type=str, help="Name of the experiment")
    parser.add_argument(
        "--trials", type=int, default=20, help="Number of trials to run (default: 20)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs per trial (default: use config value)",
    )
    parser.add_argument(
        "--multi-stage",
        action="store_true",
        help="Run multi-stage tuning with increasing data and epochs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick tuning with fewer trials and epochs (for testing)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        help="Path to cache processed data",
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = ExperimentConfig(args.config)
        logger.info(f"Loaded custom config from {args.config}")
    else:
        config = get_config()
        logger.info("Using default config")

    # Set up paths
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Generate experiment name if not provided
    if args.experiment:
        experiment_name = args.experiment
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = data_path.stem
        experiment_name = f"tuning_{dataset_name}_{timestamp}"

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"results/tuning/{experiment_name}")

    # Prepare data with caching
    logger.info("Preparing data for tuning...")
    cache_path = args.cache or str(output_dir / "data_cache.pkl")

    data_package = await prepare_data_for_experiment(
        data_file=args.data,
        output_cache=cache_path,
        config=config,
        use_cross_validation=True,  # Always use CV for tuning
        force_refresh=False
    )

    # Run tuning
    if args.quick:
        logger.info("Running quick tuning (reduced trials and epochs for testing)")
        n_trials = 5
        n_epochs = 5
    else:
        n_trials = args.trials
        n_epochs = args.epochs

    if args.multi_stage:
        logger.info(
            f"Running multi-stage tuning with experiment name: {experiment_name}"
        )

        # Define stages for multi-stage tuning
        if args.quick:
            # Quick version for testing
            n_trials_stages = [3, 2]
            n_epochs_stages = [3, 5]
            data_fraction_stages = [0.3, 1.0]
        else:
            # Full version
            n_trials_stages = [15, 10, 5]
            n_epochs_stages = [10, 20, None]  # None uses the config value
            data_fraction_stages = [0.25, 0.5, 1.0]

        results = run_multi_stage_tuning(
            data_package=data_package,  # Pass data_package instead of data_file
            experiment_name=experiment_name,
            output_dir=output_dir,
            config=config,
            n_trials_stages=n_trials_stages,
            n_epochs_stages=n_epochs_stages,
            data_fraction_stages=data_fraction_stages,
        )

        logger.info(
            f"Multi-stage tuning completed. Results saved to {results['output_dir']}"
        )
        if "best_params" in results and results["best_params"]:
            logger.info(f"Best parameters: {results['best_params']}")
    else:
        logger.info(
            f"Running hyperparameter tuning with experiment name: {experiment_name}"
        )

        results = tune_hyperparameters(
            data_package=data_package,  # Pass data_package instead of data_file
            experiment_name=experiment_name,
            n_trials=n_trials,
            n_epochs=n_epochs,
            output_dir=output_dir,
            config=config,
            retrain_best=True,
        )

        logger.info(f"Tuning completed. Results saved to {results['output_dir']}")
        if "best_params" in results and results["best_params"]:
            logger.info(f"Best parameters: {results['best_params']}")
            if "best_value" in results:
                logger.info(f"Best validation loss: {results['best_value']}")


if __name__ == "__main__":
    asyncio.run(main())