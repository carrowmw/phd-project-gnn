#!/usr/bin/env python
"""
Experiment runner for GNN traffic prediction

Usage: python run_experiment.py [--config CONFIG_PATH] [--data DATA_FILE] [--output OUTPUT_DIR]
"""
import os
import sys
import json
import pickle
import argparse
import asyncio
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

from gnn_package.config import get_config, ExperimentConfig, create_default_config
from gnn_package.src.training.experiment_manager import run_experiment
from gnn_package.src.training.preprocessing import prepare_data_for_experiment
from gnn_package.src.utils.data_utils import convert_numpy_types


async def main():
    """Main function to run an experiment"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a GNN model with isolated configuration"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument(
        "--create-config", action="store_true", help="Create a default config file"
    )
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument(
        "--no-cv", action="store_true", help="Disable cross-validation"
    )
    parser.add_argument(
        "--cache", type=str, help="Path to cache processed data"
    )
    args = parser.parse_args()

    # Create a default config if requested
    if args.create_config:
        config_path = args.config or "config.yml"
        print(f"Creating default configuration at {config_path}")
        config = create_default_config(config_path)
        print(
            "Default configuration created. Edit it as needed, then run the script again."
        )
        return

    # Load configuration
    if args.config:
        config = ExperimentConfig(args.config)
    else:
        # Create a default config, not using global singleton
        config = ExperimentConfig("config.yml")

    # Print configuration
    print(
        f"Running experiment: {config.experiment.name} (v{config.experiment.version})"
    )
    print(f"Description: {config.experiment.description}")
    print(
        f"Model architecture: {config.model.architecture}"
    )
    print(
        f"Data config: window_size={config.data.general.window_size}, horizon={config.data.general.horizon}"
    )
    print(
        f"Model config: hidden_dim={config.model.hidden_dim}, layers={config.model.num_layers}"
    )
    print(
        f"Training config: epochs={config.training.num_epochs}, lr={config.training.learning_rate}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(
            config.paths.results_dir,
            f"{config.experiment.name.replace(' ', '_')}_{timestamp}",
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Prepare data with cache support
    print("Preparing data...")
    cache_path = args.cache or os.path.join(output_dir, f"data_cache_{timestamp}.pkl")

    data_package = await prepare_data_for_experiment(
        data_file=args.data,
        output_cache=cache_path,
        config=config,
        use_cross_validation=not args.no_cv,
        force_refresh=False
    )

    # Run experiment
    print("Running experiment...")
    experiment_results = await run_experiment(
        data_package=data_package,
        output_dir=output_dir,
        config=config,
        use_cross_validation=not args.no_cv,
        save_model_checkpoints=True,
        plot_results=True
    )

    print(f"\nExperiment completed! All results saved to: {output_dir}")

    # Report final metrics
    if "training_results" in experiment_results:
        best_val_loss = experiment_results["training_results"]["best_val_loss"]
        print(f"Best validation loss: {best_val_loss:.6f}")
    elif "cv_results" in experiment_results:
        mean_val_loss = experiment_results["cv_results"]["mean_val_loss"]
        std_val_loss = experiment_results["cv_results"]["std_val_loss"]
        print(f"Cross-validation loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}")

    return output_dir


if __name__ == "__main__":
    try:
        output_dir = asyncio.run(main())
        print(f"Successfully completed experiment with results in: {output_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)