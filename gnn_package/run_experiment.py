#!/usr/bin/env python
"""
Experiment runner for GNN traffic prediction

Usage: python run_experiment.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
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

from gnn_package import training
from gnn_package import paths
from gnn_package.config import get_config, ExperimentConfig
from gnn_package.src.utils.data_utils import convert_numpy_types


runtime_stats = {
    "preprocessing": {},
    "standardization": {},
    "training": {},
}


async def main():
    """Main function to run an experiment"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument(
        "--output", type=str, help="Output directory for experiment results"
    )
    parser.add_argument("--data", type=str, help="Path to data file")
    args = parser.parse_args()

    # Create experiment timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize config
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        config = ExperimentConfig(args.config)
        experiment_name = f"{os.path.basename(args.config).split('.')[0]}_{timestamp}"
    else:
        print("Using global configuration")
        config = get_config()
        experiment_name = f"default_experiment_{timestamp}"

    # Print configuration
    print(f"Experiment: {config.experiment.name} (v{config.experiment.version})")
    print(
        f"Data config: window_size={config.data.general.window_size}, horizon={config.data.general.horizon}"
    )
    print(
        f"Model config: hidden_dim={config.model.hidden_dim}, layers={config.model.num_layers}"
    )
    print(
        f"Training config: epochs={config.training.num_epochs}, lr={config.training.learning_rate}"
    )

    # Create output directory
    if args.output:
        output_dir = os.path.join(args.output, experiment_name)
    else:
        output_dir = os.path.join("experiments", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set data file
    if args.data:
        raw_file_path = args.data
    else:
        raw_file_name = "test_data_1wk.pkl"
        raw_dir = paths.RAW_TIMESERIES_DIR
        raw_file_path = os.path.join(raw_dir, raw_file_name)

    print(f"Using data file: {raw_file_path}")
    runtime_stats["data_file"] = raw_file_path

    # Preprocess data
    preprocessed_file_name = f"data_loaders_{os.path.basename(raw_file_path)}"
    preprocessed_path = os.path.join(output_dir, preprocessed_file_name)

    # Preprocess data
    print("Preprocessing data...")
    preprocess_start = datetime.now()

    # Before preprocessing
    print("\n===== Starting preprocessing =====")

    # In main function
    data_package = await training.preprocess_data(
        data_file=raw_file_path, config=config
    )

    # After preprocessing
    print("\n===== Preprocessing completed =====")
    print(f"Data package keys: {data_package.keys()}")

    preprocess_end = datetime.now()
    print(f"Preprocessing completed in {preprocess_end - preprocess_start}")

    if data_package is None:
        raise ValueError(
            "preprocessing returned None - check the preprocessing pipeline"
        )
    print(f"Data loaders type: {type(data_package)}")
    if isinstance(data_package, dict):
        print(f"Data loaders keys: {data_package.keys()}")

    # Extract standardization stats if available
    preprocessing_stats = {}
    if (
        isinstance(data_package, dict)
        and "preprocessing_stats" in data_package["metadata"]
    ):
        preprocessing_stats = data_package["metadata"]["preprocessing_stats"]

    runtime_stats["preprocessing"] = {
        "start_time": preprocess_start.isoformat(),
        "end_time": preprocess_end.isoformat(),
        "duration_seconds": (preprocess_end - preprocess_start).total_seconds(),
        "standardization": preprocessing_stats.get("standardization", {}),
    }

    # Extract standardization stats if available (from the processor's internal data)
    # This depends on how your DataProcessor exposes the stats
    if hasattr(data_package["metadata"], "preprocessing_stats"):
        runtime_stats["standardization"] = (
            data_package.metadata.preprocessing_stats.get("standardization", {})
        )

    # Save preprocessed data
    with open(preprocessed_path, "wb") as f:
        pickle.dump(data_package, f)
    print(f"Preprocessed data saved to: {preprocessed_path}")

    # Train the model
    print("Training model...")
    training_start = datetime.now()
    results = training.train_model(data_package=data_package, config=config)
    training_end = datetime.now()

    # Store training statistics
    runtime_stats["training"] = {
        "start_time": training_start.isoformat(),
        "end_time": training_end.isoformat(),
        "duration_seconds": (training_end - training_start).total_seconds(),
        "epochs_trained": len(results["train_losses"]),
        "best_epoch": np.argmin(results["val_losses"]),
        "best_validation_loss": results["best_val_loss"],
    }

    # Save experiment outputs

    # 1. Save the model
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(results["model"].state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # 2. Save a copy of the config used
    config_path = os.path.join(output_dir, "config.yml")
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")

    # 3. Save runtime statistics
    stats_path = os.path.join(output_dir, "runtime_stats.json")
    # Convert numpy types to native Python types for JSON serialization
    runtime_stats_converted = convert_numpy_types(runtime_stats)

    with open(stats_path, "w") as f:
        json.dump(runtime_stats_converted, f, indent=2)
    print(f"Runtime statistics saved to: {stats_path}")

    # 4. Save performance metrics
    performance_path = os.path.join(output_dir, "performance.json")
    performance = {
        "best_validation_loss": results["best_val_loss"],
        "final_training_loss": results["train_losses"][-1],
        "final_validation_loss": results["val_losses"][-1],
        "epochs_trained": len(results["train_losses"]),
        "early_stopping": len(results["train_losses"]) < config.training.num_epochs,
        "training_losses": results["train_losses"],
        "validation_losses": results["val_losses"],
    }
    with open(performance_path, "w") as f:
        json.dump(performance, f, indent=2)
    print(f"Performance metrics saved to: {performance_path}")

    # 5. Generate a training curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["train_losses"], label="Training Loss")
    plt.plot(results["val_losses"], label="Validation Loss")
    plt.axhline(
        y=results["best_val_loss"],
        color="r",
        linestyle="--",
        label=f"Best Val Loss: {results['best_val_loss']:.4f}",
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Training curve plot saved to: {plot_path}")

    print(f"\nExperiment completed! All results saved to: {output_dir}")
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
