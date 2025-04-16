#!/usr/bin/env python
# train_model.py - Example script for training a GNN model using centralized configuration

import os
import argparse
import pickle
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Import gnn_package modules
from gnn_package.config import get_config, create_default_config, ExperimentConfig
from gnn_package import training
from gnn_package.src.utils.sensor_utils import get_sensor_name_id_map


def main():
    """Main function to run training with centralized configuration."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a GNN model with centralized configuration"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument(
        "--create-config", action="store_true", help="Create a default config file"
    )
    parser.add_argument("--output", type=str, help="Output directory for results")
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
        # Use global configuration or create default if none exists
        try:
            config = get_config()
        except FileNotFoundError:
            print("No configuration found. Creating default configuration.")
            config = create_default_config("config.yml")
            print(
                "Default configuration created. Edit it as needed, then run the script again."
            )
            return

    # Print configuration details
    print(
        f"Running experiment: {config.experiment.name} (v{config.experiment.version})"
    )
    print(f"Description: {config.experiment.description}")
    print(
        f"Data config: window_size={config.data.window_size}, horizon={config.data.horizon}"
    )
    print(
        f"Model config: hidden_dim={config.model.hidden_dim}, layers={config.model.num_layers}"
    )
    print(
        f"Training config: epochs={config.training.num_epochs}, lr={config.training.learning_rate}"
    )

    # Set up paths
    raw_file_name = args.data or os.path.join(
        "gnn_package/data/raw/timeseries", f"test_data_{config.data.days_back}d.pkl"
    )
    print(f"Using data file: {raw_file_name}")

    # Set up output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            config.paths.results_dir,
            f"{config.experiment.name.replace(' ', '_')}_{timestamp}",
        )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Save the configuration used for this run
    config_save_path = os.path.join(output_dir, "run_config.yml")
    config.save(config_save_path)
    print(f"Configuration saved to: {config_save_path}")

    # Preprocess data with centralized configuration
    print("Preprocessing data...")
    data_loaders = training.preprocess_data(
        data_file=raw_file_name,
        config=config,
    )

    # Save preprocessed data
    preprocessed_file_path = os.path.join(output_dir, "preprocessed_data.pkl")
    with open(preprocessed_file_path, "wb") as f:
        pickle.dump(data_loaders, f)
    print(f"Preprocessed data saved to: {preprocessed_file_path}")

    # Train model with centralized configuration
    print("Training model...")
    results = training.train_model(
        data_loaders=data_loaders,
        config=config,
    )

    # Save the trained model
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(results["model"].state_dict(), model_path)
    print(f"Trained model saved to: {model_path}")

    # Save training plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["train_losses"], label="Training Loss")
    plt.plot(results["val_losses"], label="Validation Loss")
    plt.title(f"Training Results: {config.experiment.name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Training plot saved to: {plot_path}")

    # Save training metrics
    metrics = {
        "train_losses": results["train_losses"],
        "val_losses": results["val_losses"],
        "best_val_loss": results["best_val_loss"],
        "config": config._config_dict,
    }

    metrics_path = os.path.join(output_dir, "metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"Training metrics saved to: {metrics_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
