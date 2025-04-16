# Hyperparameter Tuning Module: Usage Guide

This guide explains how to use the hyperparameter tuning module for the GNN traffic prediction model.

## Basic Usage

### Running from the Command Line

The simplest way to run hyperparameter tuning is to use the provided script:

```bash
# Run basic tuning with default parameters
python tune_model.py --data gnn_package/data/raw/timeseries/test_data_1wk.pkl

# Run with custom configuration
python tune_model.py --data gnn_package/data/raw/timeseries/test_data_1wk.pkl --config config.yml --trials 30 --epochs 15

# Run a quick test with fewer trials and epochs
python tune_model.py --data gnn_package/data/raw/timeseries/test_data_1wk.pkl --quick

# Run multi-stage tuning with progressively more data and epochs
python tune_model.py --data gnn_package/data/raw/timeseries/test_data_1mnth.pkl --multi-stage
```

### Running from Python

You can also use the tuning module directly in your Python code:

```python
import os
from pathlib import Path
from gnn_package.src.tuning import tune_hyperparameters

# Run hyperparameter tuning
results = tune_hyperparameters(
    data_file="gnn_package/data/raw/timeseries/test_data_1wk.pkl",
    experiment_name="my_tuning_experiment",
    n_trials=20,
    n_epochs=15,
    output_dir="results/my_tuning_experiment",
)

# Get best parameters
best_params = results["best_params"]
print(f"Best parameters: {best_params}")
print(f"Best validation loss: {results['best_value']}")
```

## Multi-Stage Tuning

For larger datasets, you can use multi-stage tuning to efficiently find good hyperparameters:

```python
from gnn_package.src.tuning import run_multi_stage_tuning

# Run multi-stage tuning
results = run_multi_stage_tuning(
    data_file="gnn_package/data/raw/timeseries/test_data_1mnth.pkl",
    experiment_name="multi_stage_experiment",
    n_trials_stages=[15, 10, 5],        # Number of trials per stage
    n_epochs_stages=[10, 20, None],     # Number of epochs per stage (None uses config value)
    data_fraction_stages=[0.25, 0.5, 1.0],  # Fraction of data to use per stage
)

# Get best parameters from the final stage
best_params = results["best_params"]
```

## Loading and Using Tuning Results

After running tuning, you can load and use the results:

```python
from gnn_package.src.tuning import load_tuning_results, get_best_params
from gnn_package import training
from gnn_package.config import get_config, ExperimentConfig

# Load complete tuning results
tuning_results = load_tuning_results("results/my_tuning_experiment")

# Or just get the best parameters
best_params = get_best_params("results/my_tuning_experiment")

# Update configuration with best parameters
config = get_config()  # Or load from file: ExperimentConfig("config.yml")
for param_name, value in best_params.items():
    parts = param_name.split('.')
    if len(parts) == 2:
        section, attribute = parts
        if hasattr(config, section) and hasattr(getattr(config, section), attribute):
            setattr(getattr(config, section), attribute, value)

# Train model with updated configuration
data_loaders = training.preprocess_data(
    data_file="path/to/data.pkl",
    config=config,
)

results = training.train_model(
    data_loaders=data_loaders,
    config=config,
)
```

## Key Parameters to Tune

The most important hyperparameters to tune are:

1. **Model Architecture**:
   - `model.hidden_dim`: Size of hidden layers (32-256)
   - `model.num_layers`: Number of layers in the GRU (1-3)
   - `model.num_gc_layers`: Number of graph convolutional layers (1-3)
   - `model.dropout`: Dropout rate (0.1-0.5)

2. **Training Parameters**:
   - `training.learning_rate`: Learning rate (1e-4 to 1e-2)
   - `training.weight_decay`: Weight decay for regularization (1e-6 to 1e-3)

For more complex models, you may also want to tune:
- `model.attention_heads`: Number of attention heads
- `model.gcn_normalization`: Normalization method for GCN layers

## Customizing the Parameter Space

You can define your own parameter space by creating a dictionary of parameter suggestions:

```python
import optuna
from gnn_package.src.tuning import tune_hyperparameters

# Define custom parameter space
custom_param_space = {
    "model.hidden_dim": lambda trial: trial.suggest_categorical(
        "model.hidden_dim", [64, 128, 256]
    ),
    "model.dropout": lambda trial: trial.suggest_float(
        "model.dropout", 0.1, 0.4
    ),
    "training.learning_rate": lambda trial: trial.suggest_float(
        "training.learning_rate", 1e-4, 1e-2, log=True
    ),
}

# Run tuning with custom parameter space
results = tune_hyperparameters(
    data_file="path/to/data.pkl",
    experiment_name="custom_param_space",
    param_space=custom_param_space,
)
```

## Understanding Tuning Results

The tuning process generates several output files in the specified directory:

- `best_params.json`: Best hyperparameters found
- `best_config.yml`: Full configuration file with best parameters
- `trials_summary.txt`: Human-readable summary of all trials
- `all_trials.csv`: Detailed CSV of all trial results
- `best_trial_report.json`: Detailed information about the best trial
- `optimization_history.png`: Plot of the optimization progress
- `param_importances.png`: Plot showing the importance of each parameter
- `parallel_coordinate.png`: Parallel coordinate plot for parameter relationships

In the `best_model` subdirectory:
- `best_model.pth`: Trained model with best parameters
- `loss_curves.png`: Training and validation loss curves
- `training_curves.pkl`: Raw data for loss curves

## MLflow Integration

The tuning module integrates with MLflow for experiment tracking:

1. All trials are recorded in MLflow experiments
2. Parameters, metrics, and artifacts are logged
3. You can view experiments using the MLflow UI:

```bash
mlflow ui --backend-store-uri file:///path/to/results/tuning/experiment_name/mlruns
```

## Tips for Effective Tuning

1. **Start small**: Use a smaller dataset (e.g., 1 week instead of 1 year) for initial tuning
2. **Use multi-stage tuning**: Start with a small data fraction and increase gradually
3. **Reduce epochs for exploration**: Use fewer epochs in early trials to quickly explore the parameter space
4. **Focus on important parameters**: Use the parameter importance plot to identify key parameters
5. **Save computational resources**: Use `--quick` flag for testing, then run full tuning
6. **Keep old results**: Each tuning run creates a new experiment directory, allowing you to compare different runs