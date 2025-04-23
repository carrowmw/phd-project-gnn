# GNN Package Command Line Guide

This guide covers how to use the GNN package from the command line, focusing on the standard workflow of experiment setup, training, prediction, and hyperparameter tuning. We'll explain the available commands, their arguments, and provide example usage patterns.

## Table of Contents

1. [Setup and Configuration](#setup-and-configuration)
2. [Running Experiments](#running-experiments)
3. [Training Models](#training-models)
4. [Making Predictions](#making-predictions)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Common Workflows](#common-workflows)
7. [Troubleshooting](#troubleshooting)

## Setup and Configuration

Before running experiments, you need to set up a configuration. While you can use the default configuration, it's recommended to create a custom one for your specific needs.

### Creating a Default Configuration

```bash
python -m gnn_package --create-config --config your_config.yml
```

**Arguments:**
- `--create-config`: Flag to create a default configuration file
- `--config`: Path where the configuration file will be saved

This will create a `your_config.yml` file with default settings that you can edit.

### Key Configuration Parameters

While you can edit the configuration file directly, some common parameters can be overridden from the command line:

```bash
python -m gnn_package \
    --config your_config.yml \
    --data data/raw/timeseries/test_data_1wk.pkl \
    --output results/my_experiment
```

**Arguments:**
- `--config`: Path to your configuration file
- `--data`: Path to your input data file
- `--output`: Directory where results will be saved

## Running Experiments

The main experiment script is `run_experiment.py`, which handles the complete pipeline from data loading to model training and evaluation.

### Basic Usage

```bash
python run_experiment.py \
    --config config.yml \
    --data data/raw/timeseries/test_data_1wk.pkl \
    --output results/my_experiment
```

**Arguments:**
- `--config`: Path to the configuration file
- `--data`: Path to the input data file
- `--output`: Directory where results will be saved

### Full Argument List

| Argument   | Description                  | Default                                     |
| ---------- | ---------------------------- | ------------------------------------------- |
| `--config` | Path to configuration file   | `config.yml` in current directory           |
| `--data`   | Path to input data file      | Based on config                             |
| `--output` | Output directory for results | `experiments/[experiment_name]_[timestamp]` |

### Example: Running with Time-Based Cross-Validation

```bash
python run_experiment.py \
    --config configs/time_cv.yml \
    --data data/raw/timeseries/test_data_1mnth.pkl \
    --output results/time_cv_experiment
```

## Training Models

For more control over the training process, you can use the package as a module directly or update configuration parameters.

### Running Training Only

```bash
python -m gnn_package \
    --config your_config.yml \
    --data data/raw/timeseries/test_data_1wk.pkl
```

**Key Training Parameters in Config:**
- `training.num_epochs`: Number of training epochs
- `training.learning_rate`: Initial learning rate
- `training.weight_decay`: Weight decay for regularisation
- `training.patience`: Patience for early stopping

### Advanced Training with Custom Parameters

If you need to modify specific training parameters without changing the config file, you can export them as environment variables:

```bash
export GNN_EPOCHS=100
export GNN_LEARNING_RATE=0.0005
python -m gnn_package \
    --config your_config.yml \
    --data data/raw/timeseries/test_data_1wk.pkl
```

The package will check for these environment variables and override the corresponding config values.

## Making Predictions

After training, you can use `prediction_service.py` to make predictions with a trained model.

### Basic Prediction

```bash
python prediction_service.py \
    results/my_experiment/model.pth \
    predictions/today
```

**Arguments:**

- First positional argument: Path to the trained model
- Second positional argument: Directory for saving prediction results
- Third positional argument (optional): `yes` or `no` to enable/disable visualisation (default: `yes`)

### Example: Prediction with Visualisation

```bash
python prediction_service.py \
    results/my_experiment/model.pth \
    predictions/today \
    yes
```

### Example: Prediction without Visualisation

```bash
python prediction_service.py \
    results/my_experiment/model.pth \
    predictions/today \
    no
```

### Understanding Prediction Outputs

Running the prediction service produces several outputs:

- `predictions_[timestamp].csv`: CSV file with predictions and actual values
- `summary_[timestamp].txt`: Text summary of prediction performance
- `sensors_grid_[timestamp].png`: Visualisation of predictions for all sensors
- `error_analysis_[timestamp].png`: Error distribution and analysis plots
- `dashboard_[timestamp].html`: Interactive dashboard (if visualisation is enabled)

## Hyperparameter Tuning

For optimising model performance, use `tune_model.py` to perform hyperparameter tuning.

### Basic Tuning

```bash
python tune_model.py \
    --data data/raw/timeseries/test_data_1wk.pkl \
    --trials 20 \
    --output results/tuning/experiment1
```

**Required Arguments:**

- `--data`: Path to the data file (required)

**Optional Arguments:**

- `--config`: Path to a custom config file
- `--output`: Directory to save tuning results
- `--trials`: Number of trials to run (default: 20)
- `--epochs`: Number of epochs per trial (overrides config)
- `--experiment`: Name of the experiment

### Multi-Stage Tuning

Multi-stage tuning is more efficient as it starts with less data and progressively increases both data size and training epochs:

```bash
python tune_model.py \
    --data data/raw/timeseries/test_data_1wk.pkl \
    --multi-stage \
    --output results/tuning/multi_stage
```

**Additional Arguments for Multi-Stage Tuning:**

- `--multi-stage`: Flag to enable multi-stage tuning
- `--quick`: Run a quick tuning with fewer trials and epochs (for testing)

### Example: Quick Tuning for Testing

```bash
python tune_model.py \
    --data data/raw/timeseries/test_data_1wk.pkl \
    --quick \
    --output results/tuning/quick_test
```

### Understanding Tuning Outputs

Tuning generates several outputs:

- `best_params.json`: JSON file with the best parameters
- `best_config.yml`: Configuration file with the best parameters
- `optimization_history.png`: Plot of the optimisation history
- `param_importances.png`: Visualisation of parameter importance
- `trials_summary.txt`: Text summary of all trials
- `all_trials.csv`: CSV file with details of all trials

In the `best_model` directory:

- `model.pth`: Best model trained with the optimal parameters
- `config.yml`: Configuration used for the best model
- `loss_curves.png`: Training and validation loss curves

## Common Workflows

### Complete Experiment Workflow

```bash
# 1. Create a custom configuration
python -m gnn_package --create-config --config my_config.yml

# 2. Edit the configuration (adjust parameters as needed)
# vim my_config.yml

# 3. Run a training experiment
python run_experiment.py \
    --config my_config.yml \
    --data data/raw/timeseries/test_data_1wk.pkl \
    --output results/experiment1

# 4. Make predictions with the trained model
python prediction_service.py \
    results/experiment1/model.pth \
    predictions/experiment1
```

### Hyperparameter Tuning Workflow

```bash
# 1. Run multi-stage tuning to find optimal parameters
python tune_model.py \
    --data data/raw/timeseries/test_data_1wk.pkl \
    --multi-stage \
    --output results/tuning/multi_stage

# 2. Use the best configuration for a full training run
python run_experiment.py \
    --config results/tuning/multi_stage/best_config.yml \
    --data data/raw/timeseries/test_data_1wk.pkl \
    --output results/tuned_experiment

# 3. Make predictions with the tuned model
python prediction_service.py \
    results/tuned_experiment/model.pth \
    predictions/tuned_experiment
```

### Production Deployment Workflow

```bash
# 1. Train with production data
python run_experiment.py \
    --config production_config.yml \
    --data data/raw/timeseries/production_data.pkl \
    --output models/production

# 2. Set up a cron job for regular predictions
# 0 * * * * python /path/to/prediction_service.py /path/to/models/production/model.pth /path/to/predictions/$(date +\%Y-\%m-\%d)
```

## Troubleshooting

### Common Issues and Solutions

1. **Configuration Errors**

   If you see errors related to missing configuration values:

   ```bash
   # Validate your configuration file
   python -c "from gnn_package.config import ExperimentConfig; config = ExperimentConfig('your_config.yml'); config.validate()"
   ```

2. **CUDA/GPU Issues**

   If you encounter GPU-related errors:

   ```bash
   # Force CPU usage by setting the device in config.yml
   training:
     device: "cpu"
   ```

3. **Data Loading Errors**

   For data loading issues:

   ```bash
   # Check data file integrity
   python -c "import pickle; with open('your_data_file.pkl', 'rb') as f: data = pickle.load(f); print(type(data), len(data))"
   ```

4. **Out of Memory Errors**

   If you encounter memory issues:

   ```bash
   # Reduce batch size in your config
   data:
     general:
       batch_size: 16  # Try a smaller value
   ```

### Getting Help

For more detailed information about a specific script, use the help flag:

```bash
python run_experiment.py --help
python prediction_service.py --help
python tune_model.py --help
```

Each script will display its available arguments and their descriptions.
