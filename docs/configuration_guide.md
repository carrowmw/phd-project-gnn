# GNN Package Configuration Guide

This guide explains how to use the centralized configuration system to manage all parameters in the GNN package.

## Introduction

The centralized configuration system is designed to:

1. Organize all parameters in a hierarchical, type-safe structure
2. Provide default values while allowing easy overrides
3. Support saving and loading configurations from YAML files
4. Make experiments reproducible by recording all settings

## Configuration Structure

The configuration is organized into the following sections:

- **Experiment Metadata**: Name, description, version, tags
- **Data Configuration**: Parameters for data preprocessing and handling
- **Model Configuration**: Parameters for the GNN model architecture
- **Training Configuration**: Parameters for the training process
- **Paths Configuration**: File paths for saving/loading data
- **Visualization Configuration**: Parameters for visualizations

## Using the Configuration System

### Loading Configuration

```python
from gnn_package.config import get_config, ExperimentConfig

# Get the global configuration (creates default if none exists)
config = get_config()

# Or load from a specific file
config = ExperimentConfig("path/to/config.yml")
```

### Creating a Default Configuration

```python
from gnn_package.config import create_default_config

# Create a default configuration and save it to a file
config = create_default_config("config.yml")
```

### Accessing Configuration Values

```python
# Access through class attributes (type-safe)
window_size = config.data.window_size
learning_rate = config.training.learning_rate

# Or access through dictionary-like interface
window_size = config.get("data.window_size")
```

### Using Configuration in Functions

Most functions have been updated to accept a `config` parameter:

```python
from gnn_package import training

# Load and preprocess data with configuration
data_loaders = training.preprocess_data(
    data_file="my_data.pkl",
    config=config,
)

# Train model with configuration
results = training.train_model(
    data_loaders=data_loaders,
    config=config,
)
```

### Overriding Configuration Values

You can override specific configuration values when calling functions:

```python
# Override window_size and batch_size for this run only
data_loaders = training.preprocess_data(
    data_file="my_data.pkl",
    config=config,
    window_size=36,  # Override the config.data.window_size value
    batch_size=64,   # Override the config.data.batch_size value
)
```

## Sample Configuration File

Here's a sample YAML configuration file:

```yaml
experiment:
  name: "Traffic Prediction Experiment"
  description: "Predicting traffic using spatial-temporal GNN"
  version: "1.0.0"
  tags: ["traffic", "gnn", "prediction"]

data:
  start_date: "2024-02-18 00:00:00"
  end_date: "2024-02-25 00:00:00"
  graph_prefix: "25022025_test"
  window_size: 24
  horizon: 6
  batch_size: 32
  days_back: 14
  stride: 1
  gap_threshold_minutes: 15
  standardize: true
  sigma_squared: 0.1
  epsilon: 0.5
  max_distance: 100.0
  tolerance_decimal_places: 6
  resampling_frequency: "15min"
  missing_value: -1.0

model:
  input_dim: 1
  hidden_dim: 64
  output_dim: 1
  num_layers: 2
  dropout: 0.2

training:
  learning_rate: 0.001
  weight_decay: 0.00001
  num_epochs: 50
  patience: 10
  train_val_split: 0.8
  device: null  # Will auto-detect if null

paths:
  model_save_path: "models"
  data_cache: "data/cache"
  results_dir: "results"

visualization:
  dashboard_template: "dashboard.html"
  default_sensors_to_plot: 6
  max_sensors_in_heatmap: 50
```

## Running an Experiment

For convenience, a command-line script is provided to run experiments:

```bash
# Create a default configuration
python train_model.py --create-config --config my_config.yml

# Edit my_config.yml as needed, then run training
python train_model.py --config my_config.yml --data path/to/data.pkl --output results/experiment1
```

## Best Practices

1. **Create configurations for different experiments**: Keep separate config files for different experiments to make them reproducible.

2. **Version control your configurations**: Store your config files in version control to track changes.

3. **Save configurations with results**: Always save the configuration used for a particular run alongside the results.

4. **Use kwargs for quick overrides**: Use keyword arguments to override specific parameters when needed without changing the config file.

5. **Add new parameters to the configuration**: When adding new functionality, extend the configuration classes rather than using hard-coded values.

## Extending the Configuration

To add new parameters to the configuration system:

1. Add the parameter to the appropriate dataclass in `config.py`
2. Update the default configuration in `config_manager.py`
3. Update functions to use the configuration parameter

Example:

```python
# In config.py
@dataclass
class DataConfig:
    # Existing parameters...
    new_parameter: float = 0.5  # Add with a default value

# In your function
def my_function(config=None, new_parameter=None):
    # Get configuration
    if config is None:
        config = get_config()

    # Use parameter or config value
    param = new_parameter if new_parameter is not None else config.data.new_parameter

    # Use the parameter
    # ...
```