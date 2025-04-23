# GNN Package Configuration Module Documentation

The configuration module in the GNN package provides a centralised system for managing settings across the entire application. It uses a hierarchical structure to organise configuration parameters and supports loading configurations from YAML files, overriding settings programmatically, and validating settings.

## Configuration Architecture Overview

The configuration system consists of several key components:

1. `ExperimentConfig`: The main configuration class that loads, validates, and provides access to configuration settings
2. Configuration dataclasses: Specialised dataclasses for different configuration categories (`DataConfig`, `ModelConfig`, etc.)
3. `ConfigManager`: Utilities for working with configurations (loading, creating defaults, etc.)

Let's examine how these components work together and how the rest of the package interacts with them.

## Configuration Hierarchy

The configuration follows a hierarchical structure to organise settings by domain:

```mermaid
classDiagram
    class ExperimentConfig {
        +experiment: ExperimentMetadata
        +data: DataConfig
        +model: ModelConfig
        +training: TrainingConfig
        +paths: PathsConfig
        +visualization: VisualizationConfig
        +validate()
        +log()
        +save()
        +get()
    }

    class ExperimentMetadata {
        +name: str
        +description: str
        +version: str
        +tags: List[str]
    }

    class DataConfig {
        +general: GeneralDataConfig
        +training: TrainingDataConfig
        +prediction: PredictionDataConfig
    }

    class GeneralDataConfig {
        +window_size: int
        +horizon: int
        +stride: int
        +batch_size: int
        +standardize: bool
        +missing_value: float
        +resampling_frequency: str
        +...other parameters
    }

    class TrainingDataConfig {
        +start_date: str
        +end_date: str
        +n_splits: int
        +use_cross_validation: bool
        +split_method: str
        +train_ratio: float
        +cutoff_date: str
        +cv_split_index: int
    }

    class PredictionDataConfig {
        +days_back: int
    }

    class ModelConfig {
        +input_dim: int
        +hidden_dim: int
        +output_dim: int
        +num_layers: int
        +dropout: float
        +num_gc_layers: int
        +...other parameters
    }

    class TrainingConfig {
        +learning_rate: float
        +weight_decay: float
        +num_epochs: int
        +patience: int
        +device: str
    }

    class PathsConfig {
        +model_save_path: str
        +data_cache: str
        +results_dir: str
    }

    class VisualizationConfig {
        +dashboard_template: str
        +default_sensors_to_plot: int
        +max_sensors_in_heatmap: int
    }

    ExperimentConfig *-- ExperimentMetadata
    ExperimentConfig *-- DataConfig
    ExperimentConfig *-- ModelConfig
    ExperimentConfig *-- TrainingConfig
    ExperimentConfig *-- PathsConfig
    ExperimentConfig *-- VisualizationConfig
    DataConfig *-- GeneralDataConfig
    DataConfig *-- TrainingDataConfig
    DataConfig *-- PredictionDataConfig
```


## Key Components

### `ExperimentConfig`

The `ExperimentConfig` class is the main entry point for accessing configuration settings. It loads a YAML configuration file and initialises the various configuration dataclasses.

#### Key methods:

* `__init__(config_path)`: Loads configuration from a YAML file
* `validate()`: Checks that all required parameters are present
* `log()`: Logs the configuration details
* `save(path)`: Saves the configuration to a YAML file
* `get(key, default)`: Accesses configuration values using dot notation

#### Example usage:

```python
# Load a configuration
config = ExperimentConfig('config.yml')

# Access configuration values
window_size = config.data.general.window_size
learning_rate = config.training.learning_rate

# Save updated configuration
config.save('updated_config.yml')
```

### Configuration Dataclasses

Specialised dataclasses for different configuration domains provide type hints and structured access to settings.

#### Example:

```python
@dataclass
class GeneralDataConfig:
    """Configuration for data processing shared across training and prediction."""
    window_size: int = 24
    horizon: int = 6
    stride: int = 1
    # ...more parameters
```

### `ConfigManager`

The `config_manager.py` module provides utilities for working with configurations:

* `get_config()`: Gets the global configuration instance
* `reset_config()`: Resets the global configuration instance
* `create_default_config()`: Creates a default configuration file
* `load_yaml_config()`: Loads a YAML configuration file

## Global Configuration Singleton

The configuration system implements a singleton pattern to maintain a single, global configuration instance:

```mermaid
sequenceDiagram
    participant A as Component A
    participant B as Component B
    participant CM as ConfigManager
    participant GC as Global Config Instance

    Note over A, B: First request for config
    A->>CM: get_config()
    CM->>GC: Check if exists
    GC-->>CM: Does not exist
    CM->>GC: Create new instance
    GC-->>CM: Return new instance
    CM-->>A: Return config

    Note over A, B: Subsequent request
    B->>CM: get_config()
    CM->>GC: Check if exists
    GC-->>CM: Instance exists
    CM-->>B: Return existing config
```

## Interactions with Other Modules

The configuration system interacts with many other modules in the package:

```mermaid
flowchart LR
    A[Configuration Module] --> B[Model Creation]
    A --> C[Data Preprocessing]
    A --> D[Training Pipeline]
    A --> E[Prediction Service]
    A --> F[Hyperparameter Tuning]
    A --> G[Visualization Components]

    subgraph Model Subsystem
        B --> H[STGNN Model]
        B --> I[GraphConvolution]
    end

    subgraph Data Subsystem
        C --> J[TimeSeriesPreprocessor]
        C --> K[Graph Utilities]
    end

    subgraph Training Subsystem
        D --> L[STGNNTrainer]
        D --> M[DataLoaders]
    end
```

### Data Preprocessing

The configuration guides how data is processed:

* Window size and horizon for time series segmentation
* Resampling frequency and normalisation parameters
* Train/validation split parameters

```python
# Example: Using config in TimeSeriesPreprocessor
processor = TimeSeriesPreprocessor(config=config)
split_data = processor.create_rolling_window_splits(resampled_data, config=config)
```

### Model Creation

Model architecture is defined by configuration parameters:

* Hidden dimensions, layers, and dropout
* Graph convolution parameters
* Input and output dimensions

```python
# Example: Creating a model from config
model = create_stgnn_model(config=config)
```

### Training

Training hyperparameters are stored in the configuration:

* Learning rate and weight decay
* Number of epochs and early stopping patience
* Device selection (CPU/GPU/MPS)

```python
# Example: Training with config
trainer = STGNNTrainer(model, config)
results = train_model(data_loaders=data_loaders, config=config)
```

### Prediction

Prediction services use configuration to determine:

* How far back to retrieve historical data
* Frequency of predictions
* Which model parameters to use


```python
# Example: Creating prediction config
prediction_config = create_prediction_config()
```

## Tips for Working with the Configuration System

Always use the global instance when possible:

```python
from gnn_package.config import get_config
config = get_config()
```

Access config properties consistently:

```python
# Good - direct attribute access
window_size = config.data.general.window_size

# Also good - using get() with default
window_size = config.get("data.general.window_size", default=24)
```

For prediction, create a specialised config:

```python
from gnn_package.src.utils.config_utils import create_prediction_config
prediction_config = create_prediction_config()
```

Validate after modifications:

```python
# After updating config settings
config.validate()
```

Save configuration alongside results:

```python
# When saving experiment results
config.save(output_dir / "config.yml")
```

## Under the Hood: Configuration Loading

When a configuration file is loaded, it follows this process:

```mermaid
sequenceDiagram
    participant App as Application
    participant EC as ExperimentConfig
    participant YL as YAML Loader
    participant DC as Dataclasses

    App->>EC: __init__(config_path)
    EC->>YL: Load YAML file
    YL-->>EC: Raw config dictionary

    EC->>DC: Initialize ExperimentMetadata
    EC->>DC: Initialize DataConfig
    EC->>DC: Initialize ModelConfig
    EC->>DC: Initialize TrainingConfig
    EC->>DC: Initialize PathsConfig
    EC->>DC: Initialize VisualizationConfig

    EC->>EC: validate()
    EC->>EC: log()
    EC-->>App: Return ExperimentConfig instance
```

## Summary
The configuration system in the GNN package provides a centralised, type-safe way to manage settings across the entire application. By using a hierarchical structure with specialised dataclasses, it ensures that components have access to the settings they need in a consistent and validated format.

This approach offers several benefits:

* Clear organisation of settings by domain
* Type hints and validation for configuration values
* Centralised management of global settings
* Easy serialisation to and from YAML files
* Support for both global singleton and explicit configuration instances

Understanding how this configuration system works will help you effectively customise and extend the GNN package for your specific needs.