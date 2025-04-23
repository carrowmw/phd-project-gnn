# GNN Package Overview Documentation

The GNN package is a comprehensive framework for traffic prediction using spatio-temporal graph neural networks. This overview provides a high-level understanding of the package's components, architecture, and workflow.

## Package Purpose and Scope

The GNN package is designed to:

1. Process and analyse traffic sensor data with spatial relationships
2. Build and train graph neural networks for traffic prediction
3. Make accurate predictions for future traffic patterns
4. Visualise and evaluate prediction results
5. Provide operational services for real-time traffic forecasting

## Package Architecture

The package follows a modular architecture with clear separation of concerns:

```mermaid
flowchart TD
    subgraph Core Components
        A[Configuration Module]
        B[Data Processing]
        C[Model Architecture]
        D[Training Pipeline]
        E[Prediction Pipeline]
        F[Visualisation]
    end

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F

    B --> D
    B --> E

    C --> D
    C --> E

    D --> E

    E --> F
```

### Layer Structure

The package is organised in logical layers:

1. **Configuration Layer**: Manages all settings and parameters
2. **Data Layer**: Handles data loading, processing, and transformation
3. **Model Layer**: Defines model architecture and components
4. **Training Layer**: Manages model training and hyperparameter tuning
5. **Prediction Layer**: Provides inference and evaluation capabilities
6. **Visualisation Layer**: Creates plots and dashboards of results

## Key Components

### Configuration System

The configuration system provides centralised management of all parameters:

```mermaid
classDiagram
    class ExperimentConfig {
        +experiment: ExperimentMetadata
        +data: DataConfig
        +model: ModelConfig
        +training: TrainingConfig
        +paths: PathsConfig
        +visualization: VisualizationConfig
    }

    ExperimentConfig --> DataConfig
    ExperimentConfig --> ModelConfig
    ExperimentConfig --> TrainingConfig
```

Key features:

- Hierarchical organisation of parameters
- Loading from YAML files
- Parameter validation
- Centralised access throughout the package

### Data Processing Pipeline

The data pipeline handles both spatial and temporal aspects of the data:

```mermaid
flowchart LR
    A[Data Sources] --> B[Data Processors]
    B --> C[Time Series Processing]
    B --> D[Graph Processing]
    C --> E[Dataset Creation]
    D --> E
    E --> F[DataLoaders]
```

Key features:

- Support for file and API data sources
- Time series resampling and standardisation
- Graph creation from spatial relationships
- Integration of temporal and spatial features

### Model Architecture

The model architecture is based on a spatio-temporal graph neural network:

```mermaid
flowchart TB
    A[Input Data] --> B[Graph Convolution Layers]
    B --> C[GRU Layers]
    C --> D[Decoder Layers]
    D --> E[Prediction Output]
```

Key features:

- Graph convolution for spatial dependencies
- Recurrent layers for temporal patterns
- Attention mechanisms for focusing on relevant features
- Modular design for extensibility

### Training Pipeline

The training pipeline manages the model training process:

```mermaid
flowchart TD
    A[Data Preparation] --> B[Model Creation]
    B --> C[Training Loop]
    C --> D[Validation]
    D --> E[Early Stopping]
    E --> F[Model Saving]
```

Key features:

- Support for cross-validation
- Early stopping for optimal training
- Progress tracking with tqdm
- Comprehensive logging of metrics

### Prediction Pipeline

The prediction pipeline handles inference and evaluation:

```mermaid
flowchart LR
    A[Load Model] --> B[Fetch Data]
    B --> C[Generate Predictions]
    C --> D[Evaluate Results]
    D --> E[Visualise Outputs]
```

Key features:

- End-to-end prediction workflow
- Comparison with ground truth
- Calculation of error metrics
- Comprehensive visualisation options

### Hyperparameter Tuning

The package includes a sophisticated hyperparameter tuning system:

```mermaid
flowchart TD
    A[Define Parameter Space] --> B[Create Optuna Study]
    B --> C[Run Optimisation]
    C --> D[Analyse Results]
    D --> E[Train with Best Parameters]
```

Key features:

- Integration with Optuna and MLflow
- Multi-stage tuning for efficiency
- Visualisation of parameter importance
- Results tracking and comparison

### Visualisation Components

The visualisation system provides comprehensive insights into predictions:

```mermaid
flowchart TB
    A[Prediction Results] --> B[Time Series Plots]
    A --> C[Error Distribution]
    A --> D[Sensor Grid Visualization]
    A --> E[Interactive Dashboard]
```

Key features:

- Time series plots for individual sensors
- Error analysis and distribution visualisation
- Grid views for comparing multiple sensors
- Optional interactive dashboards

## Main Workflows

### Training Workflow

```mermaid
sequenceDiagram
    participant User
    participant Config as Configuration
    participant Data as Data Processing
    participant Model as Model Creation
    participant Train as Training Pipeline

    User->>Config: Create/load config
    User->>Data: Prepare data
    Data->>Config: Get processing parameters
    User->>Model: Create model
    Model->>Config: Get architecture parameters
    User->>Train: Train model
    Train->>Data: Get processed data
    Train->>Model: Update weights
    Train->>Config: Get training parameters
    Train->>User: Return trained model
```

### Prediction Workflow

```mermaid
sequenceDiagram
    participant User
    participant Model as Model Loading
    participant Data as Data Fetching
    participant Pred as Prediction
    participant Vis as Visualisation

    User->>Model: Load trained model
    User->>Data: Fetch recent data
    Data->>Pred: Preprocess for prediction
    Model->>Pred: Generate predictions
    Pred->>Vis: Create visualisation
    Vis->>User: Return visualisation
    Pred->>User: Return predictions
```

### Tuning Workflow

```mermaid
sequenceDiagram
    participant User
    participant Tune as Tuning System
    participant Train as Training
    participant Eval as Evaluation

    User->>Tune: Define parameter space
    Tune->>Train: Train with parameters 1
    Train->>Eval: Evaluate performance
    Eval->>Tune: Return metrics
    Tune->>Train: Train with parameters 2
    Train->>Eval: Evaluate performance
    Eval->>Tune: Return metrics
    note over Tune: Repeat for N trials
    Tune->>User: Return best parameters
```

## Package Integration

The GNN package integrates with several external libraries and systems:

```mermaid
flowchart LR
    subgraph GNN Package
        A[Core Components]
    end

    subgraph External Libraries
        B[PyTorch]
        C[NetworkX/OSMnx]
        D[Pandas/GeoPandas]
        E[Matplotlib/Plotly]
        F[Optuna/MLflow]
    end

    subgraph Integration Points
        G[private_uoapi]
        H[Filesystem]
        I[Web Dashboard]
    end

    A --- B
    A --- C
    A --- D
    A --- E
    A --- F

    A --- G
    A --- H
    A --- I
```

Key integration points:

- PyTorch for deep learning models
- NetworkX/OSMnx for graph operations
- Pandas/GeoPandas for data manipulation
- Matplotlib/Plotly for visualisation
- Optuna/MLflow for hyperparameter tuning
- private_uoapi for data fetching

## Directory Structure

The package is organised into the following directory structure:

```
gnn_package/
├── __init__.py
├── __main__.py
├── config.yml
├── config/
│   ├── config.py
│   ├── config_manager.py
│   └── paths.py
├── data/
│   ├── preprocessed/
│   ├── raw/
│   └── sensors/
├── src/
│   ├── data/
│   ├── dataloaders/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   ├── tuning/
│   ├── utils/
│   └── visualization/
├── prediction_service.py
└── run_experiment.py
```

## Component Interactions

The interactions between key package components can be visualised as:

```mermaid
classDiagram
    class ConfigModule {
        +get_config()
        +ExperimentConfig
    }

    class DataModule {
        +DataSource
        +DataProcessor
        +TimeSeriesPreprocessor
    }

    class ModelModule {
        +STGNN
        +GraphConvolution
        +create_stgnn_model()
    }

    class TrainingModule {
        +train_model()
        +STGNNTrainer
        +preprocess_data()
    }

    class PredictionModule {
        +predict_with_model()
        +load_model()
        +predict_all_sensors_with_validation()
    }

    class TuningModule {
        +tune_hyperparameters()
        +run_multi_stage_tuning()
    }

    class VisualisationModule {
        +plot_predictions_with_validation()
        +plot_sensors_grid()
        +plot_error_distribution()
    }

    ConfigModule --> DataModule
    ConfigModule --> ModelModule
    ConfigModule --> TrainingModule
    ConfigModule --> PredictionModule
    ConfigModule --> TuningModule

    DataModule --> TrainingModule
    DataModule --> PredictionModule

    ModelModule --> TrainingModule
    ModelModule --> PredictionModule

    TrainingModule --> PredictionModule

    PredictionModule --> VisualisationModule

    TuningModule --> TrainingModule
```

## Entry Points

The package provides several entry points for different use cases:

1. **run_experiment.py**: Main script for running training experiments
   ```bash
   python run_experiment.py --config config.yml --data data/raw/timeseries/test_data_1wk.pkl
   ```

2. **prediction_service.py**: Service for making predictions
   ```bash
   python prediction_service.py results/test_1wk/model.pth predictions/
   ```

3. **tune_model.py**: Script for hyperparameter tuning
   ```bash
   python tune_model.py --data data/raw/timeseries/test_data_1wk.pkl --trials 20
   ```

4. **__main__.py**: Module entry point for default training
   ```bash
   python -m gnn_package --config config.yml --data data/raw/timeseries/test_data_1mnth.pkl
   ```

## Getting Started

To use the GNN package:

1. **Setup Configuration**:
   ```python
   from gnn_package.config import get_config, create_default_config

   # Create a default config
   config = create_default_config("my_config.yml")

   # Or load existing config
   config = get_config("my_config.yml")
   ```

2. **Process Data**:
   ```python
   from gnn_package import training

   # Preprocess data
   data_loaders = await training.preprocess_data(
       data_file="data/raw/timeseries/test_data.pkl",
       config=config
   )
   ```

3. **Train Model**:
   ```python
   # Train model
   results = training.train_model(
       data_loaders=data_loaders,
       config=config
   )

   # Save model
   torch.save(results["model"].state_dict(), "model.pth")
   ```

4. **Make Predictions**:
   ```python
   from gnn_package.training import predict_all_sensors_with_validation

   # Predict
   predictions = await predict_all_sensors_with_validation(
       model_path="model.pth",
       config=config
   )
   ```

## Key Features and Capabilities

1. **Data Handling**:
   - Support for multiple data sources
   - Handling of missing values
   - Time series resampling and standardisation
   - Graph structure creation from spatial data

2. **Model Architecture**:
   - Graph convolution for spatial relationships
   - Recurrent layers for temporal patterns
   - Attention mechanisms for feature importance
   - Configurable architecture parameters

3. **Training**:
   - Cross-validation support
   - Early stopping
   - Progress tracking
   - Comprehensive metric logging

4. **Prediction**:
   - Real-time prediction capabilities
   - Comprehensive validation
   - Detailed error metrics
   - Visualisation of results

5. **Hyperparameter Tuning**:
   - Automated parameter optimisation
   - Multi-stage tuning
   - Integration with MLflow
   - Parameter importance analysis

## Extension Points

The package is designed to be extensible in several ways:

1. **Custom Data Sources**:
   - Implement the `DataSource` abstract class
   - Override the `get_data` method
   - Register with the `DataProcessorFactory`

2. **Alternative Model Architectures**:
   - Create a new model class
   - Implement a factory function
   - Update the configuration schema

3. **Custom Visualisations**:
   - Add new visualisation functions
   - Extend the dashboard generation
   - Implement interactive components

4. **Additional Metrics**:
   - Add new evaluation metrics
   - Extend the prediction analysis
   - Implement custom performance scoring

## Conclusion

The GNN package provides a comprehensive framework for traffic prediction using graph neural networks. Its modular architecture, extensive configuration options, and integrated workflows make it suitable for both research and operational use. By following the documentation for each component, users can leverage the full capabilities of the package for their specific needs.