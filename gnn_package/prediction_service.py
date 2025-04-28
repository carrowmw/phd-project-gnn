#!/usr/bin/env python
# prediction_service.py

import os
import sys
import asyncio
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from gnn_package import training
from gnn_package.config import ExperimentConfig, create_prediction_config
from gnn_package.src.visualization.visualization_utils import VisualizationManager
from gnn_package.src.utils.metrics import (
    calculate_metrics_by_horizon,
    calculate_metrics_by_sensor,
)
from gnn_package.src.utils.device_utils import get_device
from gnn_package.src.utils.retry_utils import retry
from gnn_package.src.utils.exceptions import ModelLoadError, APIConnectionError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("prediction_service.log"),
    ],
)
logger = logging.getLogger("prediction-service")


@retry(max_retries=3, retry_delay=1.0, exceptions=(APIConnectionError,))
async def run_prediction_service(
    model_path,
    output_dir=None,
    visualize=True,
    cache_results=True,
    detailed_metrics=True,
    device=None,
):
    """
    Run the prediction service with enhanced features.

    Parameters:
    -----------
    model_path : str or Path
        Path to the trained model or directory containing models
    output_dir : str or Path, optional
        Directory to save predictions (defaults to 'predictions/YYYY-MM-DD')
    visualize : bool
        Whether to generate plots and visualizations
    cache_results : bool
        Whether to cache prediction results for future reuse
    detailed_metrics : bool
        Whether to calculate and save detailed metrics by sensor and horizon
    device : str, optional
        Device to run predictions on (auto-detected if None)

    Returns:
    --------
    dict
        Dictionary containing prediction results and metadata
    """
    try:
        start_time = datetime.now()

        # Create prediction configuration
        logger.info("Setting up prediction configuration")
        prediction_config = _setup_prediction_config(model_path)

        # Set up output directory
        if output_dir is None:
            today = datetime.now().strftime("%Y-%m-%d")
            output_dir = f"predictions/{today}"

        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamps for consistent file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"predictions_{timestamp}.csv"

        # Set up device
        if device is None:
            device = get_device(prediction_config.training.device)
        logger.info(f"Using device: {device}")

        # Run prediction
        logger.info(f"Running prediction using model: {model_path}")
        logger.info(f"Output will be saved to: {output_file}")

        predictions = await training.predict_all_sensors_with_validation(
            model_path=model_path,
            config=prediction_config,
            output_file=output_file,
            plot=False,  # Disable internal plotting, we'll handle it separately
        )

        # Check for valid predictions
        if not predictions or "dataframe" not in predictions:
            logger.error("Prediction failed or returned no results")
            return {"success": False, "error": "No predictions generated"}

        # Process prediction results
        predictions_df = predictions["dataframe"]
        logger.info(
            f"Generated {len(predictions_df)} predictions for {predictions_df['node_id'].nunique()} sensors"
        )

        # Calculate standard metrics
        metrics = _calculate_prediction_metrics(predictions_df)

        # Calculate detailed metrics if requested
        if detailed_metrics:
            metrics.update(_calculate_detailed_metrics(predictions_df))

        # Create and save summary report
        summary_file = output_dir / f"summary_{timestamp}.txt"
        _save_summary_report(predictions, metrics, summary_file, start_time)

        # Generate visualizations if requested
        viz_paths = {}
        if visualize:
            viz_paths = _generate_visualizations(predictions_df, output_dir, timestamp)

        # Cache results if requested
        if cache_results:
            cache_file = output_dir / f"cache_{timestamp}.pkl"
            _cache_prediction_results(predictions, cache_file)

        # Return combined results
        return {
            "success": True,
            "predictions_file": str(output_file),
            "summary_file": str(summary_file),
            "metrics": metrics,
            "visualizations": viz_paths if visualize else None,
            "execution_time": (datetime.now() - start_time).total_seconds(),
        }

    except ModelLoadError as e:
        logger.exception(f"Error loading model: {str(e)}")
        return {"success": False, "error": f"Model loading error: {str(e)}"}
    except APIConnectionError as e:
        logger.exception(f"Error connecting to data API: {str(e)}")
        return {"success": False, "error": f"API connection error: {str(e)}"}
    except Exception as e:
        logger.exception(f"Error in prediction service: {str(e)}")
        return {"success": False, "error": str(e)}


def _setup_prediction_config(model_path):
    """Set up prediction configuration from model path."""
    model_path = Path(model_path)

    # Try to load config from model directory first
    config_path = model_path.parent / "config.yml"

    if config_path.exists():
        logger.info(f"Loading configuration from model directory: {config_path}")
        # IMPORTANT: Create a fresh config directly from file WITHOUT
        # using get_config() which might return a shared singleton
        config = ExperimentConfig(str(config_path), is_prediction_mode=True)

        # Don't use create_prediction_config() which might use the global config
        # Instead, manually apply the minimal changes needed for prediction
        config.data.training.use_cross_validation = False
        config.data.prediction.days_back = max(1, config.data.prediction.days_back)

        logger.info(f"Using configuration from model with prediction mode enabled")
        return config
    else:
        logger.info("Creating default prediction configuration")
        # Create a fresh config without relying on the global instance
        default_config = ExperimentConfig("config.yml", is_prediction_mode=True)
        return default_config



def _calculate_prediction_metrics(predictions_df):
    """Calculate standard prediction metrics."""
    metrics = {}

    if "error" in predictions_df.columns:
        # Basic metrics
        metrics["mse"] = (predictions_df["error"] ** 2).mean()
        metrics["mae"] = predictions_df["abs_error"].mean()
        metrics["rmse"] = metrics["mse"] ** 0.5

        # Additional statistics
        metrics["max_error"] = predictions_df["abs_error"].max()
        metrics["min_error"] = predictions_df["abs_error"].min()
        metrics["median_error"] = predictions_df["abs_error"].median()

    return metrics


def _calculate_detailed_metrics(predictions_df):
    """Calculate detailed metrics by horizon and sensor."""
    detailed_metrics = {}

    # Calculate metrics by horizon
    horizon_metrics = calculate_metrics_by_horizon(predictions_df)
    detailed_metrics["horizon_metrics"] = horizon_metrics.to_dict()

    # Calculate metrics by sensor (top 10 worst performing)
    sensor_metrics = calculate_metrics_by_sensor(predictions_df, top_n=10)
    detailed_metrics["sensor_metrics"] = sensor_metrics.to_dict()

    return detailed_metrics


def _save_summary_report(predictions, metrics, summary_file, start_time):
    """Create and save a detailed summary report."""
    with open(summary_file, "w") as f:
        f.write(f"Prediction Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Date/Time: {datetime.now()}\n")
        f.write(
            f"Execution time: {(datetime.now() - start_time).total_seconds():.2f} seconds\n\n"
        )

        # Extract standardization stats if available
        standardization_stats = {}
        if "data" in predictions and "metadata" in predictions["data"]:
            metadata = predictions["data"]["metadata"]
            standardization_stats = metadata.get("preprocessing_stats", {}).get(
                "standardization", {}
            )

        # Write standardization information
        f.write(f"Standardization mean: {standardization_stats.get('mean', 'N/A')}\n")
        f.write(f"Standardization std: {standardization_stats.get('std', 'N/A')}\n\n")

        # Write prediction statistics
        df = predictions["dataframe"]
        f.write(f"Total predictions: {len(df)}\n")
        f.write(f"Total sensors: {df['node_id'].nunique()}\n\n")

        # Write error metrics
        f.write(f"Overall metrics:\n")
        f.write(f"  MSE: {metrics['mse']:.4f}\n")
        f.write(f"  MAE: {metrics['mae']:.4f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.4f}\n\n")

        # Write horizon-specific metrics if available
        if "horizon_metrics" in metrics:
            f.write("Metrics by prediction horizon:\n")
            horizon_df = pd.DataFrame(metrics["horizon_metrics"])
            f.write(horizon_df.to_string() + "\n\n")

        # Write sensor-specific metrics if available
        if "sensor_metrics" in metrics:
            f.write("Top 10 sensors by error:\n")
            sensor_df = pd.DataFrame(metrics["sensor_metrics"])
            f.write(sensor_df.to_string() + "\n")


def _generate_visualizations(predictions_df, output_dir, timestamp):
    """Generate and save visualizations for prediction results."""
    try:
        logger.info("Generating visualizations...")
        viz_manager = VisualizationManager()
        viz_paths = viz_manager.save_visualization_pack(
            predictions_df=predictions_df,
            output_dir=output_dir,
            timestamp=timestamp,
        )

        logger.info(f"Visualizations saved to: {output_dir}")
        for viz_type, path in viz_paths.items():
            logger.info(f"  - {viz_type}: {path}")

        return viz_paths
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return {}


def _cache_prediction_results(predictions, cache_file):
    """Cache prediction results for future reuse."""
    try:
        # Extract and save essential components only
        cache_data = {
            "dataframe": predictions["dataframe"],
            "metadata": predictions.get("metadata", {}),
            "timestamp": datetime.now().isoformat(),
        }

        with open(cache_file, "wb") as f:
            import pickle

            pickle.dump(cache_data, f)

        logger.info(f"Prediction results cached to: {cache_file}")
        return True
    except Exception as e:
        logger.warning(f"Failed to cache prediction results: {e}")
        return False


async def batch_predict(model_paths, output_base_dir=None, visualize=True):
    """
    Run predictions on multiple models in batch mode.

    Parameters:
    -----------
    model_paths : list
        List of paths to trained models
    output_base_dir : str, optional
        Base directory for outputs
    visualize : bool
        Whether to generate visualizations

    Returns:
    --------
    dict
        Dictionary with results for each model
    """
    if output_base_dir is None:
        output_base_dir = f"predictions/batch_{datetime.now().strftime('%Y%m%d')}"

    os.makedirs(output_base_dir, exist_ok=True)

    results = {}
    for i, model_path in enumerate(model_paths):
        model_name = Path(model_path).stem
        logger.info(f"Processing model {i+1}/{len(model_paths)}: {model_name}")

        output_dir = Path(output_base_dir) / model_name
        result = await run_prediction_service(
            model_path=model_path, output_dir=output_dir, visualize=visualize
        )

        results[model_name] = result

    # Create a summary of all model results
    summary_path = Path(output_base_dir) / "batch_summary.json"
    with open(summary_path, "w") as f:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models_processed": len(model_paths),
            "successful_predictions": sum(
                1 for r in results.values() if r.get("success", False)
            ),
            "failed_predictions": sum(
                1 for r in results.values() if not r.get("success", False)
            ),
            "results": {
                model: {
                    "success": result.get("success", False),
                    "metrics": result.get("metrics", {}),
                    "error": result.get("error", None),
                }
                for model, result in results.items()
            },
        }
        json.dump(summary, f, indent=2)

    logger.info(f"Batch prediction complete. Summary saved to {summary_path}")
    return {"success": True, "results": results, "summary_path": str(summary_path)}


if __name__ == "__main__":
    # Process command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Run prediction service")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    parser.add_argument("output_dir", nargs="?", type=str, help="Output directory")
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        default=True,
        help="Generate visualizations",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Disable visualizations",
    )
    parser.add_argument(
        "--detailed", "-d", action="store_true", help="Generate detailed metrics"
    )
    parser.add_argument(
        "--cache", "-c", action="store_true", help="Cache prediction results"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for prediction",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Run in batch mode (model_path should be a directory)",
    )

    args = parser.parse_args()

    # Handle batch mode
    if args.batch:
        model_dir = Path(args.model_path)
        if not model_dir.is_dir():
            print(f"Error: {args.model_path} is not a directory")
            sys.exit(1)

        model_paths = list(model_dir.glob("*.pth"))
        if not model_paths:
            print(f"Error: No model files found in {args.model_path}")
            sys.exit(1)

        print(f"Found {len(model_paths)} models for batch processing")
        result = asyncio.run(
            batch_predict(
                model_paths=model_paths,
                output_base_dir=args.output_dir,
                visualize=args.visualize,
            )
        )

    else:
        # Run single prediction
        result = asyncio.run(
            run_prediction_service(
                model_path=args.model_path,
                output_dir=args.output_dir,
                visualize=args.visualize,
                cache_results=args.cache,
                detailed_metrics=args.detailed,
                device=args.device,
            )
        )

    # Exit with appropriate code
    sys.exit(0 if result.get("success", False) else 1)
