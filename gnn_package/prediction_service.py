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

from gnn_package.config import ExperimentConfig, create_prediction_config
from gnn_package.src.training.prediction import predict_and_evaluate
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
    config_path=None,
):
    """
    Run the prediction service using the new prediction framework.
    """
    try:
        start_time = datetime.now()

        # Create prediction configuration
        logger.info("Setting up prediction configuration")
        try:
            if config_path is not None:
                config_path = Path(config_path)
                config = ExperimentConfig(str(config_path), is_prediction_mode=True)
                logger.info(f"Using configuration from specified path: {config_path}")
            else:
                # Try to find config in model directory
                model_dir = Path(model_path).parent
                config_path = model_dir / "config.yml"
                if config_path.exists():
                    config = ExperimentConfig(str(config_path), is_prediction_mode=True)
                    logger.info(f"Using configuration from model directory: {config_path}")
                else:
                    raise FileNotFoundError(
                        f"No configuration file found. "
                        f"Please provide a configuration file using the --config parameter."
                    )
        except FileNotFoundError as e:
            logger.error(f"Configuration error: {str(e)}")
            return {"success": False, "error": str(e)}

        # Set up output directory
        if output_dir is None:
            today = datetime.now().strftime("%Y-%m-%d")
            output_dir = f"predictions/{today}"

        # Use the new predict_and_evaluate function
        prediction_results = await predict_and_evaluate(
            model_path=model_path,
            output_dir=output_dir,
            config=config,
            visualize=visualize
        )

        # Format response
        execution_time = (datetime.now() - start_time).total_seconds()

        # Get file paths from the results
        predictions_file = None
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.startswith("predictions_") and file.endswith(".csv"):
                    predictions_file = os.path.join(root, file)
                    break

        summary_file = None
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.startswith("summary_") and file.endswith(".txt"):
                    summary_file = os.path.join(root, file)
                    break

        return {
            "success": True,
            "predictions_file": predictions_file,
            "summary_file": summary_file,
            "metrics": prediction_results.get("metrics", {}),
            "visualizations": prediction_results.get("visualization_paths", {}),
            "execution_time": execution_time,
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


async def batch_predict(model_paths, output_base_dir=None, visualize=True, config_path=None):
    """
    Run predictions on multiple models in batch mode.
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
            model_path=model_path,
            output_dir=output_dir,
            visualize=visualize,
            config_path=config_path
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
        "--config",
        type=str,
        help="Path to configuration file to use",
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
                config_path=args.config,
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
                config_path=args.config,
            )
        )

    # Exit with appropriate code
    sys.exit(0 if result.get("success", False) else 1)