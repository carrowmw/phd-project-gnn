#!/usr/bin/env python
# prediction_service.py

import os
import sys
import asyncio
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from gnn_package import training
from gnn_package.config import ExperimentConfig
from gnn_package.src.utils.config_utils import create_prediction_config
from gnn_package.src.visualization.prediction_plots import (
    plot_sensors_grid,
    plot_error_distribution,
    save_visualization_pack,
)

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


async def run_prediction_service(model_path, output_dir=None, visualize=True):
    """
    Run the prediction service.

    Parameters:
    -----------
    model_path : str
        Path to the trained model
    output_dir : str, optional
        Directory to save predictions (defaults to 'predictions/YYYY-MM-DD')
    visualize : bool
        Whether to generate plots and visualizations
    """
    try:
        # Create prediction configuration
        logger.info("Creating prediction configuration")

        # Try to load config from model directory first
        model_dir = Path(model_path).parent
        config_path = model_dir / "config.yml"

        if config_path.exists():
            logger.info(f"Loading configuration from model directory: {config_path}")
            # Load as prediction config
            prediction_config = ExperimentConfig(
                str(config_path), is_prediction_mode=True
            )
        else:
            logger.info("Creating default prediction configuration")
            prediction_config = create_prediction_config()

        # Make sure days_back is set to a reasonable value for prediction
        if (
            not hasattr(prediction_config.data.prediction, "days_back")
            or prediction_config.data.prediction.days_back < 1
        ):
            logger.info("Setting days_back to default (7)")
            prediction_config.data.prediction.days_back = 7

        # Set up output directory
        if output_dir is None:
            today = datetime.now().strftime("%Y-%m-%d")
            output_dir = f"predictions/{today}"

        # Create timestamps for consistent file naming
        timestamp = datetime.now().strftime("%H%M%S")

        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/predictions_{timestamp}.csv"

        # Run prediction
        logger.info(f"Running prediction using model: {model_path}")
        logger.info(f"Output will be saved to: {output_file}")

        start_time = datetime.now()
        predictions = await training.predict_all_sensors_with_validation(
            model_path=model_path,
            config=prediction_config,
            output_file=output_file,
            plot=False,  # Disable internal plotting
        )
        end_time = datetime.now()

        # Log results
        if predictions and "dataframe" in predictions:
            df = predictions["dataframe"]
            logger.info(
                f"Generated {len(df)} predictions for {df['node_id'].nunique()} sensors"
            )

            # Calculate error metrics
            mse = (df["error"] ** 2).mean()
            mae = df["abs_error"].mean()
            logger.info(f"Overall MSE: {mse:.4f}, MAE: {mae:.4f}")

            # Save summary statistics
            summary_file = f"{output_dir}/summary_{timestamp}.txt"

            with open(summary_file, "w") as f:
                f.write(f"Prediction Summary\n")
                f.write(f"=================\n\n")
                f.write(f"Date/Time: {datetime.now()}\n")
                f.write(f"Model: {model_path}\n")
                f.write(
                    f"Execution time: {(end_time - start_time).total_seconds():.2f} seconds\n\n"
                )

                # Extract standardization stats if available
                standardization_stats = {}
                if "data" in predictions and "metadata" in predictions["data"]:
                    metadata = predictions["data"]["metadata"]
                    standardization_stats = metadata.get("preprocessing_stats", {}).get(
                        "standardization", {}
                    )

                f.write(
                    f"Standardization mean: {standardization_stats.get('mean', 'N/A')}\n"
                )
                f.write(
                    f"Standardization std: {standardization_stats.get('std', 'N/A')}\n\n"
                )
                f.write(f"Predictions: {len(df)}\n")
                f.write(f"Sensors: {df['node_id'].nunique()}\n")
                f.write(f"Overall MSE: {mse:.4f}\n")
                f.write(f"Overall MAE: {mae:.4f}\n\n")
                f.write("Prediction summary by horizon:\n")
                f.write(df.groupby("horizon")[["abs_error"]].mean().to_string())

            logger.info(f"Summary saved to {summary_file}")

            # Generate visualizations if requested
            if visualize:
                logger.info("Generating visualizations...")
                try:
                    # Save comprehensive visualization pack
                    viz_paths = save_visualization_pack(
                        predictions_df=df,
                        results_dict=predictions,
                        output_dir=output_dir,
                        timestamp=timestamp,
                    )

                    logger.info(f"Visualizations saved to: {output_dir}")
                    for viz_type, path in viz_paths.items():
                        logger.info(f"  - {viz_type}: {path}")

                except Exception as e:
                    logger.error(f"Error generating visualizations: {e}")

            return {
                "success": True,
                "predictions_file": output_file,
                "summary_file": summary_file,
                "metrics": {"mse": mse, "mae": mae},
                "visualizations": viz_paths if visualize else None,
            }
        else:
            logger.error("Prediction failed or returned no results")
            return {"success": False, "error": "No predictions generated"}

    except Exception as e:
        logger.exception(f"Error in prediction service: {str(e)}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "results/test_1wk/model.pth"

    # Get output directory if provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Get visualization flag if provided
    visualize = True
    if len(sys.argv) > 3:
        visualize = sys.argv[3].lower() in ("yes", "true", "t", "1")

    # Run the prediction service
    result = asyncio.run(run_prediction_service(model_path, output_dir, visualize))

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)
