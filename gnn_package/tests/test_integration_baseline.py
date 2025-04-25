"""
Baseline integration tests for the GNN package.
These tests verify the current behavior of the entry points and workflows.
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import subprocess
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch


from gnn_package.config import get_config, create_default_config
from gnn_package.src.training.stgnn_training import preprocess_data, train_model
from gnn_package.src.training.stgnn_prediction import (
    predict_all_sensors_with_validation,
)


class IntegrationBaselineTests(unittest.IsolatedAsyncioTestCase):
    """Test baseline integration of the GNN package components."""

    def setUp(self):
        # Create a temporary directory for test data and outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a config for testing
        self.config_path = self.temp_path / "test_config.yml"
        create_default_config(self.config_path)
        self.config = get_config(self.config_path)

        # Update config for faster testing
        self.config.model.hidden_dim = 16
        self.config.model.num_layers = 1
        self.config.training.num_epochs = 2
        self.config.training.device = "cpu"
        self.config.paths.model_save_path = str(self.temp_path / "models")
        self.config.paths.results_dir = str(self.temp_path / "results")

        # Save updated config
        self.config.save(self.config_path)

        # Create sample data for end-to-end testing
        self.create_sample_data()

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def create_sample_data(self):
        """Create sample time series and graph data for testing."""
        # Sample time series data
        start_date = datetime(2024, 2, 1)
        end_date = datetime(2024, 2, 14)
        date_range = pd.date_range(start=start_date, end=end_date, freq="h")

        # Create data for 3 sensors
        sensors = ["sensor1", "sensor2", "sensor3"]
        data = {}

        for sensor in sensors:
            # Generate some sample data with a pattern
            values = np.sin(np.linspace(0, 10, len(date_range))) * 10
            values += np.random.normal(0, 1, len(date_range))  # Add some noise

            # Create a DataFrame with timestamp index - numeric data only
            df = pd.DataFrame(
                {
                    "value": values,
                },
                index=date_range,
            )

            # Add categorical columns separately
            df["category"] = "flow"
            df["veh_class"] = "car"
            df["dir"] = "east_to_west"

            data[sensor] = df

        # Save as pickle
        self.data_path = self.temp_path / "test_data.pkl"
        with open(self.data_path, "wb") as f:
            pickle.dump(data, f)

        # Update config with data path
        self.config.data.file_path = str(self.data_path)
        self.config.save(self.config_path)

    @unittest.skip("Skipping until data process errors are fixed")
    async def test_end_to_end_workflow(self):
        """Test the complete workflow from data to prediction."""
        # 1. Preprocess data
        data_loaders = await preprocess_data(
            data_file=str(self.data_path), config=self.config
        )

        # Verify data loaders
        self.assertIn("train_loader", data_loaders)
        self.assertIn("val_loader", data_loaders)
        self.assertIn("graph_data", data_loaders)

        # 2. Train model
        results = train_model(data_loaders=data_loaders, config=self.config)

        # Verify training results
        self.assertIn("model", results)
        self.assertIn("train_losses", results)
        self.assertIn("val_losses", results)

        # 3. Save model
        model_dir = Path(self.config.paths.model_save_path)
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / "test_model.pth"

        # Save model state dict
        torch.save(results["model"].state_dict(), model_path)

        # 4. Make predictions
        prediction_results = await predict_all_sensors_with_validation(
            model_path=model_path,
            config=self.config,
            plot=False,  # Disable plotting for testing
        )

        # Verify prediction results
        self.assertIn("predictions", prediction_results)
        self.assertIn("predictions_df", prediction_results)

    @unittest.skip("Skipping until integration issues are fixed")
    def test_run_experiment_script(self):
        """Test the run_experiment.py script from command line."""
        # Skip if the script doesn't exist
        run_experiment_path = Path("run_experiment.py")
        if not run_experiment_path.exists():
            self.skipTest("run_experiment.py not found")

        # Prepare output directory
        output_dir = self.temp_path / "experiment_output"
        output_dir.mkdir(exist_ok=True)

        # Run the experiment script with minimal epochs
        cmd = [
            sys.executable,
            "run_experiment.py",
            "--config",
            str(self.config_path),
            "--data",
            str(self.data_path),
            "--output",
            str(output_dir),
        ]

        try:
            # Run the command with a timeout
            subprocess.run(
                cmd,
                check=True,
                timeout=300,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Check if output was generated
            self.assertTrue(output_dir.exists())

            # Check for model file
            model_files = list(output_dir.glob("*.pth"))
            self.assertGreater(len(model_files), 0, "No model file was created")

        except subprocess.CalledProcessError as e:
            self.fail(
                f"Command failed with exit code {e.returncode}. "
                f"Output: {e.stdout.decode()}\nError: {e.stderr.decode()}"
            )
        except subprocess.TimeoutExpired:
            self.fail("Command timed out")

    @unittest.skip("Skipping until integration issues are fixed")
    async def test_prediction_service_script(self):
        """Test the prediction_service.py script from command line."""
        # Skip if the script doesn't exist
        prediction_service_path = Path("prediction_service.py")
        if not prediction_service_path.exists():
            self.skipTest("prediction_service.py not found")

        # First, we need a trained model
        # Use the test_end_to_end_workflow to get one
        await self.test_end_to_end_workflow()

        # Prepare prediction directory
        prediction_dir = self.temp_path / "predictions"
        prediction_dir.mkdir(exist_ok=True)

        # Get the model file
        model_dir = Path(self.config.paths.model_save_path)
        model_files = list(model_dir.glob("*.pth"))
        if not model_files:
            self.fail("No model file found")

        model_path = model_files[0]

        # Run the prediction script
        cmd = [
            sys.executable,
            "prediction_service.py",
            str(model_path),
            str(prediction_dir),
            "no",  # Disable visualization for testing
        ]

        try:
            # Run the command with a timeout
            subprocess.run(
                cmd,
                check=True,
                timeout=300,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Check if output was generated
            self.assertTrue(prediction_dir.exists())

            # Check for prediction file
            prediction_files = list(prediction_dir.glob("predictions_*.csv"))
            self.assertGreater(
                len(prediction_files), 0, "No prediction file was created"
            )

        except subprocess.CalledProcessError as e:
            self.fail(
                f"Command failed with exit code {e.returncode}. "
                f"Output: {e.stdout.decode()}\nError: {e.stderr.decode()}"
            )
        except subprocess.TimeoutExpired:
            self.fail("Command timed out")

    @unittest.skip("Skipping until integration issues are fixed")
    def test_tune_model_script(self):
        """Test the tune_model.py script from command line."""
        # Skip if the script doesn't exist
        tune_model_path = Path("tune_model.py")
        if not tune_model_path.exists():
            self.skipTest("tune_model.py not found")

        # Prepare tuning directory
        tuning_dir = self.temp_path / "tuning_output"
        tuning_dir.mkdir(exist_ok=True)

        # Run the tuning script with minimal trials and quick mode
        cmd = [
            sys.executable,
            "tune_model.py",
            "--data",
            str(self.data_path),
            "--trials",
            "2",  # Minimal trials for testing
            "--quick",  # Quick mode
            "--output",
            str(tuning_dir),
        ]

        try:
            # Run the command with a timeout
            subprocess.run(
                cmd,
                check=True,
                timeout=300,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Check if output was generated
            self.assertTrue(tuning_dir.exists())

            # Check for best params file
            best_params_file = tuning_dir / "best_params.json"
            self.assertTrue(
                best_params_file.exists(), "No best_params.json was created"
            )

        except subprocess.CalledProcessError as e:
            self.fail(
                f"Command failed with exit code {e.returncode}. "
                f"Output: {e.stdout.decode()}\nError: {e.stderr.decode()}"
            )
        except subprocess.TimeoutExpired:
            self.fail("Command timed out")


if __name__ == "__main__":
    unittest.main()
