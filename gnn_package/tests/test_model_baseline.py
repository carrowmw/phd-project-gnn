"""
Baseline tests for model functionality.
These tests verify the current behavior of the model creation, training, and prediction.
"""

import unittest
from unittest.mock import patch
import os
import pickle
import tempfile
from pathlib import Path
import torch
import numpy as np

# Adjust these imports to match your actual module structure
from gnn_package.config import get_config, create_default_config
from gnn_package.src.models.stgnn import create_stgnn_model
from gnn_package.src.training.stgnn_training import STGNNTrainer
from gnn_package.src.training.stgnn_prediction import predict_with_model
from gnn_package.src.utils.model_io import load_model


class ModelBaselineTests(unittest.TestCase):
    """Test baseline functionality of the model system."""

    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a config for testing
        self.config_path = self.temp_path / "test_config.yml"
        create_default_config(self.config_path)
        self.config = get_config(self.config_path)

        # Update config for faster testing
        self.config.model.hidden_dim = 16
        self.config.model.num_layers = 1
        self.config.training.device = "cpu"

        # Create sample data for model testing
        self.create_sample_data()

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def create_sample_data(self):
        """Create sample data structures for model testing."""
        batch_size = 4
        num_nodes = 3
        window_size = self.config.data.general.window_size
        in_channels = 1  # Simplified for testing

        # Create sample batch
        x = torch.randn(batch_size, num_nodes, window_size, in_channels)
        masks = torch.ones_like(x)  # All data points are valid
        y = torch.randn(
            batch_size, num_nodes, self.config.data.general.horizon, in_channels
        )

        # Create sample adjacency matrix
        adj_matrix = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)

        self.sample_batch = (x, masks, y, adj_matrix)

    def test_model_creation(self):
        """Test creating a model with the configuration."""
        model = create_stgnn_model(self.config)

        # Verify model structure
        self.assertIsNotNone(model)

        # Get some basic properties to verify
        num_parameters = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_parameters, 0)

    def test_model_forward_pass(self):
        """Test a forward pass through the model."""
        model = create_stgnn_model(self.config)

        # Unpack the sample batch
        x, masks, y, adj_matrix = self.sample_batch

        # Run a forward pass
        with torch.no_grad():
            output = model(x, adj_matrix, masks)

        # Verify output shape
        expected_shape = (
            x.shape[0],  # batch_size
            x.shape[1],  # num_nodes
            self.config.data.general.horizon,
            x.shape[3],  # in_channels
        )
        self.assertEqual(output.shape, expected_shape)

    def test_model_train_epoch(self):
        """Test a single training step with the model."""
        model = create_stgnn_model(self.config)
        trainer = STGNNTrainer(model, self.config)

        # Unpack the sample batch
        x, masks, y, adj_matrix = self.sample_batch

        # Create a batch in the format expected by the trainer
        batch = {
            "x": x,
            "x_mask": masks,
            "y": y,
            "y_mask": torch.ones_like(y),  # Add mask for y if needed
            "adj": adj_matrix,
            "node_indices": torch.tensor([0, 1, 2]),
        }

        # Create a dataloader that yields this dictionary
        class MockDataLoader:
            def __iter__(self):
                yield batch

        dataloader = MockDataLoader()

        # Call train_epoch with the properly formatted dataloader
        loss = trainer.train_epoch(dataloader)

        # Verify loss
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)  # Loss should be positive

    def test_model_save_load(self):
        """Test saving and loading a model."""
        # Create and initialize a model
        model = create_stgnn_model(self.config)

        # Save the model
        model_path = self.temp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)

        # Save the config alongside
        config_path = self.temp_path / "test_model_config.yml"
        self.config.save(config_path)

        # Load the model back
        loaded_model, metadata = load_model(
            model_path=model_path,
            model_type="stgnn",
            config=self.config,
        )

        # Verify model was loaded correctly by checking parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Create and initialize a model
        model = create_stgnn_model(self.config)

        # Mock a dataloader with a single batch
        x, masks, y, adj_matrix = self.sample_batch

        # Examine the actual method signature
        import inspect

        print(f"Signature: {inspect.signature(predict_with_model)}")

        # Create a batch dictionary instead of tuple
        batch = {
            "x": x,
            "x_mask": masks,
            "y": y,
            "adj": adj_matrix,
            "node_indices": torch.tensor([0, 1, 2]),  # Add node indices if needed
        }

        class MockDataLoader:
            def __iter__(self):
                yield batch

        # Create a data package dictionary matching expected format
        data_package = {
            "data_loaders": {"val_loader": MockDataLoader()},
            "graph_data": {
                "adj_matrix": adj_matrix,
                "node_ids": ["sensor1", "sensor2", "sensor3"],
            },
            "time_series": {"validation": {}},  # Add appropriate data
            "metadata": {"mode": "prediction"},
        }

        print(f"Data package type: {type(data_package)}")

        # Mock the validate_data_package function to avoid validation errors during testing
        with patch("gnn_package.src.utils.data_utils.validate_data_package"):
            predictions = predict_with_model(model, data_package, self.config)

        # Verify prediction structure
        self.assertIn("predictions", predictions)


if __name__ == "__main__":
    unittest.main()
