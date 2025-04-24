"""
Baseline tests for configuration module functionality.
These tests verify the current behavior of the configuration system.
"""

import os
import tempfile
import unittest
from pathlib import Path
import yaml

# Adjust these imports to match your actual module structure
from gnn_package.config import get_config, reset_config, create_default_config
from gnn_package.config.config import ExperimentConfig


class ConfigBaselineTests(unittest.TestCase):
    """Test baseline functionality of the configuration system."""

    def setUp(self):
        # Reset global config before each test
        reset_config()
        # Create a temporary directory for test configs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_create_default_config(self):
        """Test creation of default configuration file."""
        config_path = self.temp_path / "test_config.yml"
        create_default_config(config_path)

        # Verify the config file was created
        self.assertTrue(config_path.exists())

        # Verify it contains expected sections
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        expected_sections = [
            "experiment",
            "data",
            "model",
            "training",
            "paths",
            "visualization",
        ]
        for section in expected_sections:
            self.assertIn(section, config_data)

    def test_load_config(self):
        """Test loading configuration from file."""
        # Create a config file
        config_path = self.temp_path / "load_test.yml"
        create_default_config(config_path)

        # Load the config
        config = ExperimentConfig(config_path)

        # Verify basic properties
        self.assertIsNotNone(config.experiment)
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.training)

    def test_global_config_singleton(self):
        """Test the global config singleton pattern."""
        # Create a config file
        config_path = self.temp_path / "singleton_test.yml"
        create_default_config(config_path)

        # Set the global config
        config1 = get_config(config_path)

        # Get it again - should be the same instance
        config2 = get_config()

        # Verify it's the same instance
        self.assertIs(config1, config2)

    def test_config_validation(self):
        """Test configuration validation."""
        # Create a valid config
        config_path = self.temp_path / "validation_test.yml"
        create_default_config(config_path)

        # Load and validate - should not raise exceptions
        config = ExperimentConfig(config_path)

        # Now modify the file to have minimal but valid structure
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Keep experiment metadata but remove model.hidden_dim
        if "model" in config_data and "hidden_dim" in config_data["model"]:
            del config_data["model"]["hidden_dim"]

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Load and validate - should fail on hidden_dim
        try:
            config = ExperimentConfig(config_path)
            config.validate()
            self.fail("Expected validation to fail with missing hidden_dim")
        except Exception as e:
            # Success - validation failed as expected
            self.assertIn("hidden_dim", str(e).lower())

    def test_config_save(self):
        """Test saving configuration to file."""
        # Create a config
        config_path = self.temp_path / "original.yml"
        create_default_config(config_path)
        config = ExperimentConfig(config_path)

        # Modify a value
        config.model.hidden_dim = 128

        # Save to a new file
        new_path = self.temp_path / "modified.yml"
        config.save(new_path)

        # Load the new file and verify the change
        new_config = ExperimentConfig(new_path)
        self.assertEqual(new_config.model.hidden_dim, 128)


if __name__ == "__main__":
    unittest.main()
