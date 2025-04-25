# src/data/factory.py (NEW FILE)
from typing import Dict, Any, Optional, Union
from pathlib import Path

from gnn_package.config import ExperimentConfig
from .data_sources import DataSource, FileDataSource, APIDataSource
from .registry import DataSourceRegistry


def create_data_source(
    source_type: str = "file", config: Optional[ExperimentConfig] = None, **kwargs: Any
) -> DataSource:
    """
    Create a data source instance based on the source type.

    Parameters:
    -----------
    source_type : str
        Type of the data source ("file" or "api")
    config : ExperimentConfig, optional
        Configuration object
    **kwargs : Any
        Additional arguments to pass to the data source constructor

    Returns:
    --------
    DataSource
        Instance of the requested data source

    Raises:
    -------
    ValueError
        If the requested data source type is not registered
    """
    return DataSourceRegistry.get_source(source_type, **kwargs)


def create_file_data_source(file_path: Union[str, Path]) -> FileDataSource:
    """
    Create a file data source.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file containing the data

    Returns:
    --------
    FileDataSource
        File data source instance
    """
    return FileDataSource(file_path)


def create_api_data_source() -> APIDataSource:
    """
    Create an API data source.

    Returns:
    --------
    APIDataSource
        API data source instance
    """
    return APIDataSource()


def get_data_source_from_config(config: ExperimentConfig) -> DataSource:
    """
    Create a data source based on the configuration.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration object

    Returns:
    --------
    DataSource
        Data source instance based on the configuration
    """
    # Check if a specific data source is specified in the config
    if hasattr(config.data, "source") and hasattr(config.data.source, "type"):
        source_type = config.data.source.type
    else:
        # Default to file data source
        source_type = "file"

    # Get additional parameters from config if available
    kwargs = {}

    if source_type == "file":
        # Get file path from config
        if hasattr(config.data, "file_path"):
            kwargs["file_path"] = config.data.file_path
        elif hasattr(config.paths, "data_file_path"):
            kwargs["file_path"] = config.paths.data_file_path
        else:
            raise ValueError("File path not specified in configuration")

    # Create and return the data source
    return create_data_source(source_type, config, **kwargs)
