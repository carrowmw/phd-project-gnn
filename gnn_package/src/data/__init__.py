# src/data/__init__.py
from .data_sources import (
    DataSource,
    FileDataSource,
    APIDataSource,
    DataSourceException,
    DataSourceConnectionError,
    DataSourceFormatError,
)
from .registry import DataSourceRegistry
from .factory import (
    create_data_source,
    create_file_data_source,
    create_api_data_source,
    get_data_source_from_config,
)

# Register data sources with the registry
DataSourceRegistry.register(FileDataSource.source_type, FileDataSource)
DataSourceRegistry.register(APIDataSource.source_type, APIDataSource)

__all__ = [
    # Base classes and exceptions
    "DataSource",
    "DataSourceException",
    "DataSourceConnectionError",
    "DataSourceFormatError",
    # Concrete implementations
    "FileDataSource",
    "APIDataSource",
    # Registry
    "DataSourceRegistry",
    # Factory functions
    "create_data_source",
    "create_file_data_source",
    "create_api_data_source",
    "get_data_source_from_config",
]
