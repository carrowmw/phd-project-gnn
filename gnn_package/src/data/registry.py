# src/data/registry.py
from typing import Dict, Type, Any
from .data_sources import DataSource


class DataSourceRegistry:
    """Registry for data sources"""

    _sources: Dict[str, Type[DataSource]] = {}

    @classmethod
    def register(cls, name: str, source_class: Type[DataSource]) -> None:
        """
        Register a data source class with the registry.

        Parameters:
        -----------
        name : str
            Name of the data source
        source_class : Type[DataSource]
            Data source class to register
        """
        cls._sources[name] = source_class

    @classmethod
    def get_source(cls, name: str, **kwargs: Any) -> DataSource:
        """
        Get a data source instance by name.

        Parameters:
        -----------
        name : str
            Name of the data source
        **kwargs : Any
            Arguments to pass to the data source constructor

        Returns:
        --------
        DataSource
            Instance of the requested data source

        Raises:
        -------
        ValueError
            If the requested data source is not registered
        """
        if name not in cls._sources:
            raise ValueError(f"Unknown data source: {name}")
        return cls._sources[name](**kwargs)

    @classmethod
    def list_sources(cls) -> Dict[str, Type[DataSource]]:
        """
        List all registered data sources.

        Returns:
        --------
        Dict[str, Type[DataSource]]
            Dictionary mapping source names to their classes
        """
        return cls._sources.copy()
