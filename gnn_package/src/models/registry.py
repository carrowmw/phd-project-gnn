# src/models/registry.py - Updated version
from typing import Dict, Type, Any, Callable
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for model architectures with improved error handling and logging"""

    _models: Dict[str, Type[nn.Module]] = {}
    _creators: Dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module]) -> None:
        """
        Register a model class with the registry.

        Parameters:
        -----------
        name : str
            Name of the model architecture
        model_class : Type[nn.Module]
            Model class to register
        """
        if name in cls._models:
            logger.warning(f"Overriding existing model registration for '{name}'")

        cls._models[name] = model_class
        logger.debug(f"Registered model class '{name}'")

    @classmethod
    def register_creator(
        cls, name: str, creator_func: Callable[..., nn.Module]
    ) -> None:
        """
        Register a model creator function with the registry.

        Parameters:
        -----------
        name : str
            Name of the model architecture
        creator_func : Callable[..., nn.Module]
            Function that creates and returns a model instance
        """
        if name in cls._creators:
            logger.warning(f"Overriding existing creator function for '{name}'")

        cls._creators[name] = creator_func
        logger.debug(f"Registered model creator function for '{name}'")

    @classmethod
    def create_model(cls, name: str, **kwargs: Any) -> nn.Module:
        """
        Create a model instance by name with improved error handling.

        Parameters:
        -----------
        name : str
            Name of the model architecture
        **kwargs : Any
            Arguments to pass to the model constructor or creator function

        Returns:
        --------
        nn.Module
            Model instance

        Raises:
        -------
        ValueError
            If the requested model is not registered
        RuntimeError
            If model creation fails
        """
        try:
            if name in cls._creators:
                logger.debug(f"Creating model '{name}' using creator function")
                return cls._creators[name](**kwargs)
            elif name in cls._models:
                logger.debug(f"Creating model '{name}' using model class")
                return cls._models[name](**kwargs)
            else:
                available_models = list(
                    set(cls._models.keys()) | set(cls._creators.keys())
                )
                raise ValueError(
                    f"Unknown model architecture: {name}. Available models: {available_models}"
                )
        except Exception as e:
            if isinstance(e, ValueError) and "Unknown model architecture" in str(e):
                # Re-raise ValueError for unknown model
                raise
            # Wrap other exceptions
            logger.error(f"Error creating model '{name}': {str(e)}")
            raise RuntimeError(f"Failed to create model '{name}'") from e

    @classmethod
    def list_models(cls) -> Dict[str, Type[nn.Module]]:
        """
        List all registered model classes.

        Returns:
        --------
        Dict[str, Type[nn.Module]]
            Dictionary mapping model names to their classes
        """
        return cls._models.copy()

    @classmethod
    def list_creators(cls) -> Dict[str, Callable[..., nn.Module]]:
        """
        List all registered model creator functions.

        Returns:
        --------
        Dict[str, Callable[..., nn.Module]]
            Dictionary mapping creator names to their functions
        """
        return cls._creators.copy()

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get a list of all available model types.

        Returns:
        --------
        list
            List of available model names
        """
        return list(set(cls._models.keys()) | set(cls._creators.keys()))
