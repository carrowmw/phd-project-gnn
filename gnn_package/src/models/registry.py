# src/models/registry.py
from typing import Dict, Type, Any, Callable
import torch.nn as nn


class ModelRegistry:
    """Registry for model architectures"""

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
        cls._models[name] = model_class

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
        cls._creators[name] = creator_func

    @classmethod
    def create_model(cls, name: str, **kwargs: Any) -> nn.Module:
        """
        Create a model instance by name.

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
        """
        if name in cls._creators:
            return cls._creators[name](**kwargs)
        elif name in cls._models:
            return cls._models[name](**kwargs)
        else:
            raise ValueError(f"Unknown model architecture: {name}")

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
