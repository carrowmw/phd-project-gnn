# src/models/__init__.py (UPDATED)
from .stgnn import STGNN, STGNNTrainer, create_stgnn_model
from .registry import ModelRegistry

# Register model with the registry
ModelRegistry.register_model("stgnn", STGNN)
ModelRegistry.register_creator("stgnn", create_stgnn_model)

__all__ = ["STGNN", "STGNNTrainer", "create_stgnn_model", "ModelRegistry"]
