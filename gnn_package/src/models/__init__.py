# In src/models/__init__.py

from .architectures import STGNN, ImprovedSTGNN, GATWithGRU, FullAttentionSTGNN
from .layers import GraphConvolution, GraphAttentionLayer, MultiHeadAttention
from .factory import create_model
from .registry import ModelRegistry

# Register all model architectures
ModelRegistry.register_model("stgnn", STGNN)
ModelRegistry.register_model("improved_stgnn", ImprovedSTGNN)
ModelRegistry.register_model("gat_stgnn", GATWithGRU)
ModelRegistry.register_model("full_attention", FullAttentionSTGNN)

# Register creator function
ModelRegistry.register_creator("create_model", create_model)

__all__ = [
    # Model architectures
    "STGNN",
    "ImprovedSTGNN",
    "GATWithGRU",
    "FullAttentionSTGNN",

    # Layer components
    "GraphConvolution",
    "GraphAttentionLayer",
    "MultiHeadAttention",

    # Factory function
    "create_model",
]