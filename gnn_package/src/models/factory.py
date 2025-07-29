# src/models/factory.py

# src/models/factory.py
import torch.nn as nn
from gnn_package.config import ExperimentConfig
from .architectures import STGNN, ImprovedSTGNN, GATWithGRU, FullAttentionSTGNN

def create_model(config: ExperimentConfig) -> nn.Module:
    """
    Create a model instance based on configuration.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration object containing model parameters

    Returns:
    --------
    torch.nn.Module
        The created model instance
    """
    architecture = getattr(config.model, "architecture", "stgnn").lower()

    if architecture == "stgnn":
        return STGNN(config)
    elif architecture in ["improved_stgnn", "improved"]:
        return ImprovedSTGNN(config)
    elif architecture in ["gat_stgnn", "gat", "hybrid"]:
        return GATWithGRU(config)
    elif architecture in ["full_attention", "attention"]:
        return FullAttentionSTGNN(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")