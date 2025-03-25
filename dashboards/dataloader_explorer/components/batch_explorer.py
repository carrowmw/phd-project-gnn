# dashboards/dataloader_explorer/components/batch_explorer.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from ..utils.data_utils import get_batch_from_loader


def create_batch_explorer(data_loaders, loader_key="train_loader", batch_idx=0):
    """
    Create an interactive batch explorer visualization

    Parameters:
    -----------
    data_loaders : dict
        Dictionary containing data loaders
    loader_key : str
        Key to access the loader from data_loaders
    batch_idx : int
        Index of the batch to visualize

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive batch explorer figure
    """
    # Get batch data
    try:
        data_loader = data_loaders[loader_key]
        batch = get_batch_from_loader(data_loader, batch_idx)
    except Exception as e:
        # Create an error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading batch: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig

    # Extract dimensions
    x = batch["x"]
    batch_size, num_nodes, seq_len, features = x.shape

    # Create a figure with subplots to show batch overview
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Batch Data Availability by Node",
            "Batch Missing Data Overview",
        ],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.1,
    )

    # Extract data for visualization
    node_ids = batch["node_indices"].cpu().numpy()

    # Create data availability heatmap
    # For each node and time step, calculate percentage of valid data points across batch
    availability_matrix = np.zeros((num_nodes, seq_len))
    for n in range(num_nodes):
        for t in range(seq_len):
            mask_values = batch["x_mask"][:, n, t, 0].cpu().numpy()
            availability_matrix[n, t] = np.mean(mask_values) * 100

    # Create heatmap showing data availability
    fig.add_trace(
        go.Heatmap(
            z=availability_matrix,
            x=[f"t{i}" for i in range(seq_len)],
            y=[f"Node {id}" for id in node_ids],
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar=dict(title="Data Available (%)", x=0.46),
            hovertemplate="Node: %{y}<br>Time Step: %{x}<br>Data Available: %{z:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Create a summary matrix showing missing data by feature and node
    # First for input (x)
    x_missing = np.zeros((num_nodes, 2))
    for n in range(num_nodes):
        # Input data missing
        x_missing[n, 0] = 100 - (
            batch["x_mask"][:, n, :, :].sum().item()
            / batch["x_mask"][:, n, :, :].numel()
            * 100
        )
        # Target data missing
        x_missing[n, 1] = 100 - (
            batch["y_mask"][:, n, :, :].sum().item()
            / batch["y_mask"][:, n, :, :].numel()
            * 100
        )

    # Create heatmap showing missing data by feature
    fig.add_trace(
        go.Heatmap(
            z=x_missing,
            x=["Input", "Target"],
            y=[f"Node {id}" for id in node_ids],
            colorscale="Reds",
            zmin=0,
            zmax=100,
            colorbar=dict(title="Missing Data (%)", x=1.0),
            hovertemplate="Node: %{y}<br>Data Type: %{x}<br>Missing: %{z:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Update layout with more information
    fig.update_layout(
        title=f"Batch Explorer - {loader_key} (Batch {batch_idx})",
        height=600,
        margin=dict(t=100, b=50, l=50, r=50),
    )

    # Add summary information in an annotation
    summary_text = (
        f"Batch Size: {batch_size}<br>"
        f"Number of Nodes: {num_nodes}<br>"
        f"Sequence Length: {seq_len}<br>"
        f"Missing Input Data: {x_missing[:, 0].mean():.1f}%<br>"
        f"Missing Target Data: {x_missing[:, 1].mean():.1f}%"
    )

    fig.add_annotation(
        text=summary_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False,
        font=dict(size=12),
        align="center",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
    )

    return fig
