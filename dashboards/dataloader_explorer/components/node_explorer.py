# dashboards/dataloader_explorer/components/node_explorer.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from ..utils.data_utils import get_batch_from_loader, get_node_data


def create_node_explorer(
    data_loaders, loader_key="train_loader", batch_idx=0, node_idx=0
):
    """
    Create an interactive node explorer visualization

    Parameters:
    -----------
    data_loaders : dict
        Dictionary containing data loaders
    loader_key : str
        Key to access the loader from data_loaders
    batch_idx : int
        Index of the batch to visualize
    node_idx : int
        Index of the node to visualize

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive node explorer figure
    """
    # Get batch data
    try:
        data_loader = data_loaders[loader_key]
        batch = get_batch_from_loader(data_loader, batch_idx)
        node_data = get_node_data(batch, batch_idx, node_idx)
    except Exception as e:
        # Create an error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading node data: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig

    # Extract dimensions and node ID
    node_id = node_data["node_id"]
    seq_len = node_data["seq_len"]
    horizon = node_data["horizon"]

    # Create a figure with 2 subplots (input and target data)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Input Data (Historical)", "Target Data (Future)"],
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.08,
    )

    # Create time step arrays
    input_steps = np.arange(seq_len)
    target_steps = np.arange(seq_len, seq_len + horizon)

    # Plot input data
    x_data = node_data["input_data"]
    x_mask = node_data["input_mask"]

    # Create separate traces for valid and missing data to control styling
    valid_mask = x_mask > 0

    # Plot valid data points
    if np.any(valid_mask):
        fig.add_trace(
            go.Scatter(
                x=input_steps[valid_mask],
                y=x_data[valid_mask],
                mode="lines+markers",
                line=dict(color="royalblue", width=2),
                marker=dict(
                    size=8, color="royalblue", line=dict(width=1, color="darkblue")
                ),
                name="Valid Input Data",
                hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra>Valid Input</extra>",
            ),
            row=1,
            col=1,
        )

    # Plot missing data points
    missing_mask = ~valid_mask
    if np.any(missing_mask):
        fig.add_trace(
            go.Scatter(
                x=input_steps[missing_mask],
                y=x_data[missing_mask],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=10,
                    color="red",
                    line=dict(width=1, color="darkred"),
                ),
                name="Missing Input Data",
                hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra>Missing Input</extra>",
            ),
            row=1,
            col=1,
        )

        # Add a semi-transparent rectangle for each missing value
        for step in input_steps[missing_mask]:
            fig.add_shape(
                type="rect",
                xref=f"x",
                yref=f"y",
                x0=step - 0.4,
                y0=-999999,
                x1=step + 0.4,
                y1=999999,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line=dict(width=0),
                layer="below",
                row=1,
                col=1,
            )

    # Plot target data
    y_data = node_data["target_data"]
    y_mask = node_data["target_mask"]

    # Create separate traces for valid and missing target data
    valid_mask = y_mask > 0

    # Plot valid target data points
    if np.any(valid_mask):
        fig.add_trace(
            go.Scatter(
                x=target_steps[valid_mask],
                y=y_data[valid_mask],
                mode="lines+markers",
                line=dict(color="green", width=2, dash="dash"),
                marker=dict(
                    size=8, color="green", line=dict(width=1, color="darkgreen")
                ),
                name="Valid Target Data",
                hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra>Valid Target</extra>",
            ),
            row=1,
            col=2,
        )

    # Plot missing target data points
    missing_mask = ~valid_mask
    if np.any(missing_mask):
        fig.add_trace(
            go.Scatter(
                x=target_steps[missing_mask],
                y=y_data[missing_mask],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=10,
                    color="orange",
                    line=dict(width=1, color="darkorange"),
                ),
                name="Missing Target Data",
                hovertemplate="Step: %{x}<br>Value: %{y:.4f}<extra>Missing Target</extra>",
            ),
            row=1,
            col=2,
        )

        # Add a semi-transparent rectangle for each missing value
        for step in target_steps[missing_mask]:
            fig.add_shape(
                type="rect",
                xref=f"x2",
                yref=f"y2",
                x0=step - 0.4,
                y0=-999999,
                x1=step + 0.4,
                y1=999999,
                fillcolor="rgba(255, 165, 0, 0.1)",
                line=dict(width=0),
                layer="below",
                row=1,
                col=2,
            )

    # Add a vertical line at the boundary between input and prediction
    fig.add_vline(
        x=seq_len - 0.5, line=dict(color="black", width=1, dash="dash"), row=1, col=1
    )

    # Add a vertical line at the boundary between input and prediction in the prediction panel
    fig.add_vline(
        x=seq_len - 0.5, line=dict(color="black", width=1, dash="dash"), row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Node Explorer - Node {node_id} (Batch {batch_idx}, {loader_key})",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=100, b=100, l=50, r=50),
    )

    # Update x and y axes
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    # Ensure the same y-axis range for both panels for better comparison
    all_values = np.concatenate([x_data, y_data])
    if len(all_values) > 0:
        y_min = np.nanmin(all_values) - 0.5
        y_max = np.nanmax(all_values) + 0.5
        fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
        fig.update_yaxes(range=[y_min, y_max], row=1, col=2)

    # Add a summary of node data in an annotation
    summary_text = (
        f"Node ID: {node_id}<br>"
        f"Input Length: {seq_len} steps<br>"
        f"Target Horizon: {horizon} steps<br>"
        f"Valid Input Points: {np.sum(node_data['input_mask'])} / {seq_len} ({np.mean(node_data['input_mask'])*100:.1f}%)<br>"
        f"Valid Target Points: {np.sum(node_data['target_mask'])} / {horizon} ({np.mean(node_data['target_mask'])*100:.1f}%)"
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
