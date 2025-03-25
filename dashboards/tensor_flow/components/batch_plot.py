import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_batch_plot(batch_data):
    """Create a more interpretable visualization of the batch data structure"""
    if batch_data is None:
        return go.Figure().add_annotation(
            text="No batch data available", showarrow=False, font=dict(size=14)
        )

    # Extract batch components safely
    try:
        x = batch_data["x"].numpy()
        x_mask = batch_data["x_mask"].numpy()
        y = batch_data["y"].numpy()
        y_mask = batch_data["y_mask"].numpy()
        node_indices = batch_data["node_indices"].numpy()
        adj = batch_data["adj"].numpy()
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error extracting batch data: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Create a more informative subplot layout
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Input Data (Historical Values)",
            "Input Mask (Valid Data Points)",
            "Target Data (Future Values)",
            "Target Mask (Valid Predictions)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Define consistent colors for each batch/node combination
    # This helps the user track the same node across different plots
    colors = [
        "rgb(31, 119, 180)",  # Blue
        "rgb(255, 127, 14)",  # Orange
        "rgb(44, 160, 44)",  # Green
        "rgb(214, 39, 40)",  # Red
        "rgb(148, 103, 189)",  # Purple
        "rgb(140, 86, 75)",  # Brown
    ]

    # Plot x - Input data with improved styling
    batch_size, num_nodes, seq_len, _ = x.shape
    for b in range(min(batch_size, 2)):  # Show up to 2 batches
        for n in range(min(num_nodes, 4)):  # Show up to 4 nodes
            node_id = node_indices[n]
            color_idx = (b * num_nodes + n) % len(colors)

            # Create a more descriptive name
            name = f"Batch {b}, Node {node_id}"

            # Use non-broken lines to indicate continuous valid data and add markers
            mask_values = x_mask[b, n, :, 0]
            x_values = x[b, n, :, 0]

            # For missing values (mask=0), set y to None to create gaps
            y_vals = np.where(mask_values > 0, x_values, None)

            fig.add_trace(
                go.Scatter(
                    x=list(range(seq_len)),
                    y=y_vals,
                    mode="lines+markers",
                    name=name,
                    legendgroup=f"batch{b}_node{n}",
                    marker=dict(size=6),
                    line=dict(color=colors[color_idx], width=2),
                    hovertemplate="Step: %{x}<br>Value: %{y:.2f}<br>"
                    + name
                    + "<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Add markers for missing values
            missing_indices = np.where(mask_values == 0)[0]
            if len(missing_indices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=missing_indices,
                        y=[None] * len(missing_indices),
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=8,
                            color=colors[color_idx],
                            line=dict(width=1, color="black"),
                        ),
                        name=f"{name} (Missing)",
                        legendgroup=f"batch{b}_node{n}",
                        showlegend=False,
                        hovertemplate="Step: %{x}<br>Missing Value<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    # Plot x_mask - Input mask with improved styling
    for b in range(min(batch_size, 2)):
        for n in range(min(num_nodes, 4)):
            node_id = node_indices[n]
            color_idx = (b * num_nodes + n) % len(colors)

            fig.add_trace(
                go.Scatter(
                    x=list(range(seq_len)),
                    y=x_mask[b, n, :, 0],
                    mode="lines+markers",
                    marker=dict(size=6),
                    line=dict(color=colors[color_idx], width=2),
                    name=f"Batch {b}, Node {node_id}",
                    legendgroup=f"batch{b}_node{n}",
                    showlegend=False,
                    hovertemplate="Step: %{x}<br>Mask: %{y}<br>Node "
                    + str(node_id)
                    + "<extra></extra>",
                ),
                row=1,
                col=2,
            )

            # Add reference line at y=0.5 to help distinguish 0/1 values
            if b == 0 and n == 0:
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=seq_len - 1,
                    y0=0.5,
                    y1=0.5,
                    line=dict(color="gray", width=1, dash="dash"),
                    row=1,
                    col=2,
                )
                # Add annotations explaining the mask values
                fig.add_annotation(
                    text="1 = Valid data",
                    x=seq_len - 1,
                    y=0.8,
                    showarrow=False,
                    font=dict(size=10),
                    row=1,
                    col=2,
                )
                fig.add_annotation(
                    text="0 = Missing data",
                    x=seq_len - 1,
                    y=0.2,
                    showarrow=False,
                    font=dict(size=10),
                    row=1,
                    col=2,
                )

    # Plot y - Target data with improved styling
    _, _, horizon, _ = y.shape
    for b in range(min(batch_size, 2)):
        for n in range(min(num_nodes, 4)):
            node_id = node_indices[n]
            color_idx = (b * num_nodes + n) % len(colors)

            # Use non-broken lines and add markers
            mask_values = y_mask[b, n, :, 0]
            y_values = y[b, n, :, 0]

            # For missing values (mask=0), set y to None to create gaps
            y_vals = np.where(mask_values > 0, y_values, None)

            fig.add_trace(
                go.Scatter(
                    x=list(range(horizon)),
                    y=y_vals,
                    mode="lines+markers",
                    marker=dict(size=8),  # Larger markers for future values
                    line=dict(
                        color=colors[color_idx], width=2, dash="dash"
                    ),  # Dashed lines for future
                    name=f"Batch {b}, Node {node_id}",
                    legendgroup=f"batch{b}_node{n}",
                    showlegend=False,
                    hovertemplate="Future Step: %{x}<br>Value: %{y:.2f}<br>Node "
                    + str(node_id)
                    + "<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Add markers for missing values
            missing_indices = np.where(mask_values == 0)[0]
            if len(missing_indices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=missing_indices,
                        y=[None] * len(missing_indices),
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=8,
                            color=colors[color_idx],
                            line=dict(width=1, color="black"),
                        ),
                        name=f"{name} (Missing)",
                        legendgroup=f"batch{b}_node{n}",
                        showlegend=False,
                        hovertemplate="Future Step: %{x}<br>Missing Value<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

    # Plot y_mask - Target mask
    for b in range(min(batch_size, 2)):
        for n in range(min(num_nodes, 4)):
            node_id = node_indices[n]
            color_idx = (b * num_nodes + n) % len(colors)

            fig.add_trace(
                go.Scatter(
                    x=list(range(horizon)),
                    y=y_mask[b, n, :, 0],
                    mode="lines+markers",
                    marker=dict(size=8),
                    line=dict(color=colors[color_idx], width=2, dash="dash"),
                    name=f"Batch {b}, Node {node_id}",
                    legendgroup=f"batch{b}_node{n}",
                    showlegend=False,
                    hovertemplate="Future Step: %{x}<br>Mask: %{y}<br>Node "
                    + str(node_id)
                    + "<extra></extra>",
                ),
                row=2,
                col=2,
            )

            # Add reference line at y=0.5
            if b == 0 and n == 0:
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=horizon - 1,
                    y0=0.5,
                    y1=0.5,
                    line=dict(color="gray", width=1, dash="dash"),
                    row=2,
                    col=2,
                )

    # Add tensor shape annotations in a more structured box
    shapes = [
        f"<b>Tensor Shapes:</b>",
        f"x: {x.shape} [batch, nodes, time_steps, features]",
        f"x_mask: {x_mask.shape} [batch, nodes, time_steps, mask]",
        f"y: {y.shape} [batch, nodes, horizon, features]",
        f"y_mask: {y_mask.shape} [batch, nodes, horizon, mask]",
        f"adjacency: {adj.shape} [nodes, nodes]",
    ]

    # Add a legend explaining the tensor dimensions
    fig.add_annotation(
        text="<br>".join(shapes),
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        align="left",
    )

    # Add explanatory text about what the visualization is showing
    fig.add_annotation(
        text=(
            f"This visualization shows how data is organized in batches for the STGNN model.<br>"
            f"• The left column shows actual values, while the right column shows binary masks (1=valid, 0=missing).<br>"
            f"• The top row contains historical input data, and the bottom row contains future target data.<br>"
            f"• Each color represents a different node/sensor in the network."
        ),
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        align="left",
    )

    # Update layout with more informative styling
    fig.update_layout(
        title={
            "text": f"Batch Data Structure ({batch_size} batches, {num_nodes} nodes per batch)",
            "font": {"size": 18},
        },
        height=700,  # Taller figure for better visibility
        legend=dict(
            orientation="h",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(t=120, b=150),  # More space for annotations
    )

    # Update axes with more informative labels
    fig.update_xaxes(title_text="Time Step (Historical)", row=1, col=1)
    fig.update_xaxes(title_text="Time Step (Historical)", row=1, col=2)
    fig.update_xaxes(title_text="Time Step (Future)", row=2, col=1)
    fig.update_xaxes(title_text="Time Step (Future)", row=2, col=2)

    fig.update_yaxes(title_text="Standardized Value", row=1, col=1)
    fig.update_yaxes(title_text="Mask Value (1=Valid, 0=Missing)", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Value", row=2, col=1)
    fig.update_yaxes(title_text="Mask Value (1=Valid, 0=Missing)", row=2, col=2)

    return fig
