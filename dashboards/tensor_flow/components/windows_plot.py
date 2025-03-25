import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_windows_plot(X_by_sensor, masks_by_sensor, node_ids):
    """Create a more interpretable visualization of windowed data and masks"""
    # Check if we have enough data to visualize
    if not X_by_sensor or not any(
        node_id in X_by_sensor and len(X_by_sensor[node_id]) > 0 for node_id in node_ids
    ):
        return go.Figure().add_annotation(
            text="Not enough data to visualize windows",
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Get the actual node IDs that have data
    available_nodes = [
        node_id
        for node_id in node_ids
        if node_id in X_by_sensor and len(X_by_sensor[node_id]) > 0
    ]

    if not available_nodes:
        return go.Figure().add_annotation(
            text="No windows available for the provided node IDs",
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Limit to a reasonable number of nodes for clarity
    display_nodes = available_nodes[: min(4, len(available_nodes))]

    # Create a grid of subplots - one row per node
    fig = make_subplots(
        rows=len(display_nodes),
        cols=2,
        subplot_titles=[f"Node {node_id} Window Values" for node_id in display_nodes]
        + [f"Node {node_id} Mask Values" for node_id in display_nodes],
        column_widths=[0.7, 0.3],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # Define a color palette for the windows
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
    ]

    # Add a legend to explain what windows are
    legend_text = (
        "Windows are fixed-length sequences of data points extracted from continuous segments.<br>"
        "Each window becomes a training example for the model.<br>"
        "Windows with missing values use masks to indicate which values are valid."
    )

    fig.add_annotation(
        text=legend_text,
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

    for i, node_id in enumerate(display_nodes):
        # Get window data and masks
        windows = X_by_sensor[node_id]
        masks = masks_by_sensor[node_id]

        # Sample up to 5 windows for visualization
        num_windows = min(5, len(windows))

        # Add a small offset to each window for better visibility when stacked
        y_offset = 0.2

        # Add node statistics
        total_windows = len(windows)
        window_length = windows.shape[1] if windows.shape[0] > 0 else 0

        # Use proper domain references for Plotly
        left_x_ref = "x domain" if i == 0 else f"x{i*2+1} domain"
        left_y_ref = "y domain" if i == 0 else f"y{i*2+1} domain"

        right_x_ref = "x2 domain" if i == 0 else f"x{i*2+2} domain"
        right_y_ref = "y2 domain" if i == 0 else f"y{i*2+2} domain"

        # Add a textbox with node stats
        stats_text = (
            f"Node {node_id}:<br>"
            f"Total windows: {total_windows}<br>"
            f"Window length: {window_length}"
        )

        fig.add_annotation(
            text=stats_text,
            x=0.01,
            y=0.95,
            xref=left_x_ref,
            yref=left_y_ref,
            showarrow=False,
            font=dict(size=10),
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(255, 255, 255, 0.8)",
            align="left",
        )

        # For each window, create a more informative visualization
        for j in range(num_windows):
            window = windows[j]
            mask = masks[j]

            # Calculate y-position for this window (stacked from bottom to top)
            y_position = j * y_offset

            # Create filled area representing the window values
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(window))),
                    y=window + y_position,  # Add offset for stacking
                    mode="lines+markers",
                    line=dict(color=colors[j % len(colors)], width=2),
                    marker=dict(
                        size=8,
                        color=colors[j % len(colors)],
                        line=dict(width=1, color="white"),
                    ),
                    name=f"Window {j+1}",
                    legendgroup=f"window_{j}",
                    showlegend=(i == 0),  # Only show in legend for first node
                    hovertemplate=(
                        "Step: %{x}<br>"
                        "Value: %{customdata[0]:.2f}<br>"
                        "Valid: %{customdata[1]}<br>"
                        "Window: " + str(j + 1) + "<extra></extra>"
                    ),
                    customdata=np.column_stack((window, mask)),
                ),
                row=i + 1,
                col=1,
            )

            # Add markers for missing values
            missing_indices = np.where(mask == 0)[0]
            if len(missing_indices) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=missing_indices,
                        y=window[missing_indices] + y_position,  # Add offset
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=12,
                            color=colors[j % len(colors)],
                            line=dict(width=2, color="black"),
                        ),
                        name=f"Window {j+1} Missing",
                        legendgroup=f"window_{j}",
                        showlegend=False,
                        hovertemplate=(
                            "Step: %{x}<br>"
                            "Value: %{y:.2f}<br>"
                            "Missing Data Point<br>"
                            "Window: " + str(j + 1) + "<extra></extra>"
                        ),
                    ),
                    row=i + 1,
                    col=1,
                )

            # Create a heatmap-like visualization for masks using scatter points
            marker_colors = ["red" if m == 0 else "green" for m in mask]
            marker_symbols = ["x" if m == 0 else "circle" for m in mask]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mask))),
                    y=[j] * len(mask),  # One row per window
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=marker_colors,
                        symbol=marker_symbols,
                        line=dict(width=1, color="black"),
                    ),
                    name=f"Window {j+1} Mask",
                    legendgroup=f"window_{j}",
                    showlegend=False,
                    hovertemplate=(
                        "Step: %{x}<br>"
                        "Valid: %{text}<br>"
                        "Window: " + str(j + 1) + "<extra></extra>"
                    ),
                    text=["Yes" if m == 1 else "No" for m in mask],
                ),
                row=i + 1,
                col=2,
            )

        # Add a legend for the mask values in the first instance
        if i == 0:
            fig.add_annotation(
                text="Masks:<br>● = Valid<br>✕ = Missing",
                x=0.9,
                y=0.5,
                xref=right_x_ref,
                yref=right_y_ref,
                showarrow=False,
                font=dict(size=10),
                align="center",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
            )

        # Update y-axis for windows column
        fig.update_yaxes(
            title_text=f"Standardized Value (stacked)",
            row=i + 1,
            col=1,
        )

        # Update y-axis for masks column
        fig.update_yaxes(
            title_text="Window #",
            tickvals=list(range(num_windows)),
            ticktext=[f"Window {j+1}" for j in range(num_windows)],
            range=[-0.5, num_windows - 0.5],
            row=i + 1,
            col=2,
        )

        # Add x-axis label to bottom subplot only
        if i == len(display_nodes) - 1:
            fig.update_xaxes(title_text="Time Step", row=i + 1, col=1)
            fig.update_xaxes(title_text="Time Step", row=i + 1, col=2)

    # Add overall count of windows to the title
    total_windows_count = sum(len(X_by_sensor.get(node_id, [])) for node_id in node_ids)

    # Update layout with more informative title and styling
    fig.update_layout(
        title={
            "text": f"Windowed Data Representation ({total_windows_count} total windows across {len(available_nodes)} nodes)",
            "font": {"size": 18},
        },
        height=max(
            150 * len(display_nodes) + 200, 400
        ),  # Adaptive height based on number of nodes
        legend=dict(
            orientation="h",
            y=-0.1 if len(display_nodes) <= 2 else -0.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(t=120, b=80, l=60, r=60),
    )

    # Add a note about what a window is and how it's used
    note_text = (
        "Note: Only showing first 5 windows per node for clarity. "
        f"In total, there are {total_windows_count} windows available for training."
    )

    fig.add_annotation(
        text=note_text,
        x=0.5,
        y=-0.05 if len(display_nodes) <= 2 else -0.02,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    return fig
