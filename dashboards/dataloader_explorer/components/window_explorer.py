# dashboards/dataloader_explorer/components/window_explorer.py

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from ..utils.data_utils import get_batch_from_loader


def create_window_explorer(
    data_loaders, loader_key="train_loader", num_windows=10, node_idx=0
):
    """
    Create a visualization of multiple windows for a specific node

    Parameters:
    -----------
    data_loaders : dict
        Dictionary containing data loaders
    loader_key : str
        Key to access the loader from data_loaders
    num_windows : int
        Number of windows to display
    node_idx : int
        Index of the node to visualize

    Returns:
    --------
    plotly.graph_objects.Figure
        Window explorer figure
    """
    try:
        # Get dataloader
        data_loader = data_loaders[loader_key]

        # Create dataloader iterator
        iterator = iter(data_loader)

        # Collect windows
        all_windows = []
        all_masks = []
        node_ids = []
        batch_indices = []
        window_indices = []

        # Process batches to collect windows
        batch_idx = 0
        window_count = 0

        while window_count < num_windows:
            try:
                # Get next batch
                batch = next(iterator)

                # Extract batch components
                x = batch["x"]
                x_mask = batch["x_mask"]
                node_indices = batch["node_indices"]

                # Check if node_idx is valid
                if node_idx >= len(node_indices):
                    print(
                        f"Warning: node_idx {node_idx} is out of range (max {len(node_indices)-1})"
                    )
                    break

                # Get the node ID
                node_id = node_indices[node_idx].item()

                # Store node ID
                if not node_ids:
                    node_ids.append(node_id)
                elif node_id != node_ids[0]:
                    print(f"Warning: Node ID changed from {node_ids[0]} to {node_id}")
                    node_ids.append(node_id)
                    continue

                # Get all windows for this node in this batch
                batch_size = x.shape[0]
                for b in range(batch_size):
                    # Get window data and mask
                    window_data = x[b, node_idx, :, 0].cpu().numpy()
                    window_mask = x_mask[b, node_idx, :, 0].cpu().numpy()

                    # Record window data and batch/window indices
                    all_windows.append(window_data)
                    all_masks.append(window_mask)
                    batch_indices.append(batch_idx)
                    window_indices.append(b)

                    window_count += 1
                    if window_count >= num_windows:
                        break

                batch_idx += 1

            except StopIteration:
                print(f"Reached end of dataloader. Found {len(all_windows)} windows.")
                break

        # Create visualization if we have windows
        if not all_windows:
            # Create empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text="No windows found for the selected node.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

        # Convert to arrays
        windows_array = np.array(all_windows)
        masks_array = np.array(all_masks)
        seq_len = windows_array.shape[1]

        # Create a heatmap figure for the collected windows
        fig = go.Figure()

        # Determine value range for consistent color scaling
        valid_values = windows_array[masks_array > 0]
        if len(valid_values) > 0:
            vmin = np.min(valid_values)
            vmax = np.max(valid_values)
        else:
            vmin = -1
            vmax = 1

        # Create heatmap of window values
        heatmap_z = np.copy(windows_array)
        # Replace invalid values with NaN for visualization
        heatmap_z[masks_array == 0] = np.nan

        fig.add_trace(
            go.Heatmap(
                z=heatmap_z,
                x=[f"Step {i+1}" for i in range(seq_len)],
                y=[f"Window {i+1}" for i in range(len(all_windows))],
                colorscale="Viridis",
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="Value"),
                hovertemplate="Window: %{y}<br>Step: %{x}<br>Value: %{z:.4f}<extra></extra>",
            )
        )

        # Highlight missing values with markers
        missing_y = []
        missing_x = []
        for i, (window, mask) in enumerate(zip(windows_array, masks_array)):
            for j, (val, m) in enumerate(zip(window, mask)):
                if m == 0:
                    missing_y.append(i)
                    missing_x.append(j)

        if missing_y:
            fig.add_trace(
                go.Scatter(
                    x=[f"Step {x+1}" for x in missing_x],
                    y=[f"Window {y+1}" for y in missing_y],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=8,
                        color="red",
                        line=dict(width=1, color="darkred"),
                    ),
                    name="Missing Values",
                    hoverinfo="skip",
                )
            )

        # Update layout
        node_id = node_ids[0] if node_ids else "unknown"
        fig.update_layout(
            title=f"Window Explorer - Node {node_id} ({loader_key})",
            height=max(500, min(len(all_windows) * 30 + 200, 800)),
            xaxis_title="Time Step",
            yaxis_title="Window",
            yaxis=dict(autorange="reversed"),  # Put Window 1 at the top
            margin=dict(t=100, b=50, l=50, r=50),
        )

        # Add a summary annotation
        summary_text = (
            f"Node ID: {node_id}<br>"
            f"Total Windows: {len(all_windows)}<br>"
            f"Window Length: {seq_len}<br>"
            f"Missing Data: {100 - (np.sum(masks_array) / masks_array.size * 100):.1f}%"
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

    except Exception as e:
        # Create an error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating window explorer: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig
