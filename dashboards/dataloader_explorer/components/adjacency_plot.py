# dashboards/dataloader_explorer/components/adjacency_plot.py

import plotly.graph_objects as go
import numpy as np
import networkx as nx
from ..utils.data_utils import get_batch_from_loader


def create_adjacency_plot(data_loaders, loader_key="train_loader"):
    """
    Create an interactive visualization of the adjacency matrix

    Parameters:
    -----------
    data_loaders : dict
        Dictionary containing data loaders
    loader_key : str
        Key to access the loader from data_loaders

    Returns:
    --------
    plotly.graph_objects.Figure
        Adjacency matrix visualization
    """
    try:
        # Get batch data
        data_loader = data_loaders[loader_key]
        batch = get_batch_from_loader(data_loader)

        # Extract adjacency matrix and node indices
        adj = batch["adj"].cpu().numpy()
        node_indices = batch["node_indices"].cpu().numpy()

        # Create a figure with 2 subplots - heatmap and network graph
        fig = go.Figure()

        # Add adjacency matrix heatmap
        fig.add_trace(
            go.Heatmap(
                z=adj,
                x=[f"Node {id}" for id in node_indices],
                y=[f"Node {id}" for id in node_indices],
                colorscale="Viridis",
                colorbar=dict(title="Edge Weight", titleside="right"),
                hovertemplate="From: %{y}<br>To: %{x}<br>Weight: %{z:.4f}<extra></extra>",
                text=adj.round(3),  # Values to show in hover
            )
        )

        # Create a network layout using NetworkX for node positioning
        G = nx.from_numpy_array(adj)

        # Position nodes using a force-directed layout
        pos = nx.spring_layout(G, seed=42)

        # Get node positions
        node_x = []
        node_y = []
        for p in pos.values():
            node_x.append(p[0])
            node_y.append(p[1])

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[f"{id}" for id in node_indices],
            textposition="top center",
            marker=dict(
                size=15, color="lightblue", line=dict(width=1, color="darkblue")
            ),
            name="Nodes",
            hovertemplate="Node ID: %{text}<extra></extra>",
            visible=False,
        )

        # Create edge traces with varying line width based on edge weight
        edge_traces = []

        # Get edge weight range for normalization
        weight_min = np.min(adj[adj > 0]) if np.any(adj > 0) else 0.1
        weight_max = np.max(adj) if np.any(adj > 0) else 1.0

        # Function to normalize edge width
        def normalize_width(weight):
            return 1 + 8 * (weight - weight_min) / (weight_max - weight_min)

        # Create a trace for each edge weight range
        edge_weight_ranges = [
            (0, 0.25, "rgba(220, 220, 220, 0.3)"),  # Very light connections
            (0.25, 0.5, "rgba(144, 238, 144, 0.5)"),  # Light green
            (0.5, 0.75, "rgba(34, 139, 34, 0.7)"),  # Green
            (0.75, 1.01, "rgba(0, 100, 0, 0.9)"),  # Dark green
        ]

        for min_w, max_w, color in edge_weight_ranges:
            # Filter edges in this weight range
            edge_x = []
            edge_y = []
            edge_weights = []

            for i in range(len(node_indices)):
                for j in range(len(node_indices)):
                    weight = adj[i, j]

                    if weight <= 0:
                        continue

                    # Skip if not in range
                    norm_weight = weight / weight_max
                    if norm_weight < min_w or norm_weight >= max_w:
                        continue

                    edge_weights.append(weight)

                    x0, y0 = pos[i]
                    x1, y1 = pos[j]

                    # For curved edges, especially self-loops
                    if i == j:  # Self-loop
                        # Create a loop
                        loop_x = [x0, x0 + 0.1, x0 + 0.1, x0]
                        loop_y = [y0, y0 + 0.1, y0 - 0.1, y0]
                        edge_x.extend(loop_x)
                        edge_y.extend(loop_y)
                        edge_x.append(None)  # Add None to create a break
                        edge_y.append(None)
                    else:
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

            if edge_weights:
                widths = [normalize_width(w) for w in edge_weights]
                avg_width = np.mean(widths)

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=avg_width, color=color),
                    hoverinfo="none",
                    name=f"Weight: {min_w:.2f}-{max_w:.2f}",
                    visible=False,
                )

                edge_traces.append(edge_trace)

        # Add all edge traces to the figure
        for trace in edge_traces:
            fig.add_trace(trace)

        # Add node trace (after edges, so nodes appear on top)
        fig.add_trace(node_trace)

        # Update layout
        fig.update_layout(
            title="Adjacency Matrix Visualization",
            height=800,
            width=800,
            xaxis=dict(title="To Node", tickangle=45),
            yaxis=dict(
                title="From Node",
                autorange="reversed",  # Ensure (0,0) is at the top-left
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[
                                {"visible": [True] + [False] * (len(edge_traces) + 1)}
                            ],
                            label="Heatmap View",
                            method="update",
                        ),
                        dict(
                            args=[
                                {"visible": [False] + [True] * (len(edge_traces) + 1)}
                            ],
                            label="Network View",
                            method="update",
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="top",
                ),
            ],
            margin=dict(t=120, b=100, l=100, r=100),
        )

        # Add annotations explaining the visualization
        fig.add_annotation(
            text=(
                "Toggle between heatmap and network views using the buttons above.<br>"
                "Heatmap view shows the full adjacency matrix with cell values representing edge weights.<br>"
                "Network view shows the graph structure with edge thickness representing weight strength."
            ),
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

        # Update axes for network view
        fig.update_xaxes(
            showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,  # Equal aspect ratio
            range=[-1.2, 1.2],
        )

        return fig

    except Exception as e:
        # Create an error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating adjacency plot: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig
