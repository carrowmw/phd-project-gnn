# dashboards/gnn_diagnostics/components/adjacency_analyzer.py

import plotly.graph_objects as go
import numpy as np


def create_adjacency_analysis(experiment_data):
    """
    Create a visualization of the adjacency matrix properties

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data

    Returns:
    --------
    plotly.graph_objects.Figure
        Adjacency matrix visualization
    """
    # Create a figure with two subplots - adjacency matrix heatmap and edge weight distribution
    fig = go.Figure()

    # First check if graph data is available
    if "graph_data" not in experiment_data or "adj_matrix" not in experiment_data["graph_data"]:
        # No graph data
        fig.add_annotation(
            text="No graph data available for this experiment",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig

    # Get graph data
    graph_data = experiment_data["graph_data"]
    adj_matrix = graph_data["adj_matrix"]
    node_ids = graph_data["node_ids"]

    # Create a heatmap of the adjacency matrix
    heatmap_trace = go.Heatmap(
        z=adj_matrix,
        x=[str(node_id) for node_id in node_ids],
        y=[str(node_id) for node_id in node_ids],
        colorscale='Viridis',
        colorbar=dict(
            title='Edge Weight',
            thickness=15,
            titleside='right'
        ),
        hoverongaps=False,
        name='Adjacency Matrix'
    )

    # Add the heatmap trace
    fig.add_trace(heatmap_trace)

    # Get adjacency matrix statistics
    # Extract non-zero weights
    weights = adj_matrix[adj_matrix > 0]

    # Calculate statistics
    if len(weights) > 0:
        weight_min = np.min(weights)
        weight_max = np.max(weights)
        weight_mean = np.mean(weights)
        weight_median = np.median(weights)
        weight_std = np.std(weights)
        weight_q1, weight_q3 = np.percentile(weights, [25, 75])

        num_edges = len(weights)
        num_nodes = len(node_ids)
        density = num_edges / (num_nodes * (num_nodes - 1) / 2)

        weight_25, weight_50, weight_75 = np.percentile(weights, [25, 50, 75])

        # Add annotations
        stats_text = (
            f"Adjacency Matrix Statistics:<br>"
            f"Nodes: {num_nodes}<br>"
            f"Edges: {num_edges}<br>"
            f"Density: {density:.4f}<br>"
            f"Min Weight: {weight_min:.4f}<br>"
            f"Max Weight: {weight_max:.4f}<br>"
            f"Mean Weight: {weight_mean:.4f}<br>"
            f"Median Weight: {weight_median:.4f}<br>"
            f"Std Dev: {weight_std:.4f}<br>"
            f"Quartiles: [{weight_q1:.4f}, {weight_median:.4f}, {weight_q3:.4f}]"
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=stats_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
        )

        # Get sigma_squared parameter as annotation
        sigma_squared = None
        if "config" in experiment_data and "data" in experiment_data["config"] and "general" in experiment_data["config"]["data"]:
            sigma_squared = experiment_data["config"]["data"]["general"].get("sigma_squared", None)
            epsilon = experiment_data["config"]["data"]["general"].get("epsilon", None)

        if sigma_squared is not None:
            params_text = (
                f"Graph Parameters:<br>"
                f"sigma_squared: {sigma_squared}<br>"
                f"epsilon: {epsilon}"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.85,
                text=params_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
            )
    else:
        # No weights
        fig.add_annotation(
            text="No non-zero weights found in adjacency matrix",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Update layout
    fig.update_layout(
        title="Adjacency Matrix Analysis",
        height=700,
        margin=dict(t=100, b=100, l=100, r=100),
        xaxis=dict(title="Node ID"),
        yaxis=dict(title="Node ID"),
    )

    return fig