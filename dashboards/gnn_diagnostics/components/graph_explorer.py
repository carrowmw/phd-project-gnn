# dashboards/gnn_diagnostics/components/graph_explorer.py

import plotly.graph_objects as go
import numpy as np
import networkx as nx
from ..utils.data_utils import analyze_graph_connectivity


def create_graph_visualization(experiment_data):
    """
    Create a visualization of the graph structure

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data

    Returns:
    --------
    plotly.graph_objects.Figure
        Graph structure visualization
    """
    # Create a figure
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

    # Analyze graph connectivity
    connectivity_analysis = analyze_graph_connectivity(experiment_data)

    if "error" in connectivity_analysis:
        # Error in analysis
        fig.add_annotation(
            text=f"Error analyzing graph connectivity: {connectivity_analysis['error']}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig

    # Create a NetworkX graph
    G = nx.Graph()

    # Add edges from adjacency matrix
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=float(adj_matrix[i, j]))

    # Calculate node positions using a force-directed layout
    pos = nx.spring_layout(G, seed=42)

    # Calculate node degrees for sizing
    node_degrees = dict(G.degree())

    # Normalize node sizes based on degree
    max_degree = max(node_degrees.values()) if node_degrees else 1
    node_sizes = {node: 5 + (degree / max_degree * 20) for node, degree in node_degrees.items()}

    # Calculate edge widths based on weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    max_weight = max(edge_weights.values()) if edge_weights else 1
    edge_widths = {edge: 0.5 + (weight / max_weight * 3) for edge, weight in edge_weights.items()}

    # Calculate edge lengths for visualization
    edge_lengths = {edge: 1 / (weight + 0.1) for edge, weight in edge_weights.items()}

    # Add edges to the plot
    edge_x = []
    edge_y = []
    edge_colors = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Add the connection
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Edge color based on weight
        weight = edge_weights.get(edge, 0)
        color_intensity = min(1.0, weight / max_weight)
        edge_colors.extend([color_intensity, color_intensity, color_intensity])

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150, 150, 150, 0.5)'),
        hoverinfo='none',
        mode='lines'
    )

    # Add nodes to the plot
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node text for hover
        node_id = node_ids[node] if node < len(node_ids) else f"Unknown Node {node}"
        degree = node_degrees.get(node, 0)
        node_text.append(f"Node: {node_id}<br>Degree: {degree}")

        # Node size based on degree
        node_size.append(node_sizes.get(node, 5))

    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=[node_degrees.get(node, 0) for node in G.nodes()],
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0.5, color='darkgray')
        )
    )

    # Add traces to the figure
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    # Add markers for isolated nodes
    if connectivity_analysis.get("isolated_nodes", 0) > 0:
        isolated_nodes = [node for node, degree in node_degrees.items() if degree == 0]
        isolated_x = [pos[node][0] for node in isolated_nodes if node in pos]
        isolated_y = [pos[node][1] for node in isolated_nodes if node in pos]
        isolated_text = [f"Isolated Node: {node_ids[node]}" for node in isolated_nodes if node < len(node_ids)]

        fig.add_trace(go.Scatter(
            x=isolated_x,
            y=isolated_y,
            mode='markers',
            marker=dict(
                symbol='circle-open',
                size=15,
                color='red',
                line=dict(width=2, color='red')
            ),
            name='Isolated Nodes',
            text=isolated_text,
            hoverinfo='text'
        ))

    # Add annotations for graph statistics
    stats_text = (
        f"Graph Statistics:<br>"
        f"Nodes: {connectivity_analysis['num_nodes']}<br>"
        f"Edges: {connectivity_analysis['num_edges']}<br>"
        f"Graph Density: {connectivity_analysis['density']:.4f}<br>"
        f"Avg Degree: {connectivity_analysis['avg_degree']:.2f}<br>"
        f"Isolated Nodes: {connectivity_analysis['isolated_nodes']}<br>"
        f"Connected Components: {connectivity_analysis.get('num_components', 'N/A')}"
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

    # Add annotations for weight distribution
    weight_text = (
        f"Edge Weight Statistics:<br>"
        f"Mean: {connectivity_analysis['weight_mean']:.4f}<br>"
        f"Std Dev: {connectivity_analysis['weight_std']:.4f}<br>"
        f"Min: {connectivity_analysis['min_weight']:.4f}<br>"
        f"Max: {connectivity_analysis['max_weight']:.4f}<br>"
        f"Quartiles: [{connectivity_analysis['weight_quartiles'][0]:.2f}, "
        f"{connectivity_analysis['weight_quartiles'][1]:.2f}, "
        f"{connectivity_analysis['weight_quartiles'][2]:.2f}]"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.85,
        text=weight_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
    )

    # Add sigma_squared parameter as annotation
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
            y=0.73,
            text=params_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
        )

    # Update layout
    fig.update_layout(
        title="Graph Structure Analysis",
        hovermode='closest',
        showlegend=False,
        height=600,
        margin=dict(t=100, b=0, l=0, r=0),
    )

    # Update axes
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig