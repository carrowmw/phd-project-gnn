import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_adjacency_plot(raw_adj, weighted_adj, node_ids):
    """Create a more interpretable visualization of the adjacency matrices"""
    # Ensure we have at least some data to plot
    if len(node_ids) < 2:
        # Create a figure with a warning message
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough nodes to visualize adjacency matrix (only found {len(node_ids)} nodes)",
            showarrow=False,
            font=dict(size=14, color="red"),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(title="Adjacency Matrix - Insufficient Data", height=400)
        return fig

    # Create a subplot with the raw and weighted adjacency matrices
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Raw Distance Matrix (Distance in meters)",
            "Weighted Adjacency Matrix (Edge Weights)",
        ],
        horizontal_spacing=0.12,
    )

    # Determine data ranges for better color scaling
    raw_max = np.max(raw_adj[raw_adj > 0])  # Max non-zero value
    raw_min = np.min(raw_adj[raw_adj > 0])  # Min non-zero value

    # Enhanced colorscale for distances
    distance_colorscale = [
        [0, "rgb(255,255,255)"],  # White for zero/missing
        [0.1, "rgb(240,240,255)"],  # Very light blue for short distances
        [0.3, "rgb(180,180,240)"],  # Light blue for medium distances
        [0.6, "rgb(120,120,220)"],  # Medium blue for longer distances
        [0.8, "rgb(50,50,200)"],  # Darker blue for longer distances
        [1, "rgb(0,0,150)"],  # Dark blue for maximum distances
    ]

    # Plot raw adjacency matrix with improved colorscale
    fig.add_trace(
        go.Heatmap(
            z=raw_adj,
            x=node_ids,
            y=node_ids,
            colorscale=distance_colorscale,
            colorbar=dict(title="Distance (m)", thickness=15, x=0.46, tickformat=".0f"),
            hoverongaps=False,
            name="Raw Distances",
            hovertemplate="From: %{y}<br>To: %{x}<br>Distance: %{z:.1f} meters<extra></extra>",
            showscale=True,
        ),
        row=1,
        col=1,
    )

    # Enhanced colorscale for weights - Green for stronger connections
    weight_colorscale = [
        [0, "rgb(255,255,255)"],  # White for zero/missing
        [0.1, "rgb(240,255,240)"],  # Very light green for weak connections
        [0.4, "rgb(180,240,180)"],  # Light green
        [0.7, "rgb(100,200,100)"],  # Medium green
        [0.9, "rgb(50,150,50)"],  # Darker green
        [1, "rgb(0,100,0)"],  # Dark green for strongest connections
    ]

    # Plot weighted adjacency matrix with improved colorscale
    fig.add_trace(
        go.Heatmap(
            z=weighted_adj,
            x=node_ids,
            y=node_ids,
            colorscale=weight_colorscale,
            colorbar=dict(title="Weight", thickness=15, x=1.0, tickformat=".2f"),
            hoverongaps=False,
            name="Connection Weights",
            hovertemplate="From: %{y}<br>To: %{x}<br>Weight: %{z:.3f}<extra></extra>",
            showscale=True,
        ),
        row=1,
        col=2,
    )

    # Add annotations explaining the matrices
    fig.add_annotation(
        text="Raw distances between sensor nodes (in meters)",
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=-0.15,
        showarrow=False,
        row=1,
        col=1,
    )

    fig.add_annotation(
        text="Transformed weights using Gaussian kernel<br>Closer nodes have higher weights",
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=-0.15,
        showarrow=False,
        row=1,
        col=2,
    )

    # Update layout with more informative title and sizing
    fig.update_layout(
        title={
            "text": f"Adjacency Matrix Transformation ({len(node_ids)} nodes)",
            "font": {"size": 18},
        },
        height=600,
        margin=dict(t=80, b=100),  # More space for title and annotations
    )

    # Update axes for better readability
    for i in range(1, 3):
        fig.update_xaxes(
            title_text="To Node", tickangle=45, row=1, col=i, tickfont=dict(size=10)
        )
        fig.update_yaxes(
            title_text="From Node",
            autorange="reversed",  # This puts (0,0) at the top-left
            row=1,
            col=i,
            tickfont=dict(size=10),
        )

    # Add a note about the number of nodes displayed
    fig.add_annotation(
        text=f"Displaying {len(node_ids)} out of approximately 15-30 total nodes in the dataset",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        font=dict(size=12, color="gray"),
    )

    return fig
