# dashboards/dataloader_explorer/components/correlation_plot.py

import plotly.graph_objects as go
import numpy as np
from ..utils.data_utils import get_batch_from_loader, compute_node_correlations


def create_correlation_plot(data_loaders, loader_key="train_loader", batch_idx=0):
    """
    Create a correlation matrix visualization between nodes

    Parameters:
    -----------
    data_loaders : dict
        Dictionary containing data loaders
    loader_key : str
        Key to access the loader from data_loaders
    batch_idx : int
        Index of the batch to analyze

    Returns:
    --------
    plotly.graph_objects.Figure
        Correlation matrix figure
    """
    try:
        # Get batch data
        data_loader = data_loaders[loader_key]
        batch = get_batch_from_loader(data_loader, batch_idx)

        # Compute node correlations
        corr_matrix, node_ids = compute_node_correlations(batch, batch_idx)

        # Create correlation heatmap
        fig = go.Figure()

        # Add correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=[f"Node {id}" for id in node_ids],
                y=[f"Node {id}" for id in node_ids],
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="Correlation",
                    titleside="right",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
                ),
                hovertemplate="Node X: %{x}<br>Node Y: %{y}<br>Correlation: %{z:.4f}<extra></extra>",
                text=corr_matrix.values.round(2),  # Values to show in hover
            )
        )

        # Add a markers for NaN values
        nan_mask = np.isnan(corr_matrix.values)
        if np.any(nan_mask):
            nan_indices = np.where(nan_mask)
            fig.add_trace(
                go.Scatter(
                    x=[f"Node {node_ids[i]}" for i in nan_indices[1]],
                    y=[f"Node {node_ids[i]}" for i in nan_indices[0]],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=8,
                        color="gray",
                        line=dict(width=1, color="darkgray"),
                    ),
                    name="Insufficient Data",
                    hoverinfo="skip",
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Node Correlation Matrix - {loader_key} (Batch {batch_idx})",
            height=700,
            width=800,
            xaxis=dict(title="Node", tickangle=45),
            yaxis=dict(
                title="Node", autorange="reversed"  # Ensure (0,0) is at the top-left
            ),
            margin=dict(t=100, b=100, l=100, r=100),
        )

        # Add a summary annotation
        summary_text = (
            f"Correlation matrix showing the Pearson correlation coefficient<br>"
            f"between time series data from different nodes.<br>"
            f"Values range from -1 (strong negative correlation) to 1 (strong positive correlation).<br>"
            f"0 indicates no linear correlation between nodes.<br>"
            f"X markers indicate insufficient data for correlation calculation."
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
            text=f"Error creating correlation plot: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig
