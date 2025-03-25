import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_time_series_plot(time_series_dict, node_ids):
    """Create a more interpretable plot of the raw time series data"""
    # Check if we have data to plot
    available_nodes = [node_id for node_id in node_ids if node_id in time_series_dict]

    if not available_nodes:
        return go.Figure().add_annotation(
            text="No time series data available for the provided node IDs",
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Limit to a reasonable number of nodes for clarity (maximum 6)
    display_nodes = available_nodes[: min(6, len(available_nodes))]

    # Create a subplot for each node for better visibility
    fig = make_subplots(
        rows=len(display_nodes),
        cols=1,
        subplot_titles=[f"Node {node_id} Time Series" for node_id in display_nodes],
        vertical_spacing=0.05,
        shared_xaxes=True,
    )

    # Define a diverse color palette
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
    ]

    # Add an informative title at the top
    fig.add_annotation(
        text="Raw Time Series Data with Missing Values",
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="black"),
    )

    # Add an explanation of what this visualization shows
    fig.add_annotation(
        text=(
            "This visualization shows the raw sensor data over time.<br>"
            "Gaps in the lines represent missing data points that will need to be handled during processing.<br>"
            "Each sensor is shown in a separate subplot for clarity."
        ),
        x=0.5,
        y=1.03,
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

    # Plot each node's time series in its own subplot
    for i, node_id in enumerate(display_nodes):
        series = time_series_dict[node_id]
        color = colors[i % len(colors)]

        # Calculate statistics
        total_points = len(series)
        valid_points = series.count()  # Count non-NaN values
        missing_points = total_points - valid_points
        missing_percentage = (
            (missing_points / total_points * 100) if total_points > 0 else 0
        )

        # Plot line for non-NaN values with gaps for missing data
        fig.add_trace(
            go.Scatter(
                x=series.index[:1000],
                y=series.values[:1000],
                mode="lines",
                name=f"Node {node_id}",
                connectgaps=False,  # Don't connect across NaN values
                line=dict(color=color, width=1.5),
                hovertemplate=(
                    "Time: %{x}<br>"
                    "Value: %{y:.2f}<br>"
                    "Node: " + str(node_id) + "<extra></extra>"
                ),
            ),
            row=i + 1,
            col=1,
        )

        # Add markers to make the data points more visible
        fig.add_trace(
            go.Scatter(
                x=series.index[:1000],
                y=series.values[:1000],
                mode="markers",
                marker=dict(
                    color=color,
                    size=4,
                    opacity=0.6,
                ),
                name=f"Node {node_id} Points",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=i + 1,
            col=1,
        )

        # Add data quality statistics as annotations
        stats_text = (
            f"Data quality: {valid_points} valid / {total_points} total points<br>"
            f"Missing data: {missing_percentage:.1f}% ({missing_points} points)"
        )

        # Use proper domain references - the first subplot is just "x domain", subsequent ones are "x2 domain", "x3 domain", etc.
        # Similarly for y axis
        x_ref = "x domain" if i == 0 else f"x{i+1} domain"
        y_ref = "y domain" if i == 0 else f"y{i+1} domain"

        fig.add_annotation(
            text=stats_text,
            x=0.99,
            y=0.9,
            xref=x_ref,
            yref=y_ref,
            showarrow=False,
            font=dict(size=10),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
        )

        # Update y-axis title only for middle subplot to save space
        if i == len(display_nodes) // 2:
            fig.update_yaxes(title_text="Value", row=i + 1, col=1)

        # Update x-axis title only for bottom subplot
        if i == len(display_nodes) - 1:
            fig.update_xaxes(title_text="Timestamp", row=i + 1, col=1)

        # Highlight periods with missing data using shaded rectangles
        # Find gaps in the time series
        if isinstance(series.index, pd.DatetimeIndex) and len(series) > 1:
            try:
                # Calculate expected time step from the most common difference
                time_diffs = series.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    # Get most common time difference as the expected step
                    expected_step = time_diffs.mode()[0]

                    # Find gaps larger than the expected step
                    large_gaps = []
                    for j in range(1, len(series.index)):
                        actual_diff = series.index[j] - series.index[j - 1]
                        if (
                            actual_diff > expected_step * 2
                        ):  # Gap is significantly larger than expected
                            large_gaps.append((series.index[j - 1], series.index[j]))

                    # Highlight the large gaps
                    for gap_start, gap_end in large_gaps[
                        :10
                    ]:  # Limit to 10 gaps to avoid clutter
                        fig.add_shape(
                            type="rect",
                            x0=gap_start,
                            x1=gap_end,
                            y0=series.min(),
                            y1=series.max(),
                            fillcolor="rgba(200,200,200,0.3)",
                            line=dict(width=0),
                            row=i + 1,
                            col=1,
                        )
            except Exception as e:
                print(f"Warning: Could not highlight gaps for node {node_id}: {e}")

    # Update legend to be horizontal at the bottom
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2 if len(display_nodes) <= 3 else -0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        height=200 * len(display_nodes)
        + 100,  # Adaptive height based on number of nodes
        margin=dict(t=120, b=100),
    )

    # Add a note if not all nodes are shown
    if len(available_nodes) > len(display_nodes):
        fig.add_annotation(
            text=f"Note: Only showing {len(display_nodes)} out of {len(available_nodes)} available nodes for clarity.",
            x=0.5,
            y=-0.05 if len(display_nodes) <= 3 else -0.02,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

    return fig
