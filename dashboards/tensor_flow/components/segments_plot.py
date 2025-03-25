import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_segments_plot(time_series_dict, segments_dict):
    """Create a more interpretable plot showing continuous segments"""
    # Check if we have data to display
    if not segments_dict or not time_series_dict:
        return go.Figure().add_annotation(
            text="No segment data available",
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Get nodes that have both time series and segment data
    available_nodes = [
        node_id
        for node_id in segments_dict
        if node_id in time_series_dict and len(segments_dict[node_id]) > 0
    ]

    if not available_nodes:
        return go.Figure().add_annotation(
            text="No valid segments found in the data",
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Limit to a reasonable number of nodes for clarity (maximum 4)
    display_nodes = available_nodes[: min(4, len(available_nodes))]

    # Create a subplot for each node
    fig = make_subplots(
        rows=len(display_nodes),
        cols=1,
        subplot_titles=[
            f"Node {node_id} Continuous Segments" for node_id in display_nodes
        ],
        vertical_spacing=0.08,
    )

    # Define appealing colors for segments
    segment_colors = [
        "rgba(65, 105, 225, 0.3)",  # Royal Blue
        "rgba(46, 139, 87, 0.3)",  # Sea Green
        "rgba(220, 20, 60, 0.3)",  # Crimson
        "rgba(255, 140, 0, 0.3)",  # Dark Orange
        "rgba(148, 0, 211, 0.3)",  # Dark Violet
        "rgba(0, 128, 128, 0.3)",  # Teal
    ]

    # Define colors for segment borders (darker versions of fill colors)
    border_colors = [
        "rgba(65, 105, 225, 0.8)",  # Royal Blue
        "rgba(46, 139, 87, 0.8)",  # Sea Green
        "rgba(220, 20, 60, 0.8)",  # Crimson
        "rgba(255, 140, 0, 0.8)",  # Dark Orange
        "rgba(148, 0, 211, 0.8)",  # Dark Violet
        "rgba(0, 128, 128, 0.8)",  # Teal
    ]

    # Add an explanation about what segments are
    fig.add_annotation(
        text=(
            "Continuous segments are uninterrupted sequences of data points.<br>"
            "The algorithm identifies these segments by looking for gaps in the time series.<br>"
            "Each colored region represents a different segment that will be used to create training windows."
        ),
        x=0.5,
        y=1.05,
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

    # For each node, create a more informative visualization
    for i, node_id in enumerate(display_nodes):
        if node_id not in time_series_dict:
            continue

        series = time_series_dict[node_id]
        segments = segments_dict[node_id]

        # Summary statistics
        total_points = len(series)
        points_in_segments = sum(end - start for start, end in segments)
        coverage_pct = (
            (points_in_segments / total_points * 100) if total_points > 0 else 0
        )

        # Add statistics annotation
        stats_text = (
            f"Node {node_id}:<br>"
            f"Total segments: {len(segments)}<br>"
            f"Points in segments: {points_in_segments} / {total_points} ({coverage_pct:.1f}%)"
        )

        # Use proper domain references - the first subplot is just "x domain", subsequent ones are "x2 domain", "x3 domain", etc.
        # Similarly for y axis
        x_ref = "x domain" if i == 0 else f"x{i+1} domain"
        y_ref = "y domain" if i == 0 else f"y{i+1} domain"

        fig.add_annotation(
            text=stats_text,
            x=0.01,
            y=0.95,
            xref=x_ref,
            yref=y_ref,
            showarrow=False,
            font=dict(size=10),
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            align="left",
        )

        # Plot the raw time series as a light gray line
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name="Raw Data",
                line=dict(color="rgba(100, 100, 100, 0.5)", width=1),
                showlegend=(i == 0),  # Only show in legend for first node
                hovertemplate="Time: %{x}<br>Value: %{y:.2f}<extra>Raw Data</extra>",
            ),
            row=i + 1,
            col=1,
        )

        # Add markers to highlight actual data points
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="markers",
                marker=dict(
                    color="rgba(100, 100, 100, 0.5)",
                    size=3,
                ),
                name="Data Points",
                showlegend=(i == 0),
                hoverinfo="skip",
            ),
            row=i + 1,
            col=1,
        )

        # Add segment highlights with improved styling and information
        for j, (start_idx, end_idx) in enumerate(segments):
            # Skip if segment indices are invalid
            if (
                start_idx >= len(series.index)
                or end_idx > len(series.index)
                or start_idx >= end_idx
            ):
                continue

            segment_indices = series.index[start_idx:end_idx]
            if len(segment_indices) == 0:
                continue

            segment_values = series.values[start_idx:end_idx]

            # Calculate segment statistics
            segment_length = len(segment_indices)
            segment_min = (
                np.nanmin(segment_values) if len(segment_values) > 0 else np.nan
            )
            segment_max = (
                np.nanmax(segment_values) if len(segment_values) > 0 else np.nan
            )

            # Add a small padding to the y-range for better visibility
            y_padding = (
                (segment_max - segment_min) * 0.1
                if not np.isnan(segment_min) and not np.isnan(segment_max)
                else 0.1
            )
            display_min = (
                segment_min - y_padding if not np.isnan(segment_min) else series.min()
            )
            display_max = (
                segment_max + y_padding if not np.isnan(segment_max) else series.max()
            )

            # Use modulo to cycle through colors for segments
            color_idx = j % len(segment_colors)

            # Format dates for better display
            start_time = segment_indices[0]
            end_time = segment_indices[-1]
            if isinstance(start_time, pd.Timestamp) and isinstance(
                end_time, pd.Timestamp
            ):
                date_format = (
                    "%Y-%m-%d %H:%M"
                    if start_time.hour or start_time.minute
                    else "%Y-%m-%d"
                )
                start_str = start_time.strftime(date_format)
                end_str = end_time.strftime(date_format)
            else:
                start_str = str(start_time)
                end_str = str(end_time)

            # Create a more descriptive name for the segment
            segment_name = (
                f"Segment {j+1}: {start_str} to {end_str} ({segment_length} points)"
            )

            # Add a transparent polygon to highlight segment
            fig.add_trace(
                go.Scatter(
                    x=[
                        segment_indices[0],
                        segment_indices[0],
                        segment_indices[-1],
                        segment_indices[-1],
                        segment_indices[0],
                    ],
                    y=[display_min, display_max, display_max, display_min, display_min],
                    fill="toself",
                    mode="lines",
                    line=dict(color=border_colors[color_idx], width=1.5),
                    name=segment_name,
                    fillcolor=segment_colors[color_idx],
                    showlegend=(
                        i == 0 and j < 6
                    ),  # Only show first 6 segments in legend
                    legendgroup=f"segment_{j}",
                    hovertemplate=(
                        "Segment %{meta}<br>"
                        "Start: %{customdata[0]}<br>"
                        "End: %{customdata[1]}<br>"
                        "Length: %{customdata[2]} points<br>"
                        "<extra></extra>"
                    ),
                    customdata=[[start_str, end_str, segment_length]],
                    meta=j + 1,
                ),
                row=i + 1,
                col=1,
            )

            # Optionally add a label in the middle of the segment
            if segment_length > 30:  # Only label larger segments to avoid clutter
                middle_idx = start_idx + (end_idx - start_idx) // 2
                if middle_idx < len(series.index):
                    middle_time = series.index[middle_idx]
                    middle_y = (display_min + display_max) / 2

                    fig.add_annotation(
                        x=middle_time,
                        y=middle_y,
                        text=f"Seg {j+1}",
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="black",
                        borderwidth=1,
                        row=i + 1,
                        col=1,
                    )

        # Update y-axis title for middle subplot
        if i == len(display_nodes) // 2:
            fig.update_yaxes(title_text="Value", row=i + 1, col=1)

        # Update x-axis title for bottom subplot
        if i == len(display_nodes):
            fig.update_xaxes(title_text="Timestamp", row=i + 1, col=1)

    # Update overall layout
    fig.update_layout(
        title={
            "text": f"Continuous Segments in Time Series Data",
            "font": {"size": 18},
        },
        height=300 * len(display_nodes) + 100,  # Adaptive height
        legend=dict(
            orientation="h",
            y=-0.1 if len(display_nodes) <= 2 else -0.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(t=100, b=80),
    )

    # Add a note if we're not showing all nodes
    if len(available_nodes) > len(display_nodes):
        fig.add_annotation(
            text=f"Note: Only showing {len(display_nodes)} out of {len(available_nodes)} available nodes for clarity.",
            x=0.5,
            y=-0.05 if len(display_nodes) <= 2 else -0.02,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

    return fig
