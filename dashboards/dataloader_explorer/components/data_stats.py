# dashboards/dataloader_explorer/components/data_stats.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from ..utils.data_utils import get_dataloader_stats


def create_stats_panel(data_loaders, loader_key="train_loader"):
    """
    Create a statistical summary panel for a data loader

    Parameters:
    -----------
    data_loaders : dict
        Dictionary containing data loaders
    loader_key : str
        Key to access the loader from data_loaders

    Returns:
    --------
    plotly.graph_objects.Figure
        Data statistics visualization
    """
    try:
        # Get data loader statistics
        stats = get_dataloader_stats(data_loaders[loader_key])

        # Check for errors
        if "error" in stats:
            # Create an error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error computing statistics: {stats['error']}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

        # Create a figure with multiple subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Data Value Distribution",
                "Missing Data by Type",
                "Data Statistics by Type",
                "Adjacency Matrix Statistics",
            ],
            specs=[
                [{"type": "xy"}, {"type": "domain"}],
                [{"type": "table"}, {"type": "table"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
        )

        # 1. Create a box plot of data values distribution
        # Extract valid values from input data
        if (
            stats["x_stats"]["mean"] is not None
            and stats["y_stats"]["mean"] is not None
        ):
            # Create box plots
            boxplot_data = [
                go.Box(
                    name="Input Data",
                    y=[
                        stats["x_stats"]["min"],
                        stats["x_stats"]["mean"] - stats["x_stats"]["std"],
                        stats["x_stats"]["mean"],
                        stats["x_stats"]["mean"] + stats["x_stats"]["std"],
                        stats["x_stats"]["max"],
                    ],
                    boxpoints=False,
                    marker_color="blue",
                ),
                go.Box(
                    name="Target Data",
                    y=[
                        stats["y_stats"]["min"],
                        stats["y_stats"]["mean"] - stats["y_stats"]["std"],
                        stats["y_stats"]["mean"],
                        stats["y_stats"]["mean"] + stats["y_stats"]["std"],
                        stats["y_stats"]["max"],
                    ],
                    boxpoints=False,
                    marker_color="green",
                ),
            ]

            for trace in boxplot_data:
                fig.add_trace(trace, row=1, col=1)
        else:
            # Add a message if no valid data
            fig.add_annotation(
                text="Insufficient data for distribution analysis",
                xref="x",
                yref="y",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=12, color="gray"),
                row=1,
                col=1,
            )

        # 2. Create a pie chart of missing data
        labels = ["Valid Input", "Missing Input", "Valid Target", "Missing Target"]
        values = [
            100 - stats["input_missing_pct"],
            stats["input_missing_pct"],
            100 - stats["target_missing_pct"],
            stats["target_missing_pct"],
        ]

        colors = [
            "rgba(100, 149, 237, 0.8)",
            "rgba(255, 99, 71, 0.8)",
            "rgba(50, 205, 50, 0.8)",
            "rgba(255, 165, 0, 0.8)",
        ]

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                textinfo="label+percent",
                insidetextorientation="radial",
                marker=dict(colors=colors),
                hoverinfo="label+percent+value",
                hole=0.3,
            ),
            row=1,
            col=2,
        )

        # 3. Create a table of data statistics
        if (
            stats["x_stats"]["mean"] is not None
            and stats["y_stats"]["mean"] is not None
        ):
            stat_names = ["Mean", "Std Dev", "Min", "Max"]
            input_stats = [
                f"{stats['x_stats']['mean']:.4f}",
                f"{stats['x_stats']['std']:.4f}",
                f"{stats['x_stats']['min']:.4f}",
                f"{stats['x_stats']['max']:.4f}",
            ]
            target_stats = [
                f"{stats['y_stats']['mean']:.4f}",
                f"{stats['y_stats']['std']:.4f}",
                f"{stats['y_stats']['min']:.4f}",
                f"{stats['y_stats']['max']:.4f}",
            ]

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Statistic", "Input Data", "Target Data"],
                        fill_color="rgba(0, 0, 100, 0.1)",
                        align="center",
                        font=dict(size=12),
                    ),
                    cells=dict(
                        values=[stat_names, input_stats, target_stats],
                        fill_color=[
                            "rgba(0, 0, 0, 0.01)",
                            "rgba(100, 149, 237, 0.1)",
                            "rgba(50, 205, 50, 0.1)",
                        ],
                        align="center",
                        font=dict(size=11),
                    ),
                ),
                row=2,
                col=1,
            )
        else:
            # Add a message if no valid data
            fig.add_annotation(
                text="Insufficient data for statistical analysis",
                xref="x3",
                yref="y3",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=12, color="gray"),
                row=2,
                col=1,
            )

        # 4. Create a table of adjacency matrix statistics
        adj_stat_names = [
            "Min Weight",
            "Max Weight",
            "Mean Weight",
            "Sparsity (% zeros)",
        ]
        adj_stat_values = [
            f"{stats['adj_stats']['min']:.4f}",
            f"{stats['adj_stats']['max']:.4f}",
            f"{stats['adj_stats']['mean']:.4f}",
            f"{stats['adj_stats']['sparsity']:.2f}%",
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Statistic", "Value"],
                    fill_color="rgba(0, 0, 100, 0.1)",
                    align="center",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[adj_stat_names, adj_stat_values],
                    fill_color=["rgba(0, 0, 0, 0.01)", "rgba(144, 238, 144, 0.2)"],
                    align="center",
                    font=dict(size=11),
                ),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"DataLoader Statistics - {loader_key}",
            height=700,
            showlegend=False,
            margin=dict(t=100, b=50, l=50, r=50),
        )

        # Add a summary annotation
        summary_text = (
            f"Batch Size: {stats['batch_size']}<br>"
            f"Number of Nodes: {stats['num_nodes']}<br>"
            f"Input Length: {stats['seq_len']} time steps<br>"
            f"Target Length: {stats['horizon']} time steps<br>"
            f"Number of Features: {stats['features']}"
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
            text=f"Error creating statistics panel: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig
