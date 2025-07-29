# dashboards/gnn_diagnostics/components/loss_analyzer.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_loss_curve_visualization(experiment_data, comparison_data=None):
    """
    Create a visualization of training loss curves

    Parameters:
    -----------
    experiment_data : dict
        Dictionary containing experiment data
    comparison_data : dict, optional
        Dictionary containing comparison experiment data

    Returns:
    --------
    plotly.graph_objects.Figure
        Loss curve visualization
    """
    # Create a figure
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["Training and Validation Loss"],
    )

    # First try to get loss curves from the training_curves dict
    if "training_curves" in experiment_data:
        curves = experiment_data["training_curves"]

        if "train_losses" in curves and "val_losses" in curves:
            train_losses = curves["train_losses"]
            val_losses = curves["val_losses"]

            # Create epoch numbers
            epochs = list(range(1, len(train_losses) + 1))

            # Plot train loss
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_losses,
                    mode="lines",
                    name=f"{experiment_data['experiment_name']} Train Loss",
                    line=dict(color="blue", width=2),
                ),
            )

            # Plot validation loss
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=val_losses,
                    mode="lines",
                    name=f"{experiment_data['experiment_name']} Validation Loss",
                    line=dict(color="red", width=2),
                ),
            )

            # Add a marker for best validation loss
            best_epoch = np.argmin(val_losses) + 1
            best_loss = min(val_losses)

            fig.add_trace(
                go.Scatter(
                    x=[best_epoch],
                    y=[best_loss],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=12,
                        color="gold",
                        line=dict(width=1, color="black"),
                    ),
                    name=f"Best Validation: {best_loss:.6f} (Epoch {best_epoch})",
                ),
            )

            # Check if early stopping occurred
            if len(epochs) < experiment_data.get("config", {}).get("training", {}).get("num_epochs", float("inf")):
                early_stop_epoch = len(epochs)
                fig.add_annotation(
                    x=early_stop_epoch,
                    y=val_losses[-1],
                    text=f"Early stopping at epoch {early_stop_epoch}",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="black",
                    arrowsize=1,
                    arrowwidth=1,
                    ax=-40,
                    ay=-40,
                )

            # Add comparison if available
            if comparison_data and "training_curves" in comparison_data:
                comp_curves = comparison_data["training_curves"]

                if "train_losses" in comp_curves and "val_losses" in comp_curves:
                    comp_train_losses = comp_curves["train_losses"]
                    comp_val_losses = comp_curves["val_losses"]

                    # Create epoch numbers
                    comp_epochs = list(range(1, len(comp_train_losses) + 1))

                    # Plot train loss
                    fig.add_trace(
                        go.Scatter(
                            x=comp_epochs,
                            y=comp_train_losses,
                            mode="lines",
                            name=f"{comparison_data['experiment_name']} Train Loss",
                            line=dict(color="lightblue", width=2, dash="dash"),
                        ),
                    )

                    # Plot validation loss
                    fig.add_trace(
                        go.Scatter(
                            x=comp_epochs,
                            y=comp_val_losses,
                            mode="lines",
                            name=f"{comparison_data['experiment_name']} Validation Loss",
                            line=dict(color="pink", width=2, dash="dash"),
                        ),
                    )

                    # Add a marker for best validation loss
                    comp_best_epoch = np.argmin(comp_val_losses) + 1
                    comp_best_loss = min(comp_val_losses)

                    fig.add_trace(
                        go.Scatter(
                            x=[comp_best_epoch],
                            y=[comp_best_loss],
                            mode="markers",
                            marker=dict(
                                symbol="star",
                                size=12,
                                color="orange",
                                line=dict(width=1, color="black"),
                            ),
                            name=f"Comparison Best: {comp_best_loss:.6f} (Epoch {comp_best_epoch})",
                        ),
                    )

            # Add loss statistics as an annotation
            train_start = train_losses[0]
            train_end = train_losses[-1]
            train_improvement = (train_start - train_end) / train_start * 100 if train_start > 0 else 0

            val_start = val_losses[0]
            val_end = val_losses[-1]
            val_improvement = (val_start - best_loss) / val_start * 100 if val_start > 0 else 0

            stats_text = (
                f"Training Loss: {train_start:.6f} → {train_end:.6f} ({train_improvement:.1f}% improvement)<br>"
                f"Validation Loss: {val_start:.6f} → {best_loss:.6f} ({val_improvement:.1f}% improvement)<br>"
                f"Best Validation Epoch: {best_epoch}/{len(epochs)}"
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
    else:
        # No training curves found
        fig.add_annotation(
            text="No training curves found for this experiment",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )

    # Update layout
    fig.update_layout(
        title="Training Loss Analysis",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=100, b=100, l=50, r=50),
    )

    # Set y-axis to log scale for better visualization of loss curves
    fig.update_yaxes(type="log")

    return fig