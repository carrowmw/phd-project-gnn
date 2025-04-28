# src/visualization/visualization_utils.py (New consolidated file)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.dates import DateFormatter
from pathlib import Path
import os
from typing import Dict, Any, List, Tuple, Optional, Union

from gnn_package.src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VisualizationManager:
    """Centralized visualization manager for consistent plots across the package"""

    def __init__(self, theme="default", figsize_base=(10, 6)):
        """
        Initialize visualization manager.

        Parameters:
        -----------
        theme : str
            Visualization theme name
        figsize_base : tuple
            Base figure size (width, height)
        """
        self.theme = theme
        self.figsize_base = figsize_base
        self.logger = get_logger(__name__)

        # Set up plotting style based on theme
        if theme == "default":
            plt.style.use("seaborn-v0_8-whitegrid")
        elif theme == "dark":
            plt.style.use("dark_background")
        elif theme == "minimal":
            plt.style.use("seaborn-v0_8-white")

        # Theme-specific colors
        self.theme_colors = {
            "default": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "tertiary": "#2ca02c",
                "quaternary": "#d62728",
                "highlight": "#9467bd",
            },
            "dark": {
                "primary": "#56b4e9",
                "secondary": "#f0e442",
                "tertiary": "#009e73",
                "quaternary": "#e69f00",
                "highlight": "#cc79a7",
            },
            "minimal": {
                "primary": "#4e79a7",
                "secondary": "#f28e2c",
                "tertiary": "#59a14f",
                "quaternary": "#e15759",
                "highlight": "#b07aa1",
            },
        }

        # Use default theme colors if theme not found
        self.colors = self.theme_colors.get(theme, self.theme_colors["default"])

    def get_figure(self, width_scale=1.0, height_scale=1.0):
        """
        Get a figure with the theme's style.

        Parameters:
        -----------
        width_scale : float
            Scale factor for width
        height_scale : float
            Scale factor for height

        Returns:
        --------
        matplotlib.figure.Figure
            Figure configured with theme style
        """
        figsize = (
            self.figsize_base[0] * width_scale,
            self.figsize_base[1] * height_scale,
        )
        return plt.figure(figsize=figsize)

    def plot_time_series(
        self, time_index, values, label=None, ax=None, color=None, **kwargs
    ):
        """
        Plot a time series with consistent styling.

        Parameters:
        -----------
        time_index : array-like
            X values (timestamps)
        values : array-like
            Y values
        label : str, optional
            Legend label
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        color : str, optional
            Line color. If None, uses theme color.
        **kwargs : dict
            Additional keyword arguments for plot

        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_base)

        if color is None:
            color = self.colors["primary"]

        line = ax.plot(time_index, values, color=color, label=label, **kwargs)

        # Apply theme styling
        ax.grid(True, alpha=0.3)
        if label:
            ax.legend()

        # Format time axis if using datetime
        if pd.api.types.is_datetime64_any_dtype(time_index) or isinstance(
            time_index[0], (datetime, pd.Timestamp)
        ):
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
            ax.tick_params(axis="x", rotation=45)

        return ax

    def plot_comparison(
        self,
        time_index,
        actual,
        predicted,
        title=None,
        ax=None,
        y_limits=None,
        show_metrics=True,
        **kwargs,
    ):
        """
        Plot actual vs predicted values with metrics.

        Parameters:
        -----------
        time_index : array-like
            X values (timestamps)
        actual : array-like
            Actual values
        predicted : array-like
            Predicted values
        title : str, optional
            Plot title
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        show_metrics : bool
            Whether to show error metrics on plot
        **kwargs : dict
            Additional keyword arguments for plot

        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_base)

        # Plot actual and predicted
        ax.plot(
            time_index, actual, color=self.colors["primary"], label="Actual", **kwargs
        )
        ax.plot(
            time_index,
            predicted,
            color=self.colors["secondary"],
            label="Predicted",
            linestyle="--",
            **kwargs,
        )

        # Apply title
        if title:
            ax.set_title(title)

        # Set y-axis limits if provided
        if y_limits is not None:
            global_min, global_max = y_limits
            ax.set_ylim(global_min, global_max)

        # Add metrics if requested
        if show_metrics and len(actual) > 0 and len(predicted) > 0:
            # Calculate metrics
            mse = ((np.array(actual) - np.array(predicted)) ** 2).mean()
            mae = np.abs(np.array(actual) - np.array(predicted)).mean()

            # Add text box with metrics
            ax.text(
                0.05,
                0.95,
                f"MSE: {mse:.4f}\nMAE: {mae:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox={"boxstyle": "round", "alpha": 0.5},
            )

        # Style the plot
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Format time axis if using datetime
        if pd.api.types.is_datetime64_any_dtype(time_index) or isinstance(
            time_index[0], (datetime, pd.Timestamp)
        ):
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
            ax.tick_params(axis="x", rotation=45)

        return ax

    def plot_error_distribution(self, errors, ax=None, bins=30):
        """
        Plot error distribution histogram.

        Parameters:
        -----------
        errors : array-like
            Error values
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        bins : int
            Number of histogram bins

        Returns:
        --------
        matplotlib.axes.Axes
            The axes object containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_base)

        ax.hist(
            errors,
            bins=bins,
            color=self.colors["primary"],
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_title("Error Distribution")
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Add mean and std lines
        if len(errors) > 0:
            mean = np.mean(errors)
            std = np.std(errors)
            ax.axvline(
                mean,
                color=self.colors["secondary"],
                linestyle="--",
                label=f"Mean: {mean:.4f}",
            )
            ax.axvline(
                mean + std,
                color=self.colors["tertiary"],
                linestyle=":",
                label=f"±1 Std: {std:.4f}",
            )
            ax.axvline(mean - std, color=self.colors["tertiary"], linestyle=":")
            ax.legend()

        return ax

    def create_sensor_grid(
        self, predictions_df, plots_per_row=4, max_sensors=16, figsize=None
    ):
        """
        Create a grid of plots showing predictions for multiple sensors.

        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame with prediction results
        plots_per_row : int
            Number of plots per row
        max_sensors : int
            Maximum number of sensors to show
        figsize : tuple, optional
            Figure size. If None, calculated based on grid size.

        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing the grid of plots
        """
        # Get unique sensors (limit to max_sensors)
        unique_sensors = predictions_df["node_id"].unique()[:max_sensors]
        num_sensors = len(unique_sensors)

        # Calculate grid dimensions
        num_rows = (num_sensors + plots_per_row - 1) // plots_per_row

        # Calculate figure size if not provided
        if figsize is None:
            width = self.figsize_base[0] * (plots_per_row / 2)
            height = self.figsize_base[1] * (num_rows / 2)
            figsize = (width, height)

        # Create figure and axes
        fig, axes = plt.subplots(num_rows, plots_per_row, figsize=figsize)
        if num_rows == 1 and plots_per_row == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        global_min = float("inf")
        global_max = float("-inf")

        for sensor_id in unique_sensors:
                # Get min and max values for both predictions and actuals

                sensor_data = predictions_df[predictions_df["node_id"] == sensor_id]

                sensor_data = sensor_data[
                    (sensor_data["prediction"] > -10)
                    & (sensor_data["actual"] > -10)
                ]

                if len(sensor_data) > 0:
                    # Update global min and max if we have valid data
                    if 'prediction' in sensor_data.columns and 'actual' in sensor_data.columns:
                        predictions_min = sensor_data["prediction"].min()
                        predictions_max = sensor_data["prediction"].max()

                        # Handle NaN values safely
                        if pd.notna(predictions_min) and pd.notna(predictions_max):
                            # Get min/max for both predictions and actuals
                            actuals = sensor_data["actual"].dropna()
                            if not actuals.empty:
                                actuals_min = actuals.min()
                                actuals_max = actuals.max()

                                # Update global min and max
                                global_min = min(global_min, predictions_min, actuals_min)
                                global_max = max(global_max, predictions_max, actuals_max)

        # Handle case where we didn't find any valid data
        if global_min == float('inf') or global_max == float('-inf'):
            global_min = 0
            global_max = 1

        # Add a small buffer to the limits (5% padding)
        y_range = global_max - global_min
        global_min = global_min - 0.05 * y_range if y_range > 0 else global_min - 1
        global_max = global_max + 0.05 * y_range if y_range > 0 else global_max + 1

        # Plot each sensor
        for i, sensor_id in enumerate(unique_sensors):
            if i >= len(axes):
                break

            # Get data for this sensor
            sensor_data = predictions_df[predictions_df["node_id"] == sensor_id]

            if len(sensor_data) > 0:
                # Get sensor name
                sensor_name = sensor_data["sensor_name"].iloc[0]

                # Plot comparison
                self.plot_comparison(
                    sensor_data["timestamp"],
                    sensor_data["actual"],
                    sensor_data["prediction"],
                    title=f"{sensor_name}",
                    ax=axes[i],
                    y_limits=(global_min, global_max),
                    show_metrics=True,
                )
            else:
                axes[i].text(
                    0.5, 0.5, f"No data for {sensor_id}", ha="center", va="center"
                )
                axes[i].set_axis_off()

        # Turn off any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_axis_off()

        plt.tight_layout()
        return fig

    def save_visualization_pack(
        self,
        predictions_df,
        output_dir,
        timestamp=None,
        include_grid=True,
        include_error_dist=True,
    ):
        """
        Save a comprehensive set of visualizations.

        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame with prediction results
        output_dir : str or Path
            Directory to save visualizations
        timestamp : str, optional
            Timestamp string for filenames
        include_grid : bool
            Whether to include sensor grid visualization
        include_error_dist : bool
            Whether to include error distribution visualization

        Returns:
        --------
        dict
            Dictionary with paths to saved visualizations
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Dictionary to store visualization paths
        viz_paths = {}

        # Create and save sensor grid
        if include_grid and len(predictions_df) > 0:
            try:
                grid_fig = self.create_sensor_grid(predictions_df)
                grid_path = output_dir / f"sensors_grid_{timestamp}.png"
                grid_fig.savefig(grid_path, dpi=150, bbox_inches="tight")
                plt.close(grid_fig)
                viz_paths["grid_plot"] = str(grid_path)
                self.logger.info(f"Saved sensor grid to {grid_path}")
            except Exception as e:
                self.logger.error(f"Error creating sensor grid: {str(e)}")

        # Create and save error distribution
        if (
            include_error_dist
            and len(predictions_df) > 0
            and "error" in predictions_df.columns
        ):
            try:
                fig, ax = plt.subplots(figsize=self.figsize_base)
                self.plot_error_distribution(predictions_df["error"], ax=ax)
                error_path = output_dir / f"error_distribution_{timestamp}.png"
                fig.savefig(error_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                viz_paths["error_dist"] = str(error_path)
                self.logger.info(f"Saved error distribution to {error_path}")
            except Exception as e:
                self.logger.error(f"Error creating error distribution: {str(e)}")

        return viz_paths
