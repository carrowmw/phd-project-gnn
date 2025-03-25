#!/usr/bin/env python3
# Run this script to generate the GNN Tensor Flow Dashboard

from pathlib import Path
import webbrowser
import importlib.util
import os
import pandas as pd

import plotly.io as pio
import plotly.graph_objects as go

from gnn_package.src.preprocessing import (
    TimeSeriesPreprocessor,
    compute_adjacency_matrix,
)
from gnn_package.src.models.stgnn import create_stgnn_model
from gnn_package.src.dataloaders import create_dataloader
from dashboards.tensor_flow.components import (
    create_time_series_plot,
    create_segments_plot,
    create_windows_plot,
    create_adjacency_plot,
    create_batch_plot,
    create_model_plot,
)
from dashboards.eda.utils import (
    find_continuous_segments,
)
from dashboards.tensor_flow.utils import (
    load_sample_data,
)
from dashboards.eda.utils.template_utils import (
    load_template,
    render_template,
    get_template_path,
)

from gnn_package import paths


# Set up Plotly theme
pio.templates.default = "plotly_white"


def visualize_tensor_transformations(sample_data, window_size=12, horizon=6):
    """Create visualizations for each transformation step"""

    # Get input data
    adj_matrix = sample_data["adj_matrix"]
    node_ids = sample_data["node_ids"]
    time_series_dict = sample_data["time_series_dict"]

    visualization_data = {}

    # Step 1: Visualize raw time series data
    visualization_data["raw_data"] = {
        "title": "Raw Input Time Series",
        "data": time_series_dict,
        "plot": create_time_series_plot(time_series_dict, node_ids),
    }

    # Step 2: Find continuous segments
    segments_dict = {}
    for node_id, series in time_series_dict.items():
        segments = find_continuous_segments(
            series.index, series.values, gap_threshold=pd.Timedelta(minutes=15)
        )
        segments_dict[node_id] = segments

    visualization_data["segments"] = {
        "title": "Continuous Segments",
        "data": segments_dict,
        "plot": create_segments_plot(time_series_dict, segments_dict),
    }

    # Step 3: Create windows with TimeSeriesPreprocessor
    processor = TimeSeriesPreprocessor(
        window_size=window_size,
        stride=1,
        gap_threshold=pd.Timedelta(minutes=15),
        missing_value=-1.0,
    )

    X_by_sensor, masks_by_sensor, metadata_by_sensor = processor.create_windows(
        time_series_dict, standardize=True
    )

    visualization_data["windows"] = {
        "title": "Windowed Data",
        "data": {
            "X_by_sensor": X_by_sensor,
            "masks_by_sensor": masks_by_sensor,
            "metadata": metadata_by_sensor,
        },
        "plot": create_windows_plot(X_by_sensor, masks_by_sensor, node_ids),
    }

    # Step 4: Compute adjacency matrix weights
    weighted_adj = compute_adjacency_matrix(adj_matrix, sigma_squared=0.1, epsilon=0.5)

    visualization_data["adjacency"] = {
        "title": "Weighted Adjacency Matrix",
        "data": {"raw_adj": adj_matrix, "weighted_adj": weighted_adj},
        "plot": create_adjacency_plot(adj_matrix, weighted_adj, node_ids),
    }

    # Step 5: Create DataLoader
    try:
        dataloader = create_dataloader(
            X_by_sensor=X_by_sensor,
            masks_by_sensor=masks_by_sensor,
            adj_matrix=weighted_adj,
            node_ids=node_ids,
            window_size=window_size,
            horizon=horizon,
            batch_size=2,
            shuffle=False,
        )

        # Get a sample batch
        sample_batch = next(iter(dataloader))

        visualization_data["batch"] = {
            "title": "Batched Data",
            "data": sample_batch,
            "plot": create_batch_plot(sample_batch),
        }
    except Exception as e:
        print(f"Warning: Could not create dataloader visualizations: {e}")
        visualization_data["batch"] = {
            "title": "Batched Data",
            "data": None,
            "plot": go.Figure().add_annotation(
                text=f"Could not create batch visualization: {str(e)}",
                showarrow=False,
                font=dict(size=14),
            ),
        }

    # Step 6: Model Input/Output
    try:
        # Create a small model for visualization
        model = create_stgnn_model(
            input_dim=1,
            hidden_dim=16,
            output_dim=1,
            horizon=horizon,
        )

        # Show model architecture
        visualization_data["model"] = {
            "title": "Model Architecture",
            "data": model,
            "plot": create_model_plot(model),
        }
    except Exception as e:
        print(f"Warning: Could not create model visualizations: {e}")
        visualization_data["model"] = {
            "title": "Model Architecture",
            "data": None,
            "plot": go.Figure().add_annotation(
                text=f"Could not create model visualization: {str(e)}",
                showarrow=False,
                font=dict(size=14),
            ),
        }

    return visualization_data


def create_tensor_flow_dashboard(visualization_data):
    """Generate HTML dashboard with all visualizations using template_utils"""

    # Convert all plots to HTML
    html_plots = {}
    for key, data in visualization_data.items():
        html_plots[key] = pio.to_html(
            data["plot"], include_plotlyjs="cdn", full_html=False
        )

    # Load the template
    try:
        # Try to get the template from the dashboards package
        template_path = get_template_path(
            "layout.html", template_dir=Path("dashboards/tensor_flow/templates")
        )
        template = load_template(template_path)
    except Exception as e:
        # Fallback to local template if it exists
        print(f"Warning: Could not load template from dashboards package: {e}")
        template_path = Path("templates/layout.html")
        if not template_path.exists():
            print(
                f"Template file not found at {template_path}. Please ensure it exists."
            )
            return None

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

    # Create context with data for template
    context = {
        "raw_data_title": visualization_data["raw_data"]["title"],
        "segments_title": visualization_data["segments"]["title"],
        "windows_title": visualization_data["windows"]["title"],
        "adjacency_title": visualization_data["adjacency"]["title"],
        "batch_title": visualization_data["batch"]["title"],
        "model_title": visualization_data["model"]["title"],
        "raw_data_plot": html_plots["raw_data"],
        "segments_plot": html_plots["segments"],
        "windows_plot": html_plots["windows"],
        "adjacency_plot": html_plots["adjacency"],
        "batch_plot": html_plots["batch"],
        "model_plot": html_plots["model"],
    }

    # Try to use render_template from utils
    try:
        return render_template(template, context)
    except Exception as e:
        print(f"Warning: Could not use render_template: {e}")
        # Fallback to manual template replacement
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            template = template.replace(placeholder, str(value))
        return template


def main():
    """Main function to generate the dashboard"""
    # Load sample data
    file_path = os.path.join(paths.PREPROCESSED_TIMESERIES_DIR, "")
    sample_data = load_sample_data("test_data_1mnth.pkl")

    # Create visualizations
    visualization_data = visualize_tensor_transformations(sample_data)

    # Generate dashboard HTML
    dashboard_html = create_tensor_flow_dashboard(visualization_data)
    if dashboard_html is None:
        print("Failed to generate dashboard HTML")
        return

    # Ensure output directory exists
    output_dir = Path("dashboards/tensor_flow")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to file
    output_path = output_dir / "index.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print(f"Dashboard created: {output_path}")

    # Open in browser
    try:
        print(f"Opening dashboard in browser")
        webbrowser.open(f"file://{output_path.absolute()}")
    except Exception as e:
        print(f"Could not open dashboard in browser: {e}")

    return dashboard_html


if __name__ == "__main__":
    main()
