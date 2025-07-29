# dashboards/gnn_diagnostics/__main__.py

import os
import argparse
import webbrowser
from pathlib import Path
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils.data_utils import load_experiment_data, extract_metrics
from .components.model_outputs import create_prediction_comparison
from .components.data_explorer import create_raw_data_explorer
from .components.loss_analyzer import create_loss_curve_visualization
from .components.graph_explorer import create_graph_visualization
from .components.feature_distribution import create_feature_distribution
from .components.adjacency_analyzer import create_adjacency_analysis
from .components.missing_data_analyzer import create_missing_data_analysis


def create_layout(experiment_dirs, available_experiments):
    """Create the dashboard layout"""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.H1(
                            "GNN Pipeline Diagnostic Dashboard",
                            className="text-center my-4",
                        ),
                        width=12,
                    )
                ]
            ),
            # Controls section
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Experiment Selection"),
                                    dbc.CardBody(
                                        [
                                            html.Label("Experiment:"),
                                            dcc.Dropdown(
                                                id="experiment-selector",
                                                options=[
                                                    {"label": exp, "value": exp}
                                                    for exp in available_experiments
                                                ],
                                                value=(
                                                    available_experiments[0]
                                                    if available_experiments
                                                    else None
                                                ),
                                                className="mb-3",
                                            ),
                                            html.Label("Comparison Experiment (optional):"),
                                            dcc.Dropdown(
                                                id="comparison-selector",
                                                options=[
                                                    {"label": "None", "value": "none"}
                                                ] + [
                                                    {"label": exp, "value": exp}
                                                    for exp in available_experiments
                                                ],
                                                value="none",
                                                className="mb-3",
                                            ),
                                            html.Label("Sensor ID:"),
                                            dcc.Dropdown(
                                                id="sensor-selector",
                                                options=[],  # Will be populated dynamically
                                                className="mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            )
                        ],
                        width=12,
                        lg=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Experiment Overview"),
                                    dbc.CardBody([html.Div(id="experiment-overview")]),
                                ],
                                className="mb-4 h-100",
                            )
                        ],
                        width=12,
                        lg=9,
                    ),
                ]
            ),
            # Tabs for different analyses
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-prediction-comparison",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="prediction-comparison-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Prediction Analysis",
                        tab_id="tab-prediction",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-loss-curves",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="loss-curves-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Training Loss Analysis",
                        tab_id="tab-loss",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-raw-data",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="raw-data-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Raw Data Explorer",
                        tab_id="tab-raw-data",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-feature-distribution",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="feature-distribution-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Feature Distribution",
                        tab_id="tab-feature",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-graph-explorer",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="graph-explorer-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Graph Structure",
                        tab_id="tab-graph",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-adjacency-analyzer",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="adjacency-analyzer-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Adjacency Analysis",
                        tab_id="tab-adjacency",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-missing-data",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="missing-data-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Missing Data Analysis",
                        tab_id="tab-missing",
                    ),
                ],
                id="tabs",
                active_tab="tab-prediction",
            ),
            # Footer
            dbc.Row(
                [
                    dbc.Col(
                        html.Footer(
                            [
                                html.P(
                                    [
                                        "GNN Pipeline Diagnostic Dashboard | Created with Dash and Plotly",
                                    ],
                                    className="text-center text-muted",
                                )
                            ],
                            className="py-3",
                        ),
                        width=12,
                    )
                ],
                className="mt-5",
            ),
            # Store for sharing data between callbacks
            dcc.Store(id="experiment-data-store"),
            dcc.Store(id="comparison-data-store"),
        ],
        fluid=True,
        className="px-4 py-3",
    )


def create_dash_app(experiment_dirs):
    """Create and configure the Dash app"""
    # Extract available experiments
    available_experiments = [Path(exp_dir).name for exp_dir in experiment_dirs if os.path.isdir(exp_dir)]

    if not available_experiments:
        print("No experiment directories found!")

    # Create the Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="GNN Pipeline Diagnostics",
    )

    # Define the app layout
    app.layout = create_layout(experiment_dirs, available_experiments)

    # Define callbacks
    @app.callback(
        [
            Output("experiment-data-store", "data"),
            Output("sensor-selector", "options"),
            Output("sensor-selector", "value"),
        ],
        [Input("experiment-selector", "value")],
    )
    def load_experiment_data_callback(experiment_name):
        """Load data for the selected experiment"""
        if not experiment_name:
            return None, [], None

        try:
            # Find the directory for this experiment
            exp_dir = None
            for dir_path in experiment_dirs:
                if Path(dir_path).name == experiment_name:
                    exp_dir = dir_path
                    break

            if not exp_dir:
                return None, [], None

            # Load experiment data
            experiment_data = load_experiment_data(exp_dir)

            # Extract available sensors
            predictions_df = experiment_data.get("predictions_df", pd.DataFrame())
            if not predictions_df.empty and "node_id" in predictions_df.columns:
                sensor_options = []
                unique_sensors = predictions_df["node_id"].unique()
                for sensor in unique_sensors:
                    sensor_name = predictions_df[predictions_df["node_id"] == sensor]["sensor_name"].iloc[0] if "sensor_name" in predictions_df.columns else f"Sensor {sensor}"
                    sensor_options.append({"label": f"{sensor_name} ({sensor})", "value": sensor})

                return experiment_data, sensor_options, unique_sensors[0] if len(unique_sensors) > 0 else None

            return experiment_data, [], None

        except Exception as e:
            print(f"Error loading experiment data: {e}")
            return None, [], None

    @app.callback(
        Output("comparison-data-store", "data"),
        [Input("comparison-selector", "value")],
    )
    def load_comparison_data_callback(experiment_name):
        """Load data for the comparison experiment"""
        if not experiment_name or experiment_name == "none":
            return None

        try:
            # Find the directory for this experiment
            exp_dir = None
            for dir_path in experiment_dirs:
                if Path(dir_path).name == experiment_name:
                    exp_dir = dir_path
                    break

            if not exp_dir:
                return None

            # Load experiment data
            return load_experiment_data(exp_dir)

        except Exception as e:
            print(f"Error loading comparison data: {e}")
            return None

    @app.callback(
        Output("experiment-overview", "children"),
        [
            Input("experiment-data-store", "data"),
            Input("comparison-data-store", "data"),
        ],
    )
    def update_experiment_overview(experiment_data, comparison_data):
        """Update the experiment overview panel"""
        if not experiment_data:
            return html.Div("No experiment selected")

        try:
            # Extract metrics
            metrics = extract_metrics(experiment_data)

            # Create experiment info table
            exp_info = []
            config = experiment_data.get("config", {})

            # Extract key configuration settings
            exp_info.append(html.H5("Configuration"))

            model_info = []
            if "model" in config:
                model = config.get("model", {})
                model_info.extend([
                    html.Li(f"Architecture: {model.get('architecture', 'N/A')}"),
                    html.Li(f"Hidden Dim: {model.get('hidden_dim', 'N/A')}"),
                    html.Li(f"Layers: {model.get('num_layers', 'N/A')}"),
                    html.Li(f"Use GRU: {model.get('use_gru', 'N/A')}"),
                    html.Li(f"Use Temporal Attention: {model.get('use_temporal_attention', 'N/A')}"),
                ])

            training_info = []
            if "training" in config:
                training = config.get("training", {})
                training_info.extend([
                    html.Li(f"Learning Rate: {training.get('learning_rate', 'N/A')}"),
                    html.Li(f"Epochs: {training.get('num_epochs', 'N/A')}"),
                    html.Li(f"Weight Decay: {training.get('weight_decay', 'N/A')}"),
                ])

            data_info = []
            if "data" in config and "general" in config["data"]:
                data = config["data"]["general"]
                data_info.extend([
                    html.Li(f"Window Size: {data.get('window_size', 'N/A')}"),
                    html.Li(f"Horizon: {data.get('horizon', 'N/A')}"),
                    html.Li(f"Standardize: {data.get('standardize', 'N/A')}"),
                ])

            # Create config section
            exp_info.extend([
                html.H6("Model"),
                html.Ul(model_info, className="mb-2"),
                html.H6("Training"),
                html.Ul(training_info, className="mb-2"),
                html.H6("Data"),
                html.Ul(data_info, className="mb-2"),
            ])

            # Add metrics section
            metrics_section = []
            if metrics:
                metrics_section.extend([
                    html.H5("Performance Metrics"),
                    html.Table(
                        [
                            html.Tr([html.Th("Metric"), html.Th("Value")]),
                            html.Tr([html.Td("MSE"), html.Td(f"{metrics.get('mse', 'N/A'):.6f}")]),
                            html.Tr([html.Td("MAE"), html.Td(f"{metrics.get('mae', 'N/A'):.6f}")]),
                            html.Tr([html.Td("RMSE"), html.Td(f"{metrics.get('rmse', 'N/A'):.6f}")]),
                            html.Tr([html.Td("Valid Points"), html.Td(f"{metrics.get('valid_points', 'N/A')}")]),
                            html.Tr([html.Td("Total Points"), html.Td(f"{metrics.get('total_points', 'N/A')}")]),
                        ],
                        className="table table-sm table-striped",
                    ),
                ])

                # Add comparison metrics if available
                if comparison_data:
                    comparison_metrics = extract_metrics(comparison_data)
                    if comparison_metrics:
                        metrics_section.extend([
                            html.H6("Comparison Metrics"),
                            html.Table(
                                [
                                    html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("Difference")]),
                                    html.Tr([
                                        html.Td("MSE"),
                                        html.Td(f"{comparison_metrics.get('mse', 'N/A'):.6f}"),
                                        html.Td(f"{metrics.get('mse', 0) - comparison_metrics.get('mse', 0):.6f}")
                                    ]),
                                    html.Tr([
                                        html.Td("MAE"),
                                        html.Td(f"{comparison_metrics.get('mae', 'N/A'):.6f}"),
                                        html.Td(f"{metrics.get('mae', 0) - comparison_metrics.get('mae', 0):.6f}")
                                    ]),
                                    html.Tr([
                                        html.Td("RMSE"),
                                        html.Td(f"{comparison_metrics.get('rmse', 'N/A'):.6f}"),
                                        html.Td(f"{metrics.get('rmse', 0) - comparison_metrics.get('rmse', 0):.6f}")
                                    ]),
                                ],
                                className="table table-sm table-striped",
                            ),
                        ])

            return html.Div(exp_info + metrics_section)

        except Exception as e:
            print(f"Error updating experiment overview: {e}")
            return html.Div(f"Error: {str(e)}")

    @app.callback(
        Output("prediction-comparison-fig", "figure"),
        [
            Input("experiment-data-store", "data"),
            Input("comparison-data-store", "data"),
            Input("sensor-selector", "value"),
        ],
    )
    def update_prediction_comparison(experiment_data, comparison_data, sensor_id):
        """Update prediction comparison visualization"""
        if not experiment_data or not sensor_id:
            return go.Figure()

        try:
            return create_prediction_comparison(experiment_data, comparison_data, sensor_id)
        except Exception as e:
            print(f"Error updating prediction comparison: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating prediction comparison: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

    @app.callback(
        Output("loss-curves-fig", "figure"),
        [Input("experiment-data-store", "data"), Input("comparison-data-store", "data")],
    )
    def update_loss_curves(experiment_data, comparison_data):
        """Update loss curves visualization"""
        if not experiment_data:
            return go.Figure()

        try:
            return create_loss_curve_visualization(experiment_data, comparison_data)
        except Exception as e:
            print(f"Error updating loss curves: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating loss curves: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

    @app.callback(
        Output("raw-data-fig", "figure"),
        [Input("experiment-data-store", "data"), Input("sensor-selector", "value")],
    )
    def update_raw_data_explorer(experiment_data, sensor_id):
        """Update raw data explorer visualization"""
        if not experiment_data or not sensor_id:
            return go.Figure()

        try:
            return create_raw_data_explorer(experiment_data, sensor_id)
        except Exception as e:
            print(f"Error updating raw data explorer: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating raw data explorer: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

    @app.callback(
        Output("feature-distribution-fig", "figure"),
        [Input("experiment-data-store", "data"), Input("comparison-data-store", "data")],
    )
    def update_feature_distribution(experiment_data, comparison_data):
        """Update feature distribution visualization"""
        if not experiment_data:
            return go.Figure()

        try:
            return create_feature_distribution(experiment_data, comparison_data)
        except Exception as e:
            print(f"Error updating feature distribution: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating feature distribution: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

    @app.callback(
        Output("graph-explorer-fig", "figure"),
        [Input("experiment-data-store", "data")],
    )
    def update_graph_explorer(experiment_data):
        """Update graph explorer visualization"""
        if not experiment_data:
            return go.Figure()

        try:
            return create_graph_visualization(experiment_data)
        except Exception as e:
            print(f"Error updating graph explorer: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating graph visualization: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

    @app.callback(
        Output("adjacency-analyzer-fig", "figure"),
        [Input("experiment-data-store", "data")],
    )
    def update_adjacency_analyzer(experiment_data):
        """Update adjacency matrix analysis visualization"""
        if not experiment_data:
            return go.Figure()

        try:
            return create_adjacency_analysis(experiment_data)
        except Exception as e:
            print(f"Error updating adjacency analysis: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating adjacency analysis: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

    @app.callback(
        Output("missing-data-fig", "figure"),
        [Input("experiment-data-store", "data"), Input("sensor-selector", "value")],
    )
    def update_missing_data_analyzer(experiment_data, sensor_id):
        """Update missing data analysis visualization"""
        if not experiment_data or not sensor_id:
            return go.Figure()

        try:
            return create_missing_data_analysis(experiment_data, sensor_id)
        except Exception as e:
            print(f"Error updating missing data analysis: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating missing data analysis: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

    return app


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GNN Pipeline Diagnostic Dashboard")
    parser.add_argument(
        "--results",
        "-r",
        type=str,
        default="results",
        help="Path to the results directory containing experiment folders",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8050,
        help="Port to run the dashboard server on",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the app in debug mode"
    )

    args = parser.parse_args()

    # Find experiment directories
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return

    # Get subdirectories - these are the experiment results
    experiment_dirs = [str(path) for path in results_path.iterdir() if path.is_dir()]

    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in experiment_dirs:
        print(f"  - {Path(exp_dir).name}")

    # Create the Dash app
    app = create_dash_app(experiment_dirs)

    # Run the app
    print(f"\nStarting GNN Pipeline Diagnostic Dashboard on port {args.port}")
    print(f"Open your browser and navigate to http://localhost:{args.port}/")

    # Open browser automatically
    webbrowser.open_new(f"http://localhost:{args.port}/")

    # Run server
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()