# dashboards/dataloader_explorer/__main__.py
# Main entry point for the dataloader explorer dashboard

import os
import argparse
import webbrowser
from pathlib import Path
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from .utils.data_utils import load_data_loaders
from .components.batch_explorer import create_batch_explorer
from .components.node_explorer import create_node_explorer
from .components.window_explorer import create_window_explorer
from .components.correlation_plot import create_correlation_plot
from .components.adjacency_plot import create_adjacency_plot
from .components.data_stats import create_stats_panel


# Define the layout for the Dash app
def create_layout(data_loaders, available_loaders):
    """Create the dashboard layout"""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.H1(
                            "DataLoader Explorer Dashboard",
                            className="text-center my-4",
                        ),
                        width=12,
                    )
                ]
            ),
            # Controls and settings section
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Settings"),
                                    dbc.CardBody(
                                        [
                                            html.Label("DataLoader:"),
                                            dcc.Dropdown(
                                                id="loader-selector",
                                                options=[
                                                    {"label": loader, "value": loader}
                                                    for loader in available_loaders
                                                ],
                                                value=(
                                                    available_loaders[0]
                                                    if available_loaders
                                                    else None
                                                ),
                                                className="mb-3",
                                            ),
                                            html.Label("Batch Index:"),
                                            dcc.Slider(
                                                id="batch-slider",
                                                min=0,
                                                max=10,  # Will be updated dynamically
                                                step=1,
                                                value=0,
                                                marks=None,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                                className="mb-3",
                                            ),
                                            html.Label("Node Index:"),
                                            dcc.Slider(
                                                id="node-slider",
                                                min=0,
                                                max=10,  # Will be updated dynamically
                                                step=1,
                                                value=0,
                                                marks=None,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                                className="mb-3",
                                            ),
                                            html.Label("Number of Windows:"),
                                            dcc.Slider(
                                                id="window-slider",
                                                min=5,
                                                max=50,
                                                step=5,
                                                value=20,
                                                marks={
                                                    i: str(i) for i in range(5, 51, 5)
                                                },
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
                                                className="mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4",
                            )
                        ],
                        width=12,
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Dataset Statistics"),
                                    dbc.CardBody([html.Div(id="dataset-stats")]),
                                ],
                                className="mb-4",
                            )
                        ],
                        width=12,
                        md=8,
                    ),
                ]
            ),
            # Tabs for different views
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-batch-explorer",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="batch-explorer-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Batch Explorer",
                        tab_id="tab-batch",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-node-explorer",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="node-explorer-fig",
                                                style={"height": "600px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Node Explorer",
                        tab_id="tab-node",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-window-explorer",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="window-explorer-fig",
                                                style={"height": "700px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Window Explorer",
                        tab_id="tab-window",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-correlation-plot",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="correlation-plot-fig",
                                                style={"height": "700px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Correlation Analysis",
                        tab_id="tab-correlation",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-adjacency-plot",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="adjacency-plot-fig",
                                                style={"height": "700px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Adjacency Matrix",
                        tab_id="tab-adjacency",
                    ),
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-stats-panel",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="stats-panel-fig",
                                                style={"height": "700px"},
                                            ),
                                        ),
                                        width=12,
                                    )
                                ],
                                className="mb-4",
                            ),
                        ],
                        label="Statistics",
                        tab_id="tab-stats",
                    ),
                ],
                id="tabs",
                active_tab="tab-batch",
            ),
            # Footer
            dbc.Row(
                [
                    dbc.Col(
                        html.Footer(
                            [
                                html.P(
                                    [
                                        "DataLoader Explorer Dashboard | Created with Dash and Plotly | ",
                                        html.A(
                                            "GNN Package", href="#", target="_blank"
                                        ),
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
            dcc.Store(id="data-store"),
        ],
        fluid=True,
        className="px-4 py-3",
    )


def create_dash_app(data_loaders_path):
    """Create and configure the Dash app"""
    # Load data loaders
    try:
        data_loaders = load_data_loaders(data_loaders_path)
        # Find available loaders
        available_loaders = [
            key for key in data_loaders.keys() if key.endswith("_loader")
        ]

        if not available_loaders:
            raise ValueError("No valid data loaders found in the provided file.")

    except Exception as e:
        print(f"Error loading data loaders: {e}")
        # Create mock data for UI testing
        data_loaders = {}
        available_loaders = []

    # Create the Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="DataLoader Explorer",
    )

    # Define the app layout
    app.layout = create_layout(data_loaders, available_loaders)

    # Define callbacks
    @app.callback(
        [
            Output("batch-slider", "max"),
            Output("node-slider", "max"),
            Output("data-store", "data"),
        ],
        [Input("loader-selector", "value")],
    )
    def update_sliders(loader_key):
        """Update the sliders based on the selected data loader"""
        if not loader_key or loader_key not in data_loaders:
            return 0, 0, {"batch_max": 0, "node_max": 0}

        try:
            # Get a batch to extract dimensions
            batch = next(iter(data_loaders[loader_key]))
            batch_size = batch["x"].shape[0]
            num_nodes = batch["x"].shape[1]

            return (
                batch_size - 1,
                num_nodes - 1,
                {"batch_max": batch_size - 1, "node_max": num_nodes - 1},
            )

        except Exception as e:
            print(f"Error updating sliders: {e}")
            return 0, 0, {"batch_max": 0, "node_max": 0}

    @app.callback(
        Output("batch-explorer-fig", "figure"),
        [Input("loader-selector", "value"), Input("batch-slider", "value")],
    )
    def update_batch_explorer(loader_key, batch_idx):
        """Update the batch explorer visualization"""
        if not loader_key or loader_key not in data_loaders:
            # Return empty figure
            fig = dash.no_update
            return fig

        try:
            return create_batch_explorer(data_loaders, loader_key, batch_idx)
        except Exception as e:
            print(f"Error updating batch explorer: {e}")
            # Return empty figure with error message
            fig = dash.no_update
            return fig

    @app.callback(
        Output("node-explorer-fig", "figure"),
        [
            Input("loader-selector", "value"),
            Input("batch-slider", "value"),
            Input("node-slider", "value"),
        ],
    )
    def update_node_explorer(loader_key, batch_idx, node_idx):
        """Update the node explorer visualization"""
        if not loader_key or loader_key not in data_loaders:
            return dash.no_update

        try:
            return create_node_explorer(data_loaders, loader_key, batch_idx, node_idx)
        except Exception as e:
            print(f"Error updating node explorer: {e}")
            return dash.no_update

    @app.callback(
        Output("window-explorer-fig", "figure"),
        [
            Input("loader-selector", "value"),
            Input("node-slider", "value"),
            Input("window-slider", "value"),
        ],
    )
    def update_window_explorer(loader_key, node_idx, num_windows):
        """Update the window explorer visualization"""
        if not loader_key or loader_key not in data_loaders:
            return dash.no_update

        try:
            return create_window_explorer(
                data_loaders, loader_key, num_windows, node_idx
            )
        except Exception as e:
            print(f"Error updating window explorer: {e}")
            return dash.no_update

    @app.callback(
        Output("correlation-plot-fig", "figure"),
        [Input("loader-selector", "value"), Input("batch-slider", "value")],
    )
    def update_correlation_plot(loader_key, batch_idx):
        """Update the correlation plot"""
        if not loader_key or loader_key not in data_loaders:
            return dash.no_update

        try:
            return create_correlation_plot(data_loaders, loader_key, batch_idx)
        except Exception as e:
            print(f"Error updating correlation plot: {e}")
            return dash.no_update

    @app.callback(
        Output("adjacency-plot-fig", "figure"), [Input("loader-selector", "value")]
    )
    def update_adjacency_plot(loader_key):
        """Update the adjacency matrix visualization"""
        if not loader_key or loader_key not in data_loaders:
            return dash.no_update

        try:
            return create_adjacency_plot(data_loaders, loader_key)
        except Exception as e:
            print(f"Error updating adjacency plot: {e}")
            return dash.no_update

    @app.callback(
        Output("stats-panel-fig", "figure"), [Input("loader-selector", "value")]
    )
    def update_stats_panel(loader_key):
        """Update the statistics panel"""
        if not loader_key or loader_key not in data_loaders:
            return dash.no_update

        try:
            return create_stats_panel(data_loaders, loader_key)
        except Exception as e:
            print(f"Error updating stats panel: {e}")
            return dash.no_update

    @app.callback(
        Output("dataset-stats", "children"), [Input("loader-selector", "value")]
    )
    def update_dataset_stats(loader_key):
        """Update the dataset statistics summary"""
        if not loader_key or loader_key not in data_loaders:
            return html.Div("No data loader selected")

        try:
            # Create a simple HTML table with key stats
            batch = next(iter(data_loaders[loader_key]))
            batch_size, num_nodes, seq_len, feature_dim = batch["x"].shape
            _, _, horizon, _ = batch["y"].shape

            stats_table = html.Table(
                [
                    html.Tr([html.Th("Statistic"), html.Th("Value")]),
                    html.Tr([html.Td("Batch Size"), html.Td(str(batch_size))]),
                    html.Tr([html.Td("Number of Nodes"), html.Td(str(num_nodes))]),
                    html.Tr([html.Td("Sequence Length"), html.Td(str(seq_len))]),
                    html.Tr([html.Td("Prediction Horizon"), html.Td(str(horizon))]),
                    html.Tr([html.Td("Feature Dimension"), html.Td(str(feature_dim))]),
                ],
                className="table table-striped table-sm",
            )

            return stats_table

        except Exception as e:
            print(f"Error updating dataset stats: {e}")
            return html.Div(f"Error: {str(e)}")

    return app, data_loaders


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DataLoader Explorer Dashboard")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to the pickle file containing data loaders",
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

    # Create the Dash app
    app, data_loaders = create_dash_app(args.data)

    # Print data loader information
    available_loaders = [key for key in data_loaders.keys() if key.endswith("_loader")]
    print(f"Available data loaders: {available_loaders}")

    # Run the app
    print(f"\nStarting DataLoader Explorer Dashboard on port {args.port}")
    print(f"Open your browser and navigate to http://localhost:{args.port}/")

    # Open browser automatically
    webbrowser.open_new(f"http://localhost:{args.port}/")

    # Run server
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
