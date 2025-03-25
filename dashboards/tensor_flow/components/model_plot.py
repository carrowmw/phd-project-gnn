import numpy as np
import plotly.graph_objects as go


def create_model_plot(model):
    """Create a more robust visualization of the model architecture"""
    if model is None:
        return go.Figure().add_annotation(
            text="No model available", showarrow=False, font=dict(size=14)
        )

    # More robust parameter extraction with error handling
    try:
        # Extract model parameters safely
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Try multiple approaches to get the input and hidden dimensions
        try:
            # First attempt - direct STGNN structure
            input_dim = model.encoder.gc1.in_features
            hidden_dim = model.encoder.gc1.out_features
        except AttributeError:
            try:
                # Second attempt - nested encoder structure
                input_dim = model.encoder.encoder.gc1.in_features
                hidden_dim = model.encoder.encoder.gc1.out_features
            except AttributeError:
                # Fallback - just use some reasonable defaults and warn user
                input_dim = getattr(model, "input_dim", 1)
                hidden_dim = getattr(model, "hidden_dim", 64)
                print(
                    "Warning: Could not extract exact model dimensions, using defaults"
                )

        horizon = getattr(model, "horizon", 6)  # Default to 6 if not found
    except Exception as e:
        # If extraction fails, use defaults and show a warning
        print(f"Warning: Error extracting model parameters: {e}")
        input_dim = 1
        hidden_dim = 64
        horizon = 6
        model_params = 0

    # Create a more informative visualization of the architecture
    layers = [
        {
            "name": "Input",
            "size": input_dim,
            "color": "rgba(173, 216, 230, 0.7)",
            "description": "Raw sensor inputs",
        },
        {
            "name": "GCN-1",
            "size": hidden_dim,
            "color": "rgba(144, 238, 144, 0.7)",
            "description": "Graph convolution",
        },
        {
            "name": "GCN-2",
            "size": hidden_dim,
            "color": "rgba(144, 238, 144, 0.7)",
            "description": "Graph convolution",
        },
        {
            "name": "GRU",
            "size": hidden_dim,
            "color": "rgba(255, 255, 153, 0.7)",
            "description": "Temporal processing",
        },
        {
            "name": "Decoder",
            "size": hidden_dim,
            "color": "rgba(255, 182, 193, 0.7)",
            "description": "Forecast decoder",
        },
        {
            "name": "Output",
            "size": horizon,
            "color": "rgba(173, 216, 230, 0.7)",
            "description": f"{horizon}-step forecast",
        },
    ]

    # Create node positions with better spacing
    y_positions = []
    x_positions = [1, 2, 3, 4.5, 6, 7]  # Custom x positions to space out layers nicely

    for layer in layers:
        size = min(
            max(layer["size"], 4), 12
        )  # Limit size range for better visualization
        y_positions.append(np.linspace(-size / 2, size / 2, size))

    fig = go.Figure()

    # Add nodes for each layer with improved styling
    for i, (layer, x_pos) in enumerate(zip(layers, x_positions)):
        size = min(max(layer["size"], 4), 12)
        for j, y in enumerate(y_positions[i]):
            # Only draw every other node when there are many nodes to avoid overcrowding
            if size > 8 and j % 2 != 0 and j != 0 and j != size - 1:
                continue

            show_in_legend = j == 0

            # Add hover information
            hover_text = f"{layer['name']}: {layer['description']}"
            if show_in_legend:
                hover_text += f"<br>Size: {layer['size']}"

            fig.add_trace(
                go.Scatter(
                    x=[x_pos],
                    y=[y],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color=layer["color"],
                        line=dict(width=1, color="darkgray"),
                    ),
                    name=layer["name"],
                    legendgroup=layer["name"],
                    showlegend=show_in_legend,
                    hoverinfo="text",
                    hovertext=hover_text,
                )
            )

        # Add a text label for each layer
        fig.add_annotation(
            x=x_pos,
            y=max(y_positions[i]) + 0.5,
            text=layer["name"],
            showarrow=False,
            font=dict(size=10),
        )

    # Add edges between layers with improved styling
    for i in range(len(layers) - 1):
        x1, x2 = x_positions[i], x_positions[i + 1]
        for j, y1 in enumerate(y_positions[i]):
            # Skip some nodes for larger layers to avoid overcrowding
            if (
                len(y_positions[i]) > 8
                and j % 2 != 0
                and j != 0
                and j != len(y_positions[i]) - 1
            ):
                continue

            for k, y2 in enumerate(y_positions[i + 1]):
                # Skip some nodes for larger layers to avoid overcrowding
                if (
                    len(y_positions[i + 1]) > 8
                    and k % 2 != 0
                    and k != 0
                    and k != len(y_positions[i + 1]) - 1
                ):
                    continue

                # Draw connection with variable opacity based on source/target position
                # Creates a more visually appealing flow effect
                opacity = 0.15
                if (
                    j == 0
                    or j == len(y_positions[i]) - 1
                    or k == 0
                    or k == len(y_positions[i + 1]) - 1
                ):
                    opacity = 0.3  # Stronger lines for outer connections

                fig.add_trace(
                    go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode="lines",
                        line=dict(width=0.5, color=f"rgba(100, 100, 100, {opacity})"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

    # Add detailed model summary annotations
    details = []

    # Add basic model structure information
    details.append(f"STGNN Model Architecture")
    details.append(f"Parameters: {model_params:,}")
    details.append(f"Input: {input_dim}, Hidden: {hidden_dim}, Horizon: {horizon}")

    # Add explanation of the model components
    details.append("")
    details.append("Component Functions:")
    details.append("• GCNs capture spatial relationships between sensors")
    details.append("• GRU captures temporal patterns in the data")
    details.append("• Decoder projects to multi-step forecasts")

    # Create text box with model details
    fig.add_annotation(
        text="<br>".join(details),
        x=0.5,
        y=-0.2,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        align="center",
    )

    # Add a model flow diagram
    flow_text = "Input → GCN → GRU → Decoder → Output"
    fig.add_annotation(
        text=flow_text,
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
    )

    # Update layout with improved styling
    fig.update_layout(
        title="STGNN Model Architecture",
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, 8],  # Explicit range to control the width
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",  # Square aspect ratio
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        height=500,
        margin=dict(l=20, r=20, t=70, b=120),
        plot_bgcolor="rgba(240, 240, 240, 0.5)",  # Light gray background
    )

    return fig
