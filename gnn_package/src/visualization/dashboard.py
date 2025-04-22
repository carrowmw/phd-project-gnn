# gnn_package/src/visualization/dashboard.py

from pathlib import Path
import os
import pandas as pd
from datetime import datetime
import base64
import io
from typing import List, Dict, Any, Optional, Union


def encode_image_to_base64(image_path):
    """Convert an image to a base64 string for embedding in HTML"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def generate_dashboard(
    results_path: Union[str, Path],
    prediction_results: Dict[str, Any],
    include_images: bool = True,
) -> str:
    """
    Generate an HTML dashboard for prediction results.

    Parameters:
    -----------
    results_path : str or Path
        Directory containing prediction results
    prediction_results : dict
        Dictionary with prediction information
    include_images : bool
        Whether to embed images in the HTML

    Returns:
    --------
    str
        HTML content of the dashboard
    """
    results_path = Path(results_path)

    # Get data from prediction results
    metrics = prediction_results.get("metrics", {})
    viz_paths = prediction_results.get("visualizations", {})
    predictions_file = prediction_results.get("predictions_file")
    summary_file = prediction_results.get("summary_file")

    # Load summary text if available
    summary_text = ""
    if summary_file and os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            summary_text = f.read()

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Traffic Prediction Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            header {{
                background-color: #4472C4;
                color: white;
                padding: 10px 20px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            h1, h2, h3 {{
                color: #4472C4;
            }}
            .summary-box {{
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                white-space: pre-wrap;
                font-family: monospace;
            }}
            .metrics {{
                display: flex;
                justify-content: space-around;
                margin-bottom: 20px;
            }}
            .metric-card {{
                background-color: #f2f2f2;
                border-radius: 5px;
                padding: 15px;
                text-align: center;
                width: 150px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #4472C4;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
            .viz-container {{
                margin-bottom: 30px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            footer {{
                margin-top: 30px;
                border-top: 1px solid #ddd;
                padding-top: 10px;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Traffic Prediction Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </header>

            <section>
                <h2>Summary</h2>
                <div class="summary-box">
                    {summary_text if summary_text else "No summary available"}
                </div>

                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('mse', 'N/A'):.4f}</div>
                        <div class="metric-label">MSE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('mae', 'N/A'):.4f}</div>
                        <div class="metric-label">MAE</div>
                    </div>
                </div>
            </section>
    """

    # Add visualization sections if available
    if viz_paths and include_images:
        html_content += """
            <section>
                <h2>Visualizations</h2>
        """

        if "grid_plot" in viz_paths and os.path.exists(viz_paths["grid_plot"]):
            encoded_img = encode_image_to_base64(viz_paths["grid_plot"])
            html_content += f"""
                <div class="viz-container">
                    <h3>Sensor Predictions Grid</h3>
                    <img src="data:image/png;base64,{encoded_img}" alt="Sensor Grid Plot">
                </div>
            """

        if "error_analysis" in viz_paths and os.path.exists(
            viz_paths["error_analysis"]
        ):
            encoded_img = encode_image_to_base64(viz_paths["error_analysis"])
            html_content += f"""
                <div class="viz-container">
                    <h3>Error Analysis</h3>
                    <img src="data:image/png;base64,{encoded_img}" alt="Error Analysis">
                </div>
            """

        if "validation_plot" in viz_paths and os.path.exists(
            viz_paths["validation_plot"]
        ):
            encoded_img = encode_image_to_base64(viz_paths["validation_plot"])
            html_content += f"""
                <div class="viz-container">
                    <h3>Validation Plot</h3>
                    <img src="data:image/png;base64,{encoded_img}" alt="Validation Plot">
                </div>
            """

        html_content += """
            </section>
        """

    # Close HTML
    html_content += f"""
            <footer>
                <p>GNN Traffic Prediction Model | Model path: {prediction_results.get('model_path', 'Unknown')}</p>
                <p>Output files directory: {results_path}</p>
            </footer>
        </div>
    </body>
    </html>
    """

    return html_content


def save_dashboard(
    results_path: Union[str, Path],
    prediction_results: Dict[str, Any],
    output_file: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate and save an HTML dashboard for prediction results.

    Parameters:
    -----------
    results_path : str or Path
        Directory containing prediction results
    prediction_results : dict
        Dictionary with prediction information
    output_file : str or Path, optional
        File path to save the dashboard (defaults to dashboard.html in results_path)

    Returns:
    --------
    str
        Path to the saved dashboard
    """
    results_path = Path(results_path)

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_path / f"dashboard_{timestamp}.html"
    else:
        output_file = Path(output_file)

    # Generate dashboard HTML
    html_content = generate_dashboard(results_path, prediction_results)

    # Save to file
    with open(output_file, "w") as f:
        f.write(html_content)

    return str(output_file)
