# Create a multi-page dashboard using HTML and Plotly
from pathlib import Path
from plotly.io import to_html
import plotly.io as pio

from .components import (
    interactive_data_availability,
    interactive_sensor_windows,
    interactive_window_counts,
    visualize_daily_patterns,
    visualize_window_pca,
)
from .utils import load_data, load_template, render_template, get_template_path


# Create a comprehensive dashboard combining multiple visualizations
def create_sensor_window_dashboard(data_file="test_data_1yr.pkl", window_size=24):
    """Create a comprehensive dashboard for analyzing sensor time windows"""
    # Load the data
    time_series_dict = load_data(data_file)

    # Identify sensors with most data points for individual analysis
    sensor_data_counts = {
        sensor_id: len(series) for sensor_id, series in time_series_dict.items()
    }
    top_sensor = max(sensor_data_counts, key=sensor_data_counts.get)

    # Set theme
    pio.templates.default = "plotly_white"

    # Create individual visualizations
    data_avail_fig = interactive_data_availability(time_series_dict)
    top_sensor_fig = interactive_sensor_windows(
        time_series_dict, top_sensor, window_size
    )
    window_counts_fig = interactive_window_counts(time_series_dict, window_size)
    daily_patterns_fig = visualize_daily_patterns(time_series_dict, n_sensors=4)

    # Try to create PCA visualization if possible
    try:
        pca_fig = visualize_window_pca(time_series_dict, window_size)
    except Exception as e:
        print(f"Could not create PCA visualization: {e}")
        pca_fig = None

    # Load the template
    template_path = get_template_path("dashboard_template.html")
    template = load_template(template_path)

    # Prepare context with all variables for the template
    context = {
        "window_size": window_size,
        "top_sensor": top_sensor,
        "data_avail_fig": to_html(
            data_avail_fig, include_plotlyjs="cdn", full_html=False
        ),
        "top_sensor_fig": to_html(
            top_sensor_fig, include_plotlyjs="cdn", full_html=False
        ),
        "window_counts_fig": to_html(
            window_counts_fig, include_plotlyjs="cdn", full_html=False
        ),
        "daily_patterns_fig": to_html(
            daily_patterns_fig, include_plotlyjs="cdn", full_html=False
        ),
        "pca_fig": (
            to_html(pca_fig, include_plotlyjs="cdn", full_html=False)
            if pca_fig
            else None
        ),
    }

    # Render the template
    html_content = render_template(template, context)

    # Return the HTML content
    return html_content


if __name__ == "__main__":
    # Create the dashboard
    dashboard_html = create_sensor_window_dashboard(
        data_file="dashboards/data/test_data_1yr.pkl", window_size=24
    )

    # Save to a file
    output_path = Path(__file__).parent / "sensor_dashboard.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print(f"Dashboard created: {output_path}")
