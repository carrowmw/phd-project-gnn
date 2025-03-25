# Create a multi-page dashboard using HTML and Plotly
import os
from pathlib import Path
from datetime import datetime
from plotly.io import to_html
import plotly.io as pio

from .components import (
    interactive_data_availability,
    create_calendar_heatmap,
    create_completeness_trend,
    create_sensors_comparison,
    create_sensor_clustering,
    interactive_sensor_windows,
    create_monthly_coverage_matrix,
    visualize_daily_patterns,
    create_sensors_map,
    create_time_of_day_profiles,
    interactive_window_counts,
)

from .utils import load_data, load_template, render_template, get_template_path

from gnn_package.config import ExperimentConfig
from gnn_package import paths


def compute_completeness(time_series_dict):
    """
    Compute data completeness for each sensor.

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping sensor IDs to time series data

    Returns:
    --------
    dict
        Dictionary mapping sensor IDs to completeness percentage (0-1)
    """
    try:
        config = ExperimentConfig(
            "/Users/administrator/Code/python/phd-project-gnn/config.yml"
        )
        start_date = datetime.strptime(config.data.start_date, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(config.data.end_date, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Warning: Could not load config file, using default dates: {e}")
        # Fallback to using min and max dates from the data
        all_dates = []
        for series in time_series_dict.values():
            if len(series) > 0:
                all_dates.extend(series.index.tolist())
        if all_dates:
            start_date = min(all_dates)
            end_date = max(all_dates)
        else:
            raise ValueError("No data available to compute completeness")

    days_between = (end_date - start_date).total_seconds() / (60 * 60 * 24)
    expected_records = days_between * 24 * 4  # 15-minute intervals = 4 per hour
    print(f"Total days between start and end date: {days_between}")

    completeness_dict = {}
    for sensor_id, series in time_series_dict.items():
        # Remove duplicates to ensure accurate count
        series = series[~series.index.duplicated(keep="first")]
        completeness_dict[sensor_id] = len(series) / expected_records

    return completeness_dict


# Create a comprehensive dashboard combining multiple visualizations
def create_sensor_window_dashboard(data_file, window_size=24):
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

    # Compute data completeness
    completeness_dict = compute_completeness(time_series_dict)

    # Create individual visualizations
    # 1. Sensor map with completeness data
    sensor_map_fig = create_sensors_map(completeness_dict)

    # 2. Sensor availability and patterns
    data_avail_fig = interactive_data_availability(time_series_dict)
    top_sensor_fig = interactive_sensor_windows(
        time_series_dict, top_sensor, window_size
    )
    window_counts_fig = interactive_window_counts(time_series_dict, window_size)
    daily_patterns_fig = visualize_daily_patterns(time_series_dict, n_sensors=4)

    # 3. New visualizations
    try:
        # Calendar heatmap for top sensor
        calendar_heatmap_fig = create_calendar_heatmap(time_series_dict, top_sensor)
    except Exception as e:
        print(f"Could not create calendar heatmap: {e}")
        calendar_heatmap_fig = None

    try:
        # Time of day profiles
        time_of_day_fig = create_time_of_day_profiles(time_series_dict)
    except Exception as e:
        print(f"Could not create time of day profiles: {e}")
        time_of_day_fig = None

    try:
        # Completeness trend
        completeness_trend_fig = create_completeness_trend(time_series_dict)
    except Exception as e:
        print(f"Could not create completeness trend: {e}")
        completeness_trend_fig = None

    try:
        # Top sensors comparison
        sensors_comparison_fig = create_sensors_comparison(time_series_dict)
    except Exception as e:
        print(f"Could not create sensors comparison: {e}")
        sensors_comparison_fig = None

    try:
        # Monthly coverage matrix
        coverage_matrix_fig = create_monthly_coverage_matrix(time_series_dict)
    except Exception as e:
        print(f"Could not create coverage matrix: {e}")
        coverage_matrix_fig = None

    try:
        # Sensor clustering
        sensor_clustering_fig = create_sensor_clustering(time_series_dict)
    except Exception as e:
        print(f"Could not create sensor clustering: {e}")
        sensor_clustering_fig = None

    # Load the template
    template_path = get_template_path("dashboard_template.html")
    template = load_template(template_path)

    # Prepare context with all variables for the template
    context = {
        "window_size": window_size,
        "top_sensor": top_sensor,
        "sensor_map_fig": to_html(
            sensor_map_fig, include_plotlyjs="cdn", full_html=False
        ),
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
        "calendar_heatmap_fig": (
            to_html(calendar_heatmap_fig, include_plotlyjs="cdn", full_html=False)
            if calendar_heatmap_fig
            else None
        ),
        "time_of_day_fig": (
            to_html(time_of_day_fig, include_plotlyjs="cdn", full_html=False)
            if time_of_day_fig
            else None
        ),
        "completeness_trend_fig": (
            to_html(completeness_trend_fig, include_plotlyjs="cdn", full_html=False)
            if completeness_trend_fig
            else None
        ),
        "sensors_comparison_fig": (
            to_html(sensors_comparison_fig, include_plotlyjs="cdn", full_html=False)
            if sensors_comparison_fig
            else None
        ),
        "coverage_matrix_fig": (
            to_html(coverage_matrix_fig, include_plotlyjs="cdn", full_html=False)
            if coverage_matrix_fig
            else None
        ),
        "sensor_clustering_fig": (
            to_html(sensor_clustering_fig, include_plotlyjs="cdn", full_html=False)
            if sensor_clustering_fig
            else None
        ),
    }

    # Render the template
    html_content = render_template(template, context)

    # Return the HTML content
    return html_content


if __name__ == "__main__":
    # Create the dashboard
    data_file = os.path.join(paths.RAW_TIMESERIES_DIR, "test_data_1mnth.pkl")
    dashboard_html = create_sensor_window_dashboard(data_file, window_size=24)

    # Save to a file
    output_path = Path(__file__).parent / "index.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    print(f"Dashboard created: {output_path}")
