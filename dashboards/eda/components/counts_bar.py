import pandas as pd
import plotly.express as px
from dashboards.eda.utils import find_continuous_segments


# Create a window count bar chart
def interactive_window_counts(time_series_dict, window_size=24, n_sensors=20):
    """Create an interactive bar chart of window counts by sensor"""
    # Count windows per sensor
    window_counts = {}

    for sensor_id, series in time_series_dict.items():
        # Find segments without large gaps
        segments = find_continuous_segments(series.index, series.values)

        # Count windows
        total_windows = 0
        for start_seg, end_seg in segments:
            segment_len = end_seg - start_seg
            total_windows += max(0, segment_len - window_size + 1)

        window_counts[sensor_id] = total_windows

    # Sort by window count
    sorted_counts = sorted(window_counts.items(), key=lambda x: x[1], reverse=True)[
        :n_sensors
    ]

    # Create a DataFrame
    count_df = pd.DataFrame(sorted_counts, columns=["sensor_id", "window_count"])

    # Create a bar chart
    fig = px.bar(
        count_df,
        x="sensor_id",
        y="window_count",
        title=f"Number of Available Windows (size={window_size}) by Sensor",
        labels={"sensor_id": "Sensor ID", "window_count": "Number of Windows"},
        color="window_count",
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    fig.update_layout(height=600, xaxis_tickangle=-45)

    return fig
