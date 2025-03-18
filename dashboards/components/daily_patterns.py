import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Create a heatmap of daily patterns
def visualize_daily_patterns(time_series_dict, n_sensors=6):
    """Create a heatmap of daily patterns for top sensors"""
    # Identify sensors with most data points
    sensor_data_counts = {
        sensor_id: len(series) for sensor_id, series in time_series_dict.items()
    }
    top_sensors = sorted(sensor_data_counts, key=sensor_data_counts.get, reverse=True)[
        :n_sensors
    ]

    # Create subplots
    fig = make_subplots(
        rows=n_sensors,
        cols=1,
        subplot_titles=[
            f"Sensor {sensor_id} - Daily Pattern" for sensor_id in top_sensors
        ],
        vertical_spacing=0.08,
    )

    # Process each sensor
    for i, sensor_id in enumerate(top_sensors):
        series = time_series_dict[sensor_id]

        # Create a DataFrame with hour and day of week
        df = pd.DataFrame(
            {
                "value": series.values,
                "hour": series.index.hour,
                "day_of_week": series.index.dayofweek,
            }
        )

        # Group by hour and day of week
        pivot_data = df.pivot_table(
            values="value", index="day_of_week", columns="hour", aggfunc="mean"
        ).fillna(0)

        # Create heatmap
        heatmap = go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale="Viridis",
            showscale=(i == 0),  # Only show colorbar for first heatmap
        )

        fig.add_trace(heatmap, row=i + 1, col=1)

        # Update axes
        fig.update_xaxes(
            title_text="Hour of Day" if i == n_sensors - 1 else "", row=i + 1, col=1
        )
        fig.update_yaxes(title_text="Day of Week", row=i + 1, col=1)

    # Update layout
    fig.update_layout(
        height=250 * n_sensors, title_text="Daily Traffic Patterns Across Sensors"
    )

    return fig
