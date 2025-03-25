import plotly.graph_objects as go
from dashboards.eda.utils.data_utils import find_continuous_segments


# Create interactive window visualization for a given sensor
def interactive_sensor_windows(time_series_dict, sensor_id, window_size=24, stride=1):
    """Create an interactive visualization of windows for a specific sensor"""
    series = time_series_dict.get(sensor_id)
    if series is None or len(series) == 0:
        return None

    # Find continuous segments
    segments = find_continuous_segments(series.index, series.values)

    # Create a figure
    fig = go.Figure()

    # Add the raw time series
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Raw Data",
            line=dict(color="darkgray"),
        )
    )

    # Add segments and windows
    for start_seg, end_seg in segments:
        segment_indices = series.index[start_seg:end_seg]

        # Add segment highlight
        fig.add_trace(
            go.Scatter(
                x=[
                    segment_indices[0],
                    segment_indices[0],
                    segment_indices[-1],
                    segment_indices[-1],
                ],
                y=[
                    series.values.min(),
                    series.values.max(),
                    series.values.max(),
                    series.values.min(),
                ],
                fill="toself",
                mode="none",
                name=f"Segment: {segment_indices[0].date()} to {segment_indices[-1].date()}",
                fillcolor="rgba(144,238,144,0.2)",
                showlegend=True,
            )
        )

        # Add a few example windows
        n_windows = len(segment_indices) - window_size + 1

        # Only show a few windows to avoid overcrowding
        window_step = max(1, n_windows // 5)

        for i in range(0, n_windows, window_step):
            window_start = segment_indices[i]
            window_end = segment_indices[i + window_size - 1]

            fig.add_trace(
                go.Scatter(
                    x=[window_start, window_start, window_end, window_end],
                    y=[
                        series.values.min(),
                        series.values.max(),
                        series.values.max(),
                        series.values.min(),
                    ],
                    fill="toself",
                    mode="none",
                    name=f"Window: {window_start}",
                    fillcolor="rgba(0,0,255,0.1)",
                    showlegend=False,
                )
            )

    # Update layout
    fig.update_layout(
        title=f"Time Windows for Sensor {sensor_id}",
        xaxis_title="Date",
        yaxis_title="Traffic Count",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="right", x=1),
    )

    return fig
