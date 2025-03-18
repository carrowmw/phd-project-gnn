import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dashboards.utils import find_continuous_segments


# Create a PCA visualization of sensor windows
def visualize_window_pca(time_series_dict, window_size=24, n_sensors=10):
    """Create a PCA visualization of sensor windows to see patterns"""
    # Collect window data
    windows_data = []
    sensor_ids = []

    # Process top sensors
    sensor_data_counts = {
        sensor_id: len(series) for sensor_id, series in time_series_dict.items()
    }
    top_sensors = sorted(sensor_data_counts, key=sensor_data_counts.get, reverse=True)[
        :n_sensors
    ]

    for sensor_id in top_sensors:
        series = time_series_dict[sensor_id]

        # Find continuous segments
        segments = find_continuous_segments(series.index, series.values)

        # Extract windows
        for start_seg, end_seg in segments:
            segment_values = series.values[start_seg:end_seg]

            # Create windows
            for i in range(
                0, len(segment_values) - window_size + 1, window_size // 2
            ):  # 50% overlap
                window = segment_values[i : i + window_size]

                if not np.isnan(window).any():  # Skip windows with NaN values
                    windows_data.append(window)
                    sensor_ids.append(sensor_id)

    if not windows_data:
        return None

    # Convert to numpy array
    X = np.array(windows_data)

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(
        {"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "sensor_id": sensor_ids}
    )

    # Create a scatter plot
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="sensor_id",
        title="PCA of Sensor Windows",
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)",
        },
        hover_data=["sensor_id"],
    )

    fig.update_layout(height=700, width=900)

    return fig
