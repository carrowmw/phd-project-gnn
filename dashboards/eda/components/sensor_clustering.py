import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def create_sensor_clustering(time_series_dict, n_clusters=4):
    """
    Create a visualization of sensors clustered by their traffic patterns.

    Parameters:
    -----------
    time_series_dict : dict
        Dictionary mapping sensor IDs to their time series data
    n_clusters : int
        Number of clusters to identify

    Returns:
    --------
    plotly.graph_objects.Figure
        Sensor clustering figure
    """
    # First, create feature vectors for each sensor
    features = []
    sensor_ids = []

    for sensor_id, series in time_series_dict.items():
        if len(series) < 24:  # Skip sensors with very little data
            continue

        # Create a DataFrame with time components
        df = pd.DataFrame(
            {
                "value": series.values,
                "hour": series.index.hour,
                "day_of_week": series.index.dayofweek,
            }
        )

        # Create hourly profile (average by hour of day)
        hourly_profile = (
            df.groupby("hour")["value"].mean().reindex(range(24)).fillna(0).values
        )

        # Create day of week profile
        dow_profile = (
            df.groupby("day_of_week")["value"].mean().reindex(range(7)).fillna(0).values
        )

        # Create weekend vs weekday ratio feature
        weekday_avg = df[df["day_of_week"] < 5]["value"].mean() or 0
        weekend_avg = df[df["day_of_week"] >= 5]["value"].mean() or 0
        weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0

        # Create morning vs evening ratio
        morning = df[(df["hour"] >= 6) & (df["hour"] < 12)]["value"].mean() or 0
        evening = df[(df["hour"] >= 16) & (df["hour"] < 22)]["value"].mean() or 0
        ampm_ratio = morning / evening if evening > 0 else 0

        # Combine all features
        feature_vector = np.concatenate(
            [
                hourly_profile,  # 24 features
                dow_profile,  # 7 features
                [weekend_ratio, ampm_ratio],  # 2 additional features
            ]
        )

        features.append(feature_vector)
        sensor_ids.append(sensor_id)

    if len(features) < n_clusters:
        return go.Figure().update_layout(
            title=f"Not enough data for clustering. Need at least {n_clusters} sensors with sufficient data."
        )

    # Convert to numpy array
    X = np.array(features)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "cluster": clusters.astype(str),
            "pc1": X_pca[:, 0],
            "pc2": X_pca[:, 1],
        }
    )

    # Create scatter plot of clusters
    fig = px.scatter(
        plot_df,
        x="pc1",
        y="pc2",
        color="cluster",
        hover_name="sensor_id",
        title=f"Sensor Clustering Based on Traffic Patterns (k={n_clusters})",
        labels={
            "pc1": f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            "pc2": f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            "cluster": "Cluster",
        },
    )

    # Add cluster centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    fig.add_trace(
        go.Scatter(
            x=centroids_pca[:, 0],
            y=centroids_pca[:, 1],
            mode="markers",
            marker=dict(symbol="x", size=12, color="black", line=dict(width=2)),
            name="Cluster Centroids",
            hoverinfo="skip",
        )
    )

    # Add annotations for cluster numbers
    for i, (x, y) in enumerate(centroids_pca):
        fig.add_annotation(
            x=x,
            y=y,
            text=f"{i}",
            showarrow=False,
            yshift=15,
            font=dict(size=14, color="black"),
        )

    # Update layout
    fig.update_layout(height=600, legend_title="Cluster", hovermode="closest")

    return fig
