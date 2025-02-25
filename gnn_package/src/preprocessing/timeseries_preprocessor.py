import sys
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import aiohttp
import nest_asyncio
import asyncio
import uoapi
import private_uoapi
from gnn_package.src.preprocessing.graph_utils import get_sensor_name_id_map


@dataclass
class NodeGroup:
    """Groups of node IDs by API type"""

    private_nodes: List[str]
    public_nodes: List[str]


@dataclass
class APIStats:
    attempted_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    errors: Dict[str, List[str]] = field(default_factory=dict)


class FetcherResponse(NamedTuple):
    data: Dict[str, Optional[pd.Series]]
    stats: APIStats


class SensorDataFetcher:
    def __init__(self):
        self.name_id_map = get_sensor_name_id_map()
        self.id_name_map = {v: k for k, v in self.name_id_map.items()}
        self.stats = APIStats()
        self.lock = asyncio.Lock()  # prevent race conditions from multiple coroutines
        # Enable nested event loops for Jupyter
        if "ipykernel" in sys.modules:
            nest_asyncio.apply()

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
        self._session = None

    def _convert_to_utc(self, series: pd.Series) -> pd.Series:
        """Convert timestamp index to UTC if it isn't already"""
        # Ensure index is a DatetimeIndex
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series index must be a DatetimeIndex")

        # Localize naive timestamps to UTC
        if series.index.tz is None:
            series = pd.Series(series.values, index=series.index.tz_localize("UTC"))
        # Convert to UTC if different timezone
        elif series.index.tz != timezone.utc:
            series = pd.Series(
                series.values, index=series.index.tz_convert(timezone.utc)
            )
        return series

    def _log_error(self, node_id: str, error: str):
        """Log error for a specific node"""
        if node_id not in self.stats.errors:
            self.stats.errors[node_id] = []
        self.stats.errors[node_id].append(error)
        self.stats.failed_calls += 1

    def _group_nodes_by_api(self, node_ids: List[str]) -> NodeGroup:
        """Group node IDs by which API they should use"""
        private_nodes = []
        public_nodes = []

        for node_id in node_ids:
            if node_id[0] == "1":
                private_nodes.append(node_id)
            elif node_id[0] in ["7", "8"]:
                public_nodes.append(node_id)
            else:
                raise ValueError(f"Invalid node ID {node_id}")

        return NodeGroup(private_nodes, public_nodes)

    async def _fetch_private_data(
        self, node_ids: List[str], days_back: int
    ) -> Dict[str, pd.Series]:
        """Fetch data for private API nodes"""
        if not node_ids:
            return {}

        results = {}
        config = private_uoapi.APIConfig()

        try:
            # use context manager for auth and client
            async with private_uoapi.APIAuth(config) as auth:
                async with private_uoapi.APIClient(config, auth) as client:
                    sem = asyncio.Semaphore(3)  # rate limiting

                    async def fetch(node_id):
                        async with sem:
                            try:
                                async with self.lock:
                                    self.stats.attempted_calls += 1

                                sensor_name = self.id_name_map[node_id]
                                df = await client.get_historical_traffic_counts(
                                    locations=sensor_name, days_back=days_back
                                )
                                # set the dt column as the index
                                df.set_index("dt", inplace=True)
                                # ensure the index is datetime
                                df.index = pd.to_datetime(df.index)
                                # save the index and value columns as a series
                                series = pd.Series(df["value"].values, index=df.index)

                                # print(f"DEBUG: {sensor_name} - {df}")
                                # dt_index = pd.to_datetime(
                                #     df["dt"],
                                #     format="%Y-%m-%d %H:%M:%S GMT",
                                # ).tz_localize(
                                #     "GMT"
                                # )  # first localize to GMT

                                # if not isinstance(dt_index, pd.DatetimeIndex):
                                #     raise ValueError(
                                #         "Failed to parse datetime values from private API"
                                #     )

                                # # Create series with proper datetime index
                                # series = (
                                #     pd.Series(df["value"].values, index=dt_index),
                                # )

                                # # Now convert to UTC
                                # series = self._convert_to_utc(series)
                                # print(f"DEBUG: {sensor_name} - {series}")
                                async with self.lock:
                                    self.stats.successful_calls += 1
                                return node_id, series

                            except Exception as e:
                                async with self.lock:
                                    self.stats.failed_calls += 1
                                    self.stats.errors.setdefault(node_id, []).append(
                                        str(e)
                                    )
                            return node_id, None

                    # Use asyncio.gather with semaphore
                    tasks = [fetch(node_id) for node_id in node_ids]
                    for future in asyncio.as_completed(tasks):
                        node_id, result = await future
                        results[node_id] = result
        except Exception as e:
            print(f"Error fetching private data: {e}")
            raise
        finally:
            # cleanup any dangling sessions
            if hasattr(self, "_session") and self._session:
                await self._session.close()

        return results

    async def _fetch_public_data(
        self, node_ids: List[str], days_back: int
    ) -> Dict[str, pd.Series]:
        """Fetch data for public API nodes using async client"""
        if not node_ids:
            return {}

        results = {}
        sensor_names = [self.id_name_map[nid] for nid in node_ids]

        async with uoapi.AsyncAPIClient(
            uoapi.AsyncAPIConfig(max_concurrent_requests=5)
        ) as client:
            async with self.lock:
                self.stats.attempted_calls += len(node_ids)

            responses = await client.get_sensor_data_batch(sensor_names, days_back)

            for node_id in node_ids:
                try:
                    sensor_name = self.id_name_map[node_id]
                    data = (
                        responses.get(sensor_name, {})
                        .get("sensors", [{}])[0]
                        .get("data", {})
                        .get("Walking", [])
                    )
                    # print(data)
                    if data:
                        # Convert timestamps to timezone-aware datetime index
                        timestamps = pd.to_datetime(
                            [x["Timestamp"] for x in data]
                        ).tz_localize(
                            "Europe/London"
                        )  # Localize to observed timezone

                        if not isinstance(timestamps, pd.DatetimeIndex):
                            raise ValueError(
                                "Failed to parse datetime values from public API"
                            )

                        # Create series with proper datetime index
                        series = pd.Series([x["Value"] for x in data], index=timestamps)

                        # Convert to UTC
                        series = self._convert_to_utc(series)

                        async with self.lock:
                            self.stats.successful_calls += 1
                        results[node_id] = series
                    else:
                        raise ValueError("No walking data")

                except Exception as e:
                    async with self.lock:
                        self.stats.failed_calls += 1
                        self.stats.errors.setdefault(node_id, []).append(str(e))

        return results

    async def _fetch_all_data(
        self, node_ids: List[str], days_back: int
    ) -> FetcherResponse:
        self.stats = APIStats()  # Reset stats

        # Group nodes by API type
        private = [nid for nid in node_ids if nid.startswith("1")]
        public = [nid for nid in node_ids if nid[0] in {"7", "8"}]

        # Parallel fetch
        private_data, public_data = await asyncio.gather(
            self._fetch_private_data(private, days_back),
            self._fetch_public_data(public, days_back),
        )

        return FetcherResponse(data={**private_data, **public_data}, stats=self.stats)

    def get_sensor_data_batch(
        self, node_ids: List[str], days_back: int = 7
    ) -> FetcherResponse:
        """
        Main entry point for fetching sensor data in batch

        Parameters:
        -----------
        node_ids : List[str]
            List of node IDs to fetch data for
        days_back : int, optional
            Number of days of historical data to fetch

        Returns:
        --------
        FetcherResponse
            Named tuple containing:
            - data: Dict[str, Optional[pd.Series]]
                Dictionary mapping node IDs to their time series data
            - stats: APIStats
                Stats on attempted, successful, and failed API calls
        """
        try:
            # First try getting the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create explicit session for notebook environment
                async def wrapper():
                    async with self:  # use the class's own context manager
                        return await self._fetch_all_data(node_ids, days_back)

                return loop.run_until_complete(wrapper())
            else:
                return loop.run_until_complete(
                    self._fetch_all_data(node_ids, days_back)
                )

        except RuntimeError:
            # Fallback if there's any issue
            return asyncio.run(self._fetch_all_data(node_ids, days_back))

    def get_timeseries_data_for_node(
        self, node_id: str, days_back: int = 7
    ) -> Optional[pd.Series]:
        """
        Get time series data for a single node (backward compatibility)
        """
        response = self.get_sensor_data_batch([node_id], days_back)
        return response.get(node_id)


@dataclass
class TimeWindow:
    start_idx: int
    end_idx: int
    node_id: str
    mask: np.ndarray  # 1 for valid data, 0 for missing


class TimeSeriesPreprocessor:
    def __init__(
        self,
        window_size: int,
        stride: int,
        gap_threshold: pd.Timedelta,
        missing_value: float = -1.0,
    ):
        """
        Initialize the preprocessor for handling time series with gaps.

        Parameters:
        -----------
        window_size : int
            Size of the sliding window
        stride : int
            Number of steps to move the window
        gap_threshold : pd.Timedelta
            Maximum allowed time difference between consecutive points
            e.g., pd.Timedelta(hours=1) for hourly data
        missing_value : float
            Value to use for marking missing data
        """
        self.window_size = window_size
        self.stride = stride
        self.gap_threshold = gap_threshold
        self.missing_value = missing_value

    def find_continuous_segments(
        self, time_index: pd.DatetimeIndex, values: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Find continuous segments of data without gaps.

        Returns list of (start_idx, end_idx) tuples.
        """
        segments = []
        start_idx = 0

        for i in range(1, len(time_index)):
            time_diff = time_index[i] - time_index[i - 1]

            # Check for gaps in time or values
            if (time_diff > self.gap_threshold) or (
                np.isnan(values[i - 1]) or np.isnan(values[i])
            ):
                if i - start_idx >= self.window_size:
                    segments.append((start_idx, i))
                start_idx = i

        # Add the last segment if it's long enough
        if len(time_index) - start_idx >= self.window_size:
            segments.append((start_idx, len(time_index)))

        return segments

    def create_windows(
        self, time_series_dict: Dict[str, pd.Series], standardize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[TimeWindow]]:
        """
        Create windowed data with masks for missing values.

        Parameters:
        -----------
        time_series_dict : Dict[str, pd.Series]
            Dictionary mapping node IDs to their time series
        standardize : bool
            Whether to standardize the data

        Returns:
        --------
        X : np.ndarray
            Array of shape (n_windows, n_nodes, window_size)
        masks : np.ndarray
            Binary masks of same shape as X
        metadata : List[TimeWindow]
            Metadata for each window
        """
        all_windows = []
        all_masks = []
        window_metadata = []

        # Process each node's time series
        for node_id, series in time_series_dict.items():
            # Find continuous segments
            segments = self.find_continuous_segments(series.index, series.values)

            # Process each segment
            for start_seg, end_seg in segments:
                segment_values = series.values[start_seg:end_seg]

                if standardize:
                    # Standardize non-missing values
                    valid_mask = ~np.isnan(segment_values)
                    valid_values = segment_values[valid_mask]
                    if len(valid_values) > 0:
                        mean = np.mean(valid_values)
                        std = np.std(valid_values)
                        segment_values = (segment_values - mean) / (std + 1e-8)

                # Create windows for this segment
                for i in range(
                    0, len(segment_values) - self.window_size + 1, self.stride
                ):
                    window = segment_values[i : i + self.window_size]
                    mask = ~np.isnan(window)

                    # Replace NaN with missing_value
                    window = np.where(mask, window, self.missing_value)

                    all_windows.append(window)
                    all_masks.append(mask)

                    window_metadata.append(
                        TimeWindow(
                            start_idx=start_seg + i,
                            end_idx=start_seg + i + self.window_size,
                            node_id=node_id,
                            mask=mask,
                        )
                    )

        # Stack all windows
        X = np.stack(all_windows)
        masks = np.stack(all_masks)

        return X, masks, window_metadata

    def prepare_batch_data(
        self, X: np.ndarray, masks: np.ndarray, adj_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for GNN training.

        Returns:
        --------
        dict containing:
            - node_features: Time series windows
            - adjacency: Adjacency matrix
            - mask: Binary mask for missing values
        """
        return {
            "node_features": X,
            "adjacency": adj_matrix,
            "mask": masks.astype(np.float32),
        }


# # Example usage:
# if __name__ == "__main__":
#     # Create sample data
#     dates = pd.date_range('2024-01-01', '2024-01-31', freq='1H')
#     data = {
#         'node1': pd.Series(np.random.randn(len(dates)), index=dates),
#         'node2': pd.Series(np.random.randn(len(dates)), index=dates)
#     }

#     # Add some gaps
#     data['node1'].iloc[100:150] = np.nan
#     data['node2'].iloc[200:250] = np.nan

#     # Initialize preprocessor
#     preprocessor = TimeSeriesPreprocessor(
#         window_size=24,  # 24-hour windows
#         stride=12,       # 12-hour stride
#         gap_threshold=pd.Timedelta(hours=2)  # 2-hour gap threshold
#     )

#     # Process data
#     X, masks, metadata = preprocessor.create_windows(data)
#     print(f"Created {len(metadata)} windows")
#     print(f"Features shape: {X.shape}")
#     print(f"Masks shape: {masks.shape}")
