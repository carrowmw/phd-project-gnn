import aiohttp
from gnn_package import preprocessing

adj_matrix_dense, node_ids, metadata = preprocessing.load_graph_data(
    prefix="publ_priv_test", return_df=False
)
name_id_map = preprocessing.get_sensor_name_id_map()
# node_names = [name_id_map[str(node_id)] for node_id in node_ids]
# adj_matrix_dense.max()

adj = preprocessing.compute_adjacency_matrix(adj_matrix_dense)
adj[0].max()


async def main():
    fetcher = preprocessing.SensorDataFetcher()
    async with aiohttp.ClientSession() as session:
        fetcher._session = session  # Manually inject session
        response = await fetcher._fetch_all_data(node_ids, days_back=7)
        return await response


response = main()

results = response[0]

results_containing_data = {
    node_id: data for node_id, data in results.items() if data is not None
}

print(len(results_containing_data))
