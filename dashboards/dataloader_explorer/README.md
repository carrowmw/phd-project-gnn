# DataLoader Explorer

An interactive dashboard for exploring DataLoader structures for GNN-based time series forecasting.

## Features

- **Batch Explorer**: Visualize data availability across nodes and time steps in a batch
- **Node Explorer**: Examine individual node data with input/target visualization
- **Window Explorer**: View multiple windows for a specific node in a heatmap
- **Correlation Analysis**: Analyze correlations between different nodes
- **Adjacency Matrix**: Visualize and explore the graph structure
- **Statistics**: View comprehensive statistics about your data

## Installation

This module requires the following dependencies:

```bash
pip install dash dash-bootstrap-components plotly pandas numpy torch networkx
```

## Usage

### Running the Dashboard

To run the DataLoader Explorer dashboard, use the following command:

```bash
python -m dashboards.dataloader_explorer --data path/to/data_loaders.pkl
```

Optional arguments:
- `--port PORT`: Specify the port to run the dashboard on (default: 8050)
- `--debug`: Run the dashboard in debug mode

### Programmatic Usage

You can also use the components programmatically in your own code:

```python
from dashboards.dataloader_explorer import load_data_loaders, create_node_explorer

# Load your data
data_loaders = load_data_loaders("path/to/data_loaders.pkl")

# Create a visualization
node_fig = create_node_explorer(data_loaders, loader_key='train_loader', batch_idx=0, node_idx=0)

# Display the figure
node_fig.show()
```

## Dashboard Components

1. **Batch Explorer**
   - Overview of data availability in a batch
   - Highlights missing data across nodes and time steps

2. **Node Explorer**
   - Detailed view of individual node data
   - Shows input (historical) and target (future) values
   - Clearly marks missing data points

3. **Window Explorer**
   - Visualizes multiple windows for a node
   - Shows patterns across windows using a heatmap

4. **Correlation Analysis**
   - Calculates and visualizes correlations between nodes
   - Helps identify related nodes in the network

5. **Adjacency Matrix**
   - Visualizes the graph structure
   - Toggle between heatmap and network views

6. **Statistics**
   - Comprehensive statistics about your data
   - Distribution, missing values, adjacency matrix properties

## Integration with Existing Dashboards

This module follows the same structure as other dashboard modules in the project, making it easy to integrate with existing visualization workflows.

## Development

To contribute to DataLoader Explorer, follow the standard development practices:

1. Fork the repository
2. Install development dependencies
3. Implement your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.