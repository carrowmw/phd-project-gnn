.
├── README.md
├── config.yml
├── dashboards
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── data
│   │   └── sensors.geojson
│   ├── dataloader_explorer
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── __main__.cpython-311.pyc
│   │   ├── components
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── adjacency_plot.cpython-311.pyc
│   │   │   │   ├── batch_explorer.cpython-311.pyc
│   │   │   │   ├── correlation_plot.cpython-311.pyc
│   │   │   │   ├── data_stats.cpython-311.pyc
│   │   │   │   ├── node_explorer.cpython-311.pyc
│   │   │   │   └── window_explorer.cpython-311.pyc
│   │   │   ├── adjacency_plot.py
│   │   │   ├── batch_explorer.py
│   │   │   ├── correlation_plot.py
│   │   │   ├── data_stats.py
│   │   │   ├── node_explorer.py
│   │   │   └── window_explorer.py
│   │   ├── static
│   │   │   ├── css
│   │   │   │   └── styles.css
│   │   │   └── js
│   │   │       └── custom.js
│   │   ├── templates
│   │   │   ├── __init__.py
│   │   │   └── layout.html
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-311.pyc
│   │       │   └── data_utils.cpython-311.pyc
│   │       └── data_utils.py
│   ├── digester.sh
│   └── eda
│       ├── __init__.py
│       ├── __main__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-311.pyc
│       │   └── __main__.cpython-311.pyc
│       ├── components
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   ├── calendar_heatmap.cpython-311.pyc
│       │   │   ├── completeness_trend.cpython-311.pyc
│       │   │   ├── counts_bar.cpython-311.pyc
│       │   │   ├── daily_patterns.cpython-311.pyc
│       │   │   ├── heatmap.cpython-311.pyc
│       │   │   ├── monthly_data_coverage.cpython-311.pyc
│       │   │   ├── pca_plot.cpython-311.pyc
│       │   │   ├── sensor_clustering.cpython-311.pyc
│       │   │   ├── sensor_map.cpython-311.pyc
│       │   │   ├── traffic_comparison.cpython-311.pyc
│       │   │   ├── traffic_profile.cpython-311.pyc
│       │   │   └── window_segments.cpython-311.pyc
│       │   ├── calendar_heatmap.py
│       │   ├── completeness_trend.py
│       │   ├── counts_bar.py
│       │   ├── daily_patterns.py
│       │   ├── heatmap.py
│       │   ├── monthly_data_coverage.py
│       │   ├── sensor_clustering.py
│       │   ├── sensor_map.py
│       │   ├── traffic_comparison.py
│       │   ├── traffic_profile.py
│       │   └── window_segments.py
│       ├── index.html
│       ├── templates
│       │   ├── __init__.py
│       │   └── dashboard_template.html
│       └── utils
│           ├── __init__.py
│           ├── __pycache__
│           │   ├── __init__.cpython-311.pyc
│           │   ├── data_utils.cpython-311.pyc
│           │   └── template_utils.cpython-311.pyc
│           ├── data_utils.py
│           └── template_utils.py
├── digested_gnn_package_20250415.txt
├── docker-compose.yml
├── docs
│   ├── configuration_guide.md
│   ├── dataflow.mmd
│   ├── project_tree.md
│   └── tensor_flow.mmd
├── gnn_package
│   ├── __init__.py
│   ├── __main__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── config
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── config.cpython-311.pyc
│   │   │   └── paths.cpython-311.pyc
│   │   ├── config.py
│   │   ├── config.yml
│   │   ├── config_manager.py
│   │   └── paths.py
│   ├── data
│   │   ├── preprocessed
│   │   │   ├── graphs
│   │   │   │   ├── 25022025_test_adj_matrix.npy
│   │   │   │   └── 25022025_test_metadata.json
│   │   │   └── timeseries
│   │   │       ├── data_loaders_test_1mnth.pkl
│   │   │       ├── data_loaders_test_1wk.pkl
│   │   │       └── data_loaders_test_1yr.pkl
│   │   ├── raw
│   │   │   └── timeseries
│   │   │       ├── test_data_1mnth.pkl
│   │   │       ├── test_data_1wk.pkl
│   │   │       └── test_data_1yr.pkl
│   │   └── sensors
│   │       ├── sensor_name_id_map.json
│   │       ├── sensors.cpg
│   │       ├── sensors.dbf
│   │       ├── sensors.prj
│   │       ├── sensors.shp
│   │       ├── sensors.shx
│   │       ├── sensors_gdf.cpg
│   │       ├── sensors_gdf.dbf
│   │       ├── sensors_gdf.prj
│   │       ├── sensors_gdf.shp
│   │       └── sensors_gdf.shx
│   ├── digester.sh
│   └── src
│       ├── dataloaders
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   └── dataloaders.cpython-311.pyc
│       │   └── dataloaders.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   └── stgnn.cpython-311.pyc
│       │   └── stgnn.py
│       ├── preprocessing
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   ├── fetch_sensor_data.cpython-311.pyc
│       │   │   ├── graph_analysis.cpython-311.pyc
│       │   │   ├── graph_computation.cpython-311.pyc
│       │   │   ├── graph_manipulation.cpython-311.pyc
│       │   │   ├── graph_utils.cpython-311.pyc
│       │   │   ├── graph_visualization.cpython-311.pyc
│       │   │   ├── import_graph.cpython-311.pyc
│       │   │   └── timeseries_preprocessor.cpython-311.pyc
│       │   ├── fetch_sensor_data.py
│       │   ├── graph_analysis.py
│       │   ├── graph_computation.py
│       │   ├── graph_manipulation.py
│       │   ├── graph_utils.py
│       │   ├── graph_visualization.py
│       │   └── timeseries_preprocessor.py
│       ├── training
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   ├── stgnn_prediction.cpython-311.pyc
│       │   │   └── stgnn_training.cpython-311.pyc
│       │   ├── stgnn_prediction.py
│       │   └── stgnn_training.py
│       ├── tuning
│       └── utils
│           ├── __init__.py
│           ├── __pycache__
│           │   ├── __init__.cpython-311.pyc
│           │   ├── data_utils.cpython-311.pyc
│           │   ├── paths.cpython-311.pyc
│           │   └── sensor_utils.cpython-311.pyc
│           ├── data_utils.py
│           └── sensor_utils.py
├── notebooks
│   ├── cache
│   │   ├── 498747a3c22f88c45a64f3de6b282b9cdb15cb39.json
│   │   └── 97deb2f4ddab0fd512ce7459f621ec48cb7ec583.json
│   ├── database
│   │   └── cache
│   │       ├── a9d991c54733a3e32479743ff9f5e001.pickle
│   │       └── c2ca55db1fb6a77a2968417b483ae067.pickle
│   ├── eda.ipynb
│   ├── graph_construction.ipynb
│   ├── prediction.ipynb
│   ├── predictions.csv
│   ├── predictions.png
│   ├── stgnn_model.pth
│   ├── stgnn_model_test_data_1mnth.pth
│   ├── stgnn_model_test_data_1yr.pth
│   ├── testing
│   │   ├── test.ipynb
│   │   └── testing_graph_comp.ipynb
│   └── training.ipynb
├── poetry.lock
├── pyproject.toml
├── requirements
└── test
    └── test.py

52 directories, 164 files
