│   ├── __init__.py
│   ├── __main__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   └── __main__.cpython-311.pyc
│   ├── components
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── counts_bar.cpython-311.pyc
│   │   │   ├── daily_patterns.cpython-311.pyc
│   │   │   ├── heatmap.cpython-311.pyc
│   │   │   ├── pca_plot.cpython-311.pyc
│   │   │   └── window_segments.cpython-311.pyc
│   │   ├── counts_bar.py
│   │   ├── daily_patterns.py
│   │   ├── heatmap.py
│   │   ├── pca_plot.py
│   │   └── window_segments.py
│   ├── data
│   │   └── test_data_1yr.pkl
│   ├── sensor_dashboard.html
│   ├── templates
│   │   ├── __init__.py
│   │   └── layout.html
│   ├── training_winodows.ipynb
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-311.pyc
│       │   └── data_utils.cpython-311.pyc
│       └── data_utils.py
├── docker-compose.yml
├── docs
│   └── dataflow.mmd
├── gnn_package
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── data
│   │   ├── preprocessed
│   │   │   └── graphs
│   │   │       ├── 25022025_test_adj_matrix.npy
│   │   │       └── 25022025_test_metadata.json
│   │   └── sensors
│   │       ├── sensor_name_id_map.json
│   │       ├── sensors.cpg
│   │       ├── sensors.dbf
│   │       ├── sensors.prj
│   │       ├── sensors.shp
│   │       ├── sensors.shx
│   │       ├── sensors_gdf.cpg
│   │       ├── sensors_gdf.dbf
│   │       ├── sensors_gdf.prj
│   │       ├── sensors_gdf.shp
│   │       └── sensors_gdf.shx
│   └── src
│       ├── dataloaders
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   └── dataloaders.cpython-311.pyc
│       │   └── dataloaders.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   └── stgnn.cpython-311.pyc
│       │   └── stgnn.py
│       ├── preprocessing
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   ├── graph_analysis.cpython-311.pyc
│       │   │   ├── graph_computation.cpython-311.pyc
│       │   │   ├── graph_manipulation.cpython-311.pyc
│       │   │   ├── graph_utils.cpython-311.pyc
│       │   │   ├── graph_visualization.cpython-311.pyc
│       │   │   ├── import_graph.cpython-311.pyc
│       │   │   └── timeseries_preprocessor.cpython-311.pyc
│       │   ├── graph_analysis.py
│       │   ├── graph_computation.py
│       │   ├── graph_manipulation.py
│       │   ├── graph_utils.py
│       │   ├── graph_visualization.py
│       │   └── timeseries_preprocessor.py
│       ├── training
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   └── stgnn_training.cpython-311.pyc
│       │   ├── stgnn_prediction.py
│       │   └── stgnn_training.py
│       ├── tuning
│       └── utils
│           ├── __pycache__
│           │   ├── data_utils.cpython-311.pyc
│           │   └── paths.cpython-311.pyc
│           ├── data_utils.py
│           └── paths.py
├── notebooks
│   ├── cache
│   │   ├── 498747a3c22f88c45a64f3de6b282b9cdb15cb39.json
│   │   └── 97deb2f4ddab0fd512ce7459f621ec48cb7ec583.json
│   ├── database
│   │   └── cache
│   │       ├── a9d991c54733a3e32479743ff9f5e001.pickle
│   │       └── c2ca55db1fb6a77a2968417b483ae067.pickle
│   ├── eda.ipynb
│   ├── gnn.ipynb
│   ├── graph_construction.ipynb
│   ├── pipeline.ipynb
│   ├── sensor_locations.ipynb
│   ├── stgnn_model.pth
│   ├── test_data.pkl
│   ├── test_data_1yr.pkl
│   └── testing
│       ├── test.ipynb
│       └── testing_graph_comp.ipynb
├── poetry.lock
├── pyproject.toml
├── requirements
├── services
│   ├── inference
│   ├── preprocessing
│   └── training
└── test
    └── test.py

39 directories, 92 files
gnn-package-py3.11➜  phd-project-gnn git:(dev)

39 directories, 92 files