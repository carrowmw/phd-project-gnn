.
├── __init__.py
├── __main__.py
├── __pycache__
│   └── __init__.cpython-311.pyc
├── config
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── config.cpython-311.pyc
│   │   ├── config_manager.cpython-311.pyc
│   │   └── paths.cpython-311.pyc
│   ├── config.py
│   ├── config_manager.py
│   └── paths.py
├── config.yml
├── data
│   ├── preprocessed
│   │   ├── graphs
│   │   │   ├── 25022025_test_adj_matrix.npy
│   │   │   ├── 25022025_test_metadata.json
│   │   │   ├── default_graph_adj_matrix.npy
│   │   │   └── default_graph_metadata.json
│   │   └── timeseries
│   ├── raw
│   │   └── timeseries
│   │       ├── test_data_1mo_person.pkl
│   │       ├── test_data_1wk_person.pkl
│   │       ├── test_data_1yr_person.pkl
│   │       └── test_data_3mo_person.pkl
│   └── sensors
│       ├── sensor_name_id_map.json
│       ├── sensors.cpg
│       ├── sensors.dbf
│       ├── sensors.prj
│       ├── sensors.shp
│       ├── sensors.shx
│       ├── sensors_gdf.cpg
│       ├── sensors_gdf.dbf
│       ├── sensors_gdf.prj
│       ├── sensors_gdf.shp
│       └── sensors_gdf.shx
├── digester.sh
├── experiments
│   ├── test_config_1mo.yml
│   ├── test_config_1wk.yml
│   ├── test_config_1yr.yml
│   └── test_config_3mo.yml
├── prediction_service.log
├── prediction_service.py
├── results
│   ├── test_1mo
│   │   ├── config.yml
│   │   ├── config_backup.yml
│   │   ├── data_loaders_test_data_1mo_person.pkl_20250429_212844.pkl
│   │   ├── model.pth
│   │   ├── performance.json
│   │   ├── prediction
│   │   │   ├── error_distribution_20250429_220814.png
│   │   │   ├── predictions_20250429_220814.csv
│   │   │   ├── sensors_grid_20250429_220814.png
│   │   │   └── summary_20250429_220814.txt
│   │   ├── runtime_stats.json
│   │   └── training_curve.png
│   ├── test_1wk
│   │   ├── config.yml
│   │   ├── config_backup.yml
│   │   ├── data_loaders_test_data_1wk_person.pkl_20250429_212807.pkl
│   │   ├── model.pth
│   │   ├── performance.json
│   │   ├── prediction
│   │   │   ├── error_distribution_20250429_220554.png
│   │   │   ├── predictions_20250429_220554.csv
│   │   │   ├── sensors_grid_20250429_220554.png
│   │   │   └── summary_20250429_220554.txt
│   │   ├── runtime_stats.json
│   │   └── training_curve.png
│   ├── test_1yr
│   │   ├── config.yml
│   │   ├── config_backup.yml
│   │   ├── data_loaders_test_data_1yr_person.pkl_20250429_213027.pkl
│   │   ├── model.pth
│   │   ├── performance.json
│   │   ├── prediction
│   │   │   ├── error_distribution_20250430_082303.png
│   │   │   ├── error_distribution_20250430_102336.png
│   │   │   ├── predictions_20250430_082303.csv
│   │   │   ├── predictions_20250430_102336.csv
│   │   │   ├── sensors_grid_20250430_082303.png
│   │   │   ├── sensors_grid_20250430_102336.png
│   │   │   ├── summary_20250430_082303.txt
│   │   │   └── summary_20250430_102336.txt
│   │   ├── runtime_stats.json
│   │   └── training_curve.png
│   └── test_3mo
│       ├── config.yml
│       ├── config_backup.yml
│       ├── data_loaders_test_data_3mo_person.pkl_20250429_213013.pkl
│       ├── model.pth
│       ├── performance.json
│       ├── prediction
│       ├── runtime_stats.json
│       └── training_curve.png
├── run_experiment.py
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── data_sources.cpython-311.pyc
│   │   │   ├── factory.cpython-311.pyc
│   │   │   ├── processors.cpython-311.pyc
│   │   │   └── registry.cpython-311.pyc
│   │   ├── data_sources.py
│   │   ├── factory.py
│   │   ├── processors.py
│   │   └── registry.py
│   ├── dataloaders
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── dataloaders.cpython-311.pyc
│   │   └── dataloaders.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── registry.cpython-311.pyc
│   │   │   └── stgnn.cpython-311.pyc
│   │   ├── architectures.py
│   │   ├── factory.py
│   │   ├── layers.py
│   │   └── registry.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── fetch_sensor_data.cpython-311.pyc
│   │   │   ├── graph_analysis.cpython-311.pyc
│   │   │   ├── graph_computation.cpython-311.pyc
│   │   │   ├── graph_manipulation.cpython-311.pyc
│   │   │   ├── graph_utils.cpython-311.pyc
│   │   │   ├── graph_visualization.cpython-311.pyc
│   │   │   ├── import_graph.cpython-311.pyc
│   │   │   └── timeseries_preprocessor.cpython-311.pyc
│   │   ├── fetch_sensor_data.py
│   │   ├── graph_analysis.py
│   │   ├── graph_computation.py
│   │   ├── graph_manipulation.py
│   │   ├── graph_utils.py
│   │   ├── graph_visualization.py
│   │   └── timeseries_preprocessor.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── stgnn_prediction.cpython-311.pyc
│   │   │   └── stgnn_training.cpython-311.pyc
│   │   ├── stgnn_prediction.py
│   │   └── stgnn_training.py
│   ├── tuning
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── experiment_manager.cpython-311.pyc
│   │   │   ├── objective.cpython-311.pyc
│   │   │   ├── parameter_space.cpython-311.pyc
│   │   │   └── tuning_utils.cpython-311.pyc
│   │   ├── experiment_manager.py
│   │   ├── objective.py
│   │   ├── parameter_space.py
│   │   └── tuning_utils.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── config_utils.cpython-311.pyc
│   │   │   ├── data_utils.cpython-311.pyc
│   │   │   ├── device_utils.cpython-311.pyc
│   │   │   ├── exceptions.cpython-311.pyc
│   │   │   ├── logging_utils.cpython-311.pyc
│   │   │   ├── metrics.cpython-311.pyc
│   │   │   ├── model_io.cpython-311.pyc
│   │   │   ├── paths.cpython-311.pyc
│   │   │   ├── retry_utils.cpython-311.pyc
│   │   │   └── sensor_utils.cpython-311.pyc
│   │   ├── config_utils.py
│   │   ├── data_management.py
│   │   ├── data_utils.py
│   │   ├── device_utils.py
│   │   ├── exceptions.py
│   │   ├── logging_utils.py
│   │   ├── metrics.py
│   │   ├── model_io.py
│   │   ├── retry_utils.py
│   │   └── sensor_utils.py
│   └── visualization
│       ├── __pycache__
│       │   ├── prediction_plots.cpython-311.pyc
│       │   └── visualization_utils.cpython-311.pyc
│       └── visualization_utils.py
├── tests
│   ├── README.md
│   ├── __pycache__
│   │   ├── test_api_interface_baseline.cpython-311-pytest-8.3.5.pyc
│   │   ├── test_api_interface_baseline.cpython-311.pyc
│   │   ├── test_config_baseline.cpython-311-pytest-8.3.5.pyc
│   │   ├── test_config_baseline.cpython-311.pyc
│   │   ├── test_integration_baseline.cpython-311-pytest-8.3.5.pyc
│   │   ├── test_integration_baseline.cpython-311.pyc
│   │   ├── test_model_baseline.cpython-311-pytest-8.3.5.pyc
│   │   ├── test_model_baseline.cpython-311.pyc
│   │   ├── test_processing_baseline.cpython-311-pytest-8.3.5.pyc
│   │   └── test_processing_baseline.cpython-311.pyc
│   ├── reports
│   │   ├── baseline_test_report_20250425_105159.json
│   │   ├── baseline_test_report_20250425_115516.json
│   │   ├── baseline_test_report_20250425_121715.json
│   │   ├── baseline_test_report_20250425_122152.json
│   │   ├── baseline_test_report_20250425_122658.json
│   │   ├── baseline_test_report_20250425_145159.json
│   │   ├── baseline_test_report_20250425_145957.json
│   │   ├── baseline_test_report_20250425_150012.json
│   │   ├── baseline_test_report_20250425_150244.json
│   │   ├── baseline_test_report_20250425_162759.json
│   │   ├── baseline_test_report_20250425_223522.json
│   │   ├── baseline_test_report_20250425_223539.json
│   │   └── baseline_test_report_20250425_224442.json
│   ├── run_baseline_tests.py
│   ├── test_api_interface_baseline.py
│   ├── test_config_baseline.py
│   ├── test_integration_baseline.py
│   ├── test_model_baseline.py
│   └── test_processing_baseline.py
└── tune_model.py

41 directories, 194 files
