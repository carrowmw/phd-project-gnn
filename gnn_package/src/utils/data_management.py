# src/utils/data_management.py
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

def generate_preprocessing_id(raw_data_path: Union[str, Path], config) -> str:
    """
    Generate a unique identifier for preprocessed data based on raw data path and config.

    This creates a reproducible hash combining the raw data path and relevant preprocessing
    parameters from the configuration to uniquely identify preprocessing results.
    """
    # Convert path to string and normalize
    raw_path_str = str(Path(raw_data_path).absolute())

    # Extract relevant parameters that affect preprocessing
    relevant_params = {
        "window_size": config.data.general.window_size,
        "horizon": config.data.general.horizon,
        "stride": config.data.general.stride,
        "standardize": config.data.general.standardize,
        "resampling_frequency": config.data.general.resampling_frequency,
        "missing_value": config.data.general.missing_value,
        "graph_prefix": config.data.general.graph_prefix
    }

    # Create a stable string representation of parameters
    param_str = json.dumps(relevant_params, sort_keys=True)

    # Combine raw data path and parameters to create unique hash
    combined_str = f"{raw_path_str}:{param_str}"
    hash_id = hashlib.md5(combined_str.encode()).hexdigest()

    return hash_id

def get_preprocessed_path(raw_data_path: Union[str, Path], config, output_dir: Optional[Path] = None) -> Path:
    """
    Generate the expected path for preprocessed data based on raw data and config.

    Parameters:
    -----------
    raw_data_path : str or Path
        Path to the raw data file
    config : ExperimentConfig
        Configuration object
    output_dir : Path, optional
        Directory to save preprocessed data (defaults to standard location)

    Returns:
    --------
    Path
        Path where preprocessed data would be stored
    """
    # Generate unique identifier
    preprocessing_id = generate_preprocessing_id(raw_data_path, config)

    # Determine base directory
    if output_dir is None:
        base_dir = Path("data/preprocessed/timeseries")
    else:
        base_dir = output_dir

    # Create directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)

    # Get raw filename without extension
    raw_filename = Path(raw_data_path).stem

    # Construct final path
    return base_dir / f"{raw_filename}_{preprocessing_id}.pkl"

def get_metadata_path(preprocessed_path: Union[str, Path]) -> Path:
    """Get the path for metadata file associated with preprocessed data."""
    preprocessed_path = Path(preprocessed_path)
    return preprocessed_path.with_name(f"{preprocessed_path.stem}_metadata.json")

def save_preprocessing_metadata(raw_data_path: Union[str, Path],
                               preprocessed_path: Union[str, Path],
                               config) -> None:
    """
    Save metadata about preprocessing to accompany the preprocessed data.

    This metadata records the relationship between raw and preprocessed data,
    along with the configuration parameters used for preprocessing.
    """
    # Convert paths to absolute paths
    raw_data_path = Path(raw_data_path).absolute()
    preprocessed_path = Path(preprocessed_path).absolute()

    # Gather metadata
    metadata = {
        "raw_data_path": str(raw_data_path),
        "preprocessed_path": str(preprocessed_path),
        "preprocessing_timestamp": datetime.now().isoformat(),
        "config_parameters": {
            "window_size": config.data.general.window_size,
            "horizon": config.data.general.horizon,
            "stride": config.data.general.stride,
            "standardize": config.data.general.standardize,
            "resampling_frequency": config.data.general.resampling_frequency,
            "missing_value": config.data.general.missing_value,
            "graph_prefix": config.data.general.graph_prefix
        },
        "version": "1.0.0"  # For future compatibility
    }

    # Save metadata
    metadata_path = get_metadata_path(preprocessed_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved preprocessing metadata to {metadata_path}")

def is_preprocessed_data_package(data: Any) -> bool:
    """
    Check if a data object appears to be a preprocessed data package.

    Parameters:
    -----------
    data : Any
        Data object to check

    Returns:
    --------
    bool
        True if the data has the structure of a preprocessed data package
    """
    # Check for expected structure of preprocessed data
    if not isinstance(data, dict):
        return False

    required_keys = ['data_loaders', 'graph_data', 'time_series', 'metadata']
    return all(key in data for key in required_keys)

def is_preprocessed_data_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file contains preprocessed data rather than raw time series.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file to check

    Returns:
    --------
    bool
        True if the file contains preprocessed data
    """
    file_path = Path(file_path)

    # First, check for metadata file as a quick way to identify preprocessed data
    metadata_path = get_metadata_path(file_path)
    if metadata_path.exists():
        return True

    # If no metadata, try to load the file and check its structure
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return is_preprocessed_data_package(data)
    except Exception as e:
        logger.warning(f"Error checking if {file_path} is preprocessed data: {e}")
        return False

def find_preprocessed_data(raw_data_path: Union[str, Path], config) -> Optional[Path]:
    """
    Look for existing preprocessed data for a raw data file and configuration.

    Parameters:
    -----------
    raw_data_path : str or Path
        Path to the raw data file
    config : ExperimentConfig
        Configuration object

    Returns:
    --------
    Optional[Path]
        Path to preprocessed data if found, None otherwise
    """
    # Generate expected path for preprocessed data
    expected_path = get_preprocessed_path(raw_data_path, config)

    # Check if preprocessed data and metadata exist
    metadata_path = get_metadata_path(expected_path)

    if expected_path.exists() and metadata_path.exists():
        # Verify that metadata matches current configuration
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Get stored parameters
            stored_params = metadata.get("config_parameters", {})

            # Check key parameters match
            if (stored_params.get("window_size") == config.data.general.window_size and
                stored_params.get("horizon") == config.data.general.horizon and
                stored_params.get("standardize") == config.data.general.standardize):

                logger.info(f"Found matching preprocessed data: {expected_path}")
                return expected_path

            logger.info(f"Found preprocessed data but parameters don't match")
        except Exception as e:
            logger.warning(f"Error validating metadata for {expected_path}: {e}")

    return None

def save_preprocessed_data(data_package: Dict[str, Any],
                          raw_data_path: Union[str, Path],
                          preprocessed_path: Union[str, Path],
                          config) -> None:
    """
    Save preprocessed data package along with its metadata.

    Parameters:
    -----------
    data_package : Dict[str, Any]
        Preprocessed data package to save
    raw_data_path : str or Path
        Path to the raw data file
    preprocessed_path : str or Path
        Path where to save preprocessed data
    config : ExperimentConfig
        Configuration object used for preprocessing
    """
    preprocessed_path = Path(preprocessed_path)

    # Create directory if it doesn't exist
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the preprocessed data
    with open(preprocessed_path, 'wb') as f:
        pickle.dump(data_package, f)

    logger.info(f"Saved preprocessed data to {preprocessed_path}")

    # Save metadata
    save_preprocessing_metadata(raw_data_path, preprocessed_path, config)