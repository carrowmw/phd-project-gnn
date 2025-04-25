# src/utils/logging_utils.py
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime


def configure_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    module_levels: Optional[Dict[str, int]] = None,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Configure logging for the application.

    Parameters:
    -----------
    log_level : int
        Default logging level for the root logger
    log_file : str or Path, optional
        Path to log file. If None, logs to console only.
    log_format : str, optional
        Format string for log messages. If None, uses default format.
    module_levels : Dict[str, int], optional
        Dictionary mapping module names to specific log levels
    include_timestamp : bool
        Whether to include timestamp in log filename

    Returns:
    --------
    logging.Logger
        Configured root logger
    """
    # Create default log format if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file provided
    if log_file:
        log_path = Path(log_file)

        # Add timestamp to filename if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = log_path.stem
            suffix = log_path.suffix
            log_path = log_path.with_name(f"{stem}_{timestamp}{suffix}")

        # Create directory if it doesn't exist
        os.makedirs(log_path.parent, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific levels for modules if provided
    if module_levels:
        for module_name, level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(level)

    return root_logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Parameters:
    -----------
    name : str
        Name of the logger (typically __name__)
    level : int, optional
        Logging level. If None, uses parent logger level.

    Returns:
    --------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
