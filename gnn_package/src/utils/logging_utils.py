# src/utils/logging_utils.py (Enhanced version)

import os
import sys
import logging
import time
import asyncio
import contextlib
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime

# Standard log message patterns
LOG_PATTERNS = {
    "start_operation": "Starting {operation_name}",
    "end_operation": "Completed {operation_name} in {duration:.2f}s",
    "config_loaded": "Configuration loaded from {config_path}",
    "data_loaded": "Loaded {num_records} records from {source}",
    "error_occurred": "Error in {context}: {error}",
}


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


@contextlib.contextmanager
def log_operation(logger, operation_name, level=logging.INFO):
    """
    Context manager for logging operations with timing.

    Parameters:
    -----------
    logger : logging.Logger
        Logger to use
    operation_name : str
        Name of the operation being performed
    level : int
        Logging level

    Yields:
    -------
    None
    """
    start_time = time.time()
    logger.log(
        level, LOG_PATTERNS["start_operation"].format(operation_name=operation_name)
    )
    try:
        yield
    except Exception as e:
        logger.log(
            logging.ERROR,
            LOG_PATTERNS["error_occurred"].format(context=operation_name, error=str(e)),
        )
        raise
    finally:
        duration = time.time() - start_time
        logger.log(
            level,
            LOG_PATTERNS["end_operation"].format(
                operation_name=operation_name, duration=duration
            ),
        )


def log_function(level=logging.INFO, show_args=False):
    """
    Decorator to log function calls with timing.

    Parameters:
    -----------
    level : int
        Logging level
    show_args : bool
        Whether to include function arguments in log

    Returns:
    --------
    Callable
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            func_name = func.__name__

            if show_args:
                args_str = ", ".join([f"{arg}" for arg in args[1:]])  # Skip self
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                params = (
                    f"{args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str}"
                )
                operation_name = f"{func_name}({params})"
            else:
                operation_name = func_name

            with log_operation(logger, operation_name, level=level):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            func_name = func.__name__

            if show_args:
                args_str = ", ".join([f"{arg}" for arg in args[1:]])  # Skip self
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                params = (
                    f"{args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str}"
                )
                operation_name = f"{func_name}({params})"
            else:
                operation_name = func_name

            with log_operation(logger, operation_name, level=level):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
