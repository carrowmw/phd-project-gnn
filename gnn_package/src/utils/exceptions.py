# src/utils/exceptions.py

import logging
import functools
import inspect
from typing import Any, Callable, Type, Optional, Union, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GNNException(Exception):
    """Base exception class for all GNN package exceptions"""

    pass


def safe_execute(
    func: Callable[..., T],
    error_msg: str = "Operation failed",
    exception_type: Type[Exception] = GNNException,
    fallback_value: Optional[T] = None,
    log_level: int = logging.ERROR,
) -> Union[T, Any]:
    """
    Execute a function with standardized error handling.

    Parameters:
    -----------
    func : Callable
        Function to execute
    error_msg : str
        Error message prefix
    exception_type : Type[Exception]
        Exception type to raise if an error occurs
    fallback_value : Any
        Value to return if an error occurs and exception_type is None
    log_level : int
        Logging level for error messages

    Returns:
    --------
    Any
        Result of the function or fallback value

    Raises:
    -------
    exception_type
        If an error occurs and exception_type is not None
    """
    try:
        return func()
    except Exception as e:
        logger.log(log_level, f"{error_msg}: {str(e)}")
        if exception_type:
            raise exception_type(f"{error_msg}: {str(e)}") from e
        return fallback_value


def handle_exceptions(
    exception_mapping: dict,
    default_exception: Type[Exception] = GNNException,
    error_msg: str = "Operation failed",
):
    """
    Decorator for handling exceptions with mapping to custom exception types.

    Parameters:
    -----------
    exception_mapping : dict
        Mapping from caught exception types to raised exception types
    default_exception : Type[Exception]
        Default exception type for unhandled exceptions
    error_msg : str
        Error message prefix

    Returns:
    --------
    Callable
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Find matching exception type
                for caught_type, raised_type in exception_mapping.items():
                    if isinstance(e, caught_type):
                        raise raised_type(f"{error_msg}: {str(e)}") from e

                # Default handling
                if default_exception:
                    raise default_exception(f"{error_msg}: {str(e)}") from e
                raise

        # Handle async functions
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Find matching exception type
                for caught_type, raised_type in exception_mapping.items():
                    if isinstance(e, caught_type):
                        raise raised_type(f"{error_msg}: {str(e)}") from e

                # Default handling
                if default_exception:
                    raise default_exception(f"{error_msg}: {str(e)}") from e
                raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# --- Data-related exceptions ---


class DataException(GNNException):
    """Base exception for data-related errors"""

    pass


class DataLoadError(DataException):
    """Error loading data from a source"""

    pass


class DataFormatError(DataException):
    """Error with data format or structure"""

    pass


class DataProcessingError(DataException):
    """Error during data processing or transformation"""

    pass


class DataValidationError(DataException):
    """Error validating data package"""

    pass


# --- Model-related exceptions ---


class ModelException(GNNException):
    """Base exception for model-related errors"""

    pass


class ModelCreationError(ModelException):
    """Error creating a model instance"""

    pass


class ModelLoadError(ModelException):
    """Error loading a model from file"""

    pass


class ModelParameterError(ModelException):
    """Error with model parameters or hyperparameters"""

    pass


class ModelPredictionError(ModelException):
    """Error during model prediction"""

    pass


class ModelEvaluationError(ModelException):
    """Error during model evaluation"""

    pass


# --- Configuration-related exceptions ---


class ConfigException(GNNException):
    """Base exception for configuration-related errors"""

    pass


class ConfigValidationError(ConfigException):
    """Configuration validation error"""

    pass


class ConfigLoadError(ConfigException):
    """Error loading configuration from file"""

    pass


# --- Training-related exceptions ---


class TrainingException(GNNException):
    """Base exception for training-related errors"""

    pass


class EarlyStoppingException(TrainingException):
    """Exception raised when early stopping is triggered"""

    pass


class ValidationError(TrainingException):
    """Exception during model validation"""

    pass


# --- API-related exceptions ---


class APIException(GNNException):
    """Base exception for API-related errors"""

    pass


class APIConnectionError(APIException):
    """Error connecting to API"""

    pass


class APIAuthenticationError(APIException):
    """API authentication error"""

    pass


class APIRateLimitError(APIException):
    """API rate limit exceeded"""

    pass


class APIRequestError(APIException):
    """Error with API request parameters or format"""

    pass
