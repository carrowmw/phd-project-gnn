# src/utils/exceptions.py
class GNNException(Exception):
    """Base exception class for all GNN package exceptions"""

    pass


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
