# src/utils/retry_utils.py
import time
import asyncio
import functools
import logging
from typing import Callable, Any, Type, Union, Optional, Tuple, List, Set

logger = logging.getLogger(__name__)


def retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: Optional[float] = None,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    retry_if: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator for retrying a function on specified exceptions.

    Parameters:
    -----------
    max_retries : int
        Maximum number of retries
    retry_delay : float
        Initial delay between retries in seconds
    backoff_factor : float
        Factor by which the delay increases for each retry
    max_delay : float, optional
        Maximum delay between retries in seconds
    exceptions : Exception or tuple of Exceptions
        Exceptions to catch and retry on
    retry_if : Callable[[Exception], bool], optional
        Function that determines if the exception should trigger a retry
    on_retry : Callable[[int, Exception], None], optional
        Function called on each retry with retry count and exception

    Returns:
    --------
    Callable
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = retry_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1

                    # Check if we should retry
                    if attempts >= max_retries or (retry_if and not retry_if(e)):
                        logger.error(f"Failed after {attempts} attempts: {str(e)}")
                        raise

                    # Calculate next delay
                    if on_retry:
                        on_retry(attempts, e)

                    logger.warning(
                        f"Retry {attempts}/{max_retries} after error: {str(e)}. "
                        f"Waiting {delay:.2f} seconds..."
                    )

                    # Wait before retry
                    time.sleep(delay)

                    # Calculate next delay with backoff
                    delay *= backoff_factor
                    if max_delay is not None:
                        delay = min(delay, max_delay)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            attempts = 0
            delay = retry_delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1

                    # Check if we should retry
                    if attempts >= max_retries or (retry_if and not retry_if(e)):
                        logger.error(f"Failed after {attempts} attempts: {str(e)}")
                        raise

                    # Calculate next delay
                    if on_retry:
                        on_retry(attempts, e)

                    logger.warning(
                        f"Retry {attempts}/{max_retries} after error: {str(e)}. "
                        f"Waiting {delay:.2f} seconds..."
                    )

                    # Wait before retry
                    await asyncio.sleep(delay)

                    # Calculate next delay with backoff
                    delay *= backoff_factor
                    if max_delay is not None:
                        delay = min(delay, max_delay)

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator
