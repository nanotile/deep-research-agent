"""
Retry handler with exponential backoff for API calls.
Provides both sync and async decorators.
"""

import asyncio
import functools
import random
import time
from typing import Callable, Type, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RetryExhausted(Exception):
    """Raised when all retry attempts have been exhausted."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for synchronous functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt_number) called on each retry

    Example:
        @retry_with_backoff(max_retries=3, exceptions=(httpx.HTTPError,))
        def fetch_data(url):
            return httpx.get(url)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise RetryExhausted(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        ) from e

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter (0-25% of delay)
                    if jitter:
                        delay = delay * (1 + random.uniform(0, 0.25))

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(delay)

            # Should not reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for async functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt_number) called on each retry

    Example:
        @async_retry_with_backoff(max_retries=3, exceptions=(httpx.HTTPError,))
        async def fetch_data(url):
            async with httpx.AsyncClient() as client:
                return await client.get(url)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise RetryExhausted(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        ) from e

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter (0-25% of delay)
                    if jitter:
                        delay = delay * (1 + random.uniform(0, 0.25))

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    await asyncio.sleep(delay)

            # Should not reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


# Pre-configured decorators for common use cases
retry_api_call = async_retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exceptions=(Exception,),  # Will be narrowed in actual usage
)

retry_http_call = async_retry_with_backoff(
    max_retries=3,
    base_delay=0.5,
    max_delay=15.0,
    exceptions=(Exception,),  # httpx.HTTPError, httpx.TimeoutException
)
