"""Retry utilities with exponential backoff."""

import asyncio
import random
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

from llm_scheduler.config import settings


P = ParamSpec("P")
T = TypeVar("T")


class RetryError(Exception):
    """All retries exhausted."""
    
    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Current attempt number (1-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        jitter: Whether to add random jitter
        
    Returns:
        Delay in seconds
    """
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    
    if jitter:
        # Add 0-25% random jitter
        delay = delay * (1 + random.random() * 0.25)
    
    return delay


def retry_with_backoff(
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay cap
        retryable_exceptions: Exceptions that trigger retry
        on_retry: Callback on retry (attempt, exception)
    """
    _max_attempts = max_attempts or settings.max_retries
    _base_delay = base_delay or settings.retry_base_delay
    _max_delay = max_delay or settings.retry_max_delay
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None
            
            for attempt in range(1, _max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_error = e
                    
                    if attempt >= _max_attempts:
                        break
                    
                    delay = calculate_backoff(
                        attempt,
                        _base_delay,
                        _max_delay,
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    await asyncio.sleep(delay)
            
            raise RetryError(
                f"Failed after {_max_attempts} attempts",
                last_error=last_error,
            )
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import time
            
            last_error: Exception | None = None
            
            for attempt in range(1, _max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_error = e
                    
                    if attempt >= _max_attempts:
                        break
                    
                    delay = calculate_backoff(
                        attempt,
                        _base_delay,
                        _max_delay,
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(delay)
            
            raise RetryError(
                f"Failed after {_max_attempts} attempts",
                last_error=last_error,
            )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
