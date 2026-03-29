"""LiteLLM dispatcher with retry and backoff."""

from typing import Any, AsyncGenerator

import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from llm_scheduler.config import settings
from llm_scheduler.core.queue_manager import QueuedRequest
from llm_scheduler.rate_limiting.key_manager import KeyManager, KeySelection
from llm_scheduler.observability.logging import get_logger


class DispatchError(Exception):
    """Error during request dispatch."""
    
    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class Dispatcher:
    """
    Dispatches requests to LLM providers via LiteLLM.
    
    Handles:
    - LiteLLM SDK calls
    - Retry with exponential backoff
    - Streaming responses
    - Error handling and classification
    """
    
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self._logger = get_logger()
        
        # Configure LiteLLM
        litellm.drop_params = True  # Drop unsupported params
        litellm.set_verbose = settings.debug
    
    async def dispatch(
        self,
        request: QueuedRequest,
        key_selection: KeySelection,
    ) -> dict[str, Any]:
        """
        Dispatch a request to LiteLLM.
        
        Args:
            request: The queued request
            key_selection: Selected API key configuration
            
        Returns:
            LiteLLM response dict
        """
        litellm_kwargs = key_selection.to_litellm_kwargs(request.model)
        
        # Build completion kwargs
        completion_kwargs = {
            **litellm_kwargs,
            "messages": request.messages,
            "max_tokens": request.max_tokens or settings.default_max_tokens,
            "temperature": request.temperature,
            **request.extra_params,
        }
        
        self._logger.debug(
            "Dispatching request",
            request_id=request.request_id,
            model=completion_kwargs.get("model"),
            key_id=key_selection.key_id,
        )
        
        try:
            response = await self._execute_with_retry(completion_kwargs)
            
            # Extract actual token usage
            usage = response.get("usage", {})
            actual_tokens = (
                usage.get("total_tokens") or
                usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            )
            
            # Record success
            self.key_manager.record_success(key_selection.key_id, actual_tokens)
            
            self._logger.info(
                "Request completed",
                request_id=request.request_id,
                tokens_used=actual_tokens,
                model=response.get("model"),
            )
            
            return response
            
        except Exception as e:
            self._handle_error(e, request, key_selection)
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(DispatchError),
    )
    async def _execute_with_retry(
        self,
        completion_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute LiteLLM completion with retry."""
        try:
            response = await litellm.acompletion(**completion_kwargs)
            return response.model_dump()
            
        except litellm.RateLimitError as e:
            raise DispatchError(
                str(e),
                status_code=429,
                retryable=True,
            )
        except litellm.APIConnectionError as e:
            raise DispatchError(
                str(e),
                status_code=503,
                retryable=True,
            )
        except litellm.ServiceUnavailableError as e:
            raise DispatchError(
                str(e),
                status_code=503,
                retryable=True,
            )
        except litellm.APIError as e:
            status = getattr(e, "status_code", None)
            retryable = status in self.RETRYABLE_STATUS_CODES if status else False
            raise DispatchError(
                str(e),
                status_code=status,
                retryable=retryable,
            )
        except Exception as e:
            raise DispatchError(str(e), retryable=False)
    
    async def dispatch_stream(
        self,
        request: QueuedRequest,
        key_selection: KeySelection,
    ) -> AsyncGenerator[str, None]:
        """
        Dispatch a streaming request.
        
        Yields:
            Chunks of the response content
        """
        litellm_kwargs = key_selection.to_litellm_kwargs(request.model)
        
        completion_kwargs = {
            **litellm_kwargs,
            "messages": request.messages,
            "max_tokens": request.max_tokens or settings.default_max_tokens,
            "temperature": request.temperature,
            "stream": True,
            **request.extra_params,
        }
        
        self._logger.debug(
            "Dispatching streaming request",
            request_id=request.request_id,
            model=completion_kwargs.get("model"),
            key_id=key_selection.key_id,
        )
        
        try:
            response = await litellm.acompletion(**completion_kwargs)
            
            total_tokens = 0
            async for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        total_tokens += 1  # Approximate
                        yield delta.content
            
            # Record approximate success
            self.key_manager.record_success(
                key_selection.key_id,
                request.estimated_tokens,  # Use estimate for streaming
            )
            
        except Exception as e:
            self._handle_error(e, request, key_selection)
            raise
    
    def _handle_error(
        self,
        error: Exception,
        request: QueuedRequest,
        key_selection: KeySelection,
    ) -> None:
        """Handle and log dispatch errors."""
        status_code = getattr(error, "status_code", None)
        
        self.key_manager.record_failure(key_selection.key_id, status_code)
        
        self._logger.error(
            "Dispatch failed",
            request_id=request.request_id,
            key_id=key_selection.key_id,
            error=str(error),
            status_code=status_code,
        )
        
        # Refund capacity if we didn't actually use tokens
        if status_code in {429, 500, 502, 503, 504}:
            self.key_manager.refund_capacity(
                key_selection.key_id,
                request.estimated_tokens,
            )
