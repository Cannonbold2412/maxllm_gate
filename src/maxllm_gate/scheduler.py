"""maxllm_gate Scheduler - orchestrates requests with edge case handling."""

import asyncio
import signal
import time
from typing import Any, AsyncGenerator

import tiktoken
import litellm

from maxllm_gate.config import maxllm_gate_config
from maxllm_gate.rate_limiter import RateLimiter


class CapacityExhaustedError(Exception):
    """Raised when all API keys are exhausted and max wait time exceeded."""
    
    def __init__(self, message: str, wait_time: float = 0.0):
        super().__init__(message)
        self.wait_time = wait_time


class CircuitBreakerOpenError(Exception):
    """Raised when all keys have circuit breakers open."""
    pass


class GracefulShutdown:
    """Handles graceful shutdown with queue draining."""
    
    def __init__(self):
        self._shutdown = False
        self._pending_requests: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
    
    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown
    
    def register_task(self, task: asyncio.Task) -> None:
        """Register an in-flight request task."""
        self._pending_requests.add(task)
        task.add_done_callback(self._pending_requests.discard)
    
    async def initiate_shutdown(self, timeout: float = 30.0) -> None:
        """
        Initiate graceful shutdown.
        
        Args:
            timeout: Max seconds to wait for pending requests
        """
        self._shutdown = True
        self._shutdown_event.set()
        
        if not self._pending_requests:
            return
        
        # Wait for pending requests with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._pending_requests, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in self._pending_requests:
                task.cancel()
    
    def setup_signal_handlers(self) -> None:
        """Setup SIGTERM/SIGINT handlers."""
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.initiate_shutdown()),
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass


class Scheduler:
    """Schedules and executes LLM requests with edge case handling."""
    
    def __init__(self, config: maxllm_gate_config, rate_limiter: RateLimiter):
        self.config = config
        self.rate_limiter = rate_limiter
        self._encoders: dict[str, tiktoken.Encoding] = {}
        self._shutdown = GracefulShutdown()
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._active_requests = 0
        self._total_requests = 0
        self._total_failures = 0
        self._queue_rejections = 0
        
        # Configure LiteLLM
        litellm.drop_params = True
    
    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        return self._active_requests
    
    @property
    def queue_utilization(self) -> float:
        """Current semaphore utilization (0.0 to 1.0)."""
        # Semaphore value = available slots
        available = self._semaphore._value
        max_concurrent = self.config.max_concurrent_requests
        return 1.0 - (available / max_concurrent) if max_concurrent > 0 else 0.0
    
    def can_accept_request(self) -> bool:
        """Check if scheduler can accept a new request without blocking."""
        return self._semaphore._value > 0 and not self._shutdown.is_shutting_down
    
    def queue_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "active_requests": self._active_requests,
            "max_concurrent": self.config.max_concurrent_requests,
            "queue_utilization": self.queue_utilization,
            "total_processed": self._total_requests,
            "total_failures": self._total_failures,
            "queue_rejections": self._queue_rejections,
            "can_accept": self.can_accept_request(),
        }
    
    def _estimate_tokens(self, messages: list[dict], model: str) -> int:
        """Estimate token count for messages."""
        try:
            if model not in self._encoders:
                try:
                    self._encoders[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    self._encoders[model] = tiktoken.get_encoding("cl100k_base")
            
            encoder = self._encoders[model]
            total = 0
            
            for msg in messages:
                total += 4  # Message overhead
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += len(encoder.encode(content))
            
            total += 3  # Priming
            return total
            
        except Exception:
            # Fallback: ~4 chars per token
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            return max(1, total_chars // 4)
    
    def _estimate_output(self, input_tokens: int, max_tokens: int | None) -> int:
        """Estimate output tokens."""
        estimated = int(input_tokens * 1.5)
        if max_tokens:
            estimated = min(estimated, max_tokens)
        return max(1, estimated)
    
    async def schedule(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        priority: str = "medium",
        timeout: float = 120.0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Schedule and execute a request with concurrency control.
        
        Uses a semaphore to limit concurrent LLM requests, preventing
        connection pool exhaustion and rate limiter race conditions.
        
        Raises:
            CapacityExhaustedError: If all keys exhausted and wait time exceeds limit
            CircuitBreakerOpenError: If all keys have circuit breakers open
            RuntimeError: If scheduler is shutting down
        """
        
        # Check shutdown
        if self._shutdown.is_shutting_down:
            raise RuntimeError("Scheduler is shutting down, not accepting new requests")
        
        # Acquire semaphore to limit concurrent requests
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.max_wait_time,
            )
        except asyncio.TimeoutError:
            self._queue_rejections += 1
            raise CapacityExhaustedError(
                f"Queue full: {self.config.max_concurrent_requests} concurrent requests. "
                f"Wait time exceeded {self.config.max_wait_time}s."
            )
        
        self._active_requests += 1
        try:
            result = await self._execute_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                priority=priority,
                timeout=timeout,
                **kwargs,
            )
            self._total_requests += 1
            return result
        except Exception:
            self._total_failures += 1
            raise
        finally:
            self._active_requests -= 1
            self._semaphore.release()
    
    async def _execute_request(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        priority: str = "medium",
        timeout: float = 120.0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute a single request with edge case handling.
        
        Handles:
        - Max wait time enforcement
        - Circuit breaker pattern
        - Request timeout
        - Fallback to different keys on failure
        """
        request_start = time.time()
        
        # Estimate tokens with safety buffer
        input_tokens = self._estimate_tokens(messages, model)
        output_tokens = self._estimate_output(input_tokens, max_tokens)
        total_estimate = int((input_tokens + output_tokens) * self.config.token_buffer)
        
        # Select key with circuit breaker awareness
        key_state, wait_time = self.rate_limiter.select_key(
            model=model,
            estimated_tokens=total_estimate,
            strategy=self.config.strategy,
            circuit_threshold=self.config.circuit_breaker_threshold,
            circuit_timeout=self.config.circuit_breaker_timeout,
        )
        
        if key_state is None:
            raise CapacityExhaustedError(
                f"No available keys for model: {model}. All keys exhausted or circuit breakers open."
            )
        
        # EDGE CASE: Max wait time enforcement
        if wait_time > self.config.max_wait_time:
            raise CapacityExhaustedError(
                f"Wait time ({wait_time:.1f}s) exceeds max_wait_time ({self.config.max_wait_time}s). "
                f"System overloaded.",
                wait_time=wait_time,
            )
        
        # Wait if needed (deferred execution) with timeout
        if wait_time > 0:
            try:
                await asyncio.wait_for(
                    asyncio.sleep(wait_time),
                    timeout=self.config.max_wait_time,
                )
            except asyncio.TimeoutError:
                raise CapacityExhaustedError(
                    f"Timeout waiting for capacity after {self.config.max_wait_time}s"
                )
        
        # Consume capacity
        self.rate_limiter.consume(key_state.key_config.key_id, total_estimate)
        
        # Build LiteLLM kwargs
        litellm_model = self._get_litellm_model(model, key_state.key_config.provider)
        
        completion_kwargs = {
            "model": litellm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "api_key": key_state.key_config.api_key,
            "timeout": min(timeout, self.config.request_timeout),  # Enforce timeout
            **kwargs,
        }
        
        # Track keys tried for fallback
        tried_keys = {key_state.key_config.key_id}
        current_key = key_state
        
        # Execute with retry and fallback
        last_error = None
        total_attempts = 0
        max_total_attempts = self.config.max_retries * 2  # Allow some fallback attempts
        
        while total_attempts < max_total_attempts:
            total_attempts += 1
            
            try:
                # EDGE CASE: Request timeout enforcement
                llm_start = time.time()
                response = await asyncio.wait_for(
                    litellm.acompletion(**completion_kwargs),
                    timeout=min(timeout, self.config.request_timeout),
                )
                request_latency = time.time() - llm_start
                
                result = response.model_dump()
                
                # Record success with latency
                usage = result.get("usage", {})
                actual_tokens = usage.get("total_tokens", total_estimate)
                self.rate_limiter.record_success(
                    current_key.key_config.key_id,
                    actual_tokens,
                    latency=request_latency,
                )
                
                result["_key_used"] = current_key.key_config.key_id
                result["_latency"] = request_latency
                result["_attempts"] = total_attempts
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Request timeout after {timeout}s")
                self.rate_limiter.record_failure(current_key.key_config.key_id)
                
            except Exception as e:
                last_error = e
                self.rate_limiter.record_failure(current_key.key_config.key_id)
                
                # Check if it's a rate limit error - try different key
                error_str = str(e).lower()
                is_rate_limit = any(x in error_str for x in ["429", "rate limit", "quota"])
                is_server_error = any(x in error_str for x in ["500", "502", "503", "504"])
                
                if is_rate_limit or is_server_error:
                    # EDGE CASE: Try fallback to different key
                    fallback_key, _ = self.rate_limiter.select_key(
                        model=model,
                        estimated_tokens=total_estimate,
                        strategy=self.config.strategy,
                        exclude_keys=tried_keys,
                    )
                    
                    if fallback_key and fallback_key.key_config.key_id not in tried_keys:
                        tried_keys.add(fallback_key.key_config.key_id)
                        current_key = fallback_key
                        completion_kwargs["api_key"] = fallback_key.key_config.api_key
                        completion_kwargs["model"] = self._get_litellm_model(
                            model, fallback_key.key_config.provider
                        )
                        continue  # Try immediately with new key
            
            # Exponential backoff before retry
            if total_attempts < max_total_attempts:
                delay = self.config.retry_base_delay * (2 ** (total_attempts - 1))
                delay = min(delay, self.config.retry_max_delay)
                await asyncio.sleep(delay)
        
        raise last_error or RuntimeError("Request failed")
    
    async def schedule_stream(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Schedule and stream a request with concurrency control.
        
        Yields content chunks as they arrive from the LLM provider.
        Tracks actual token usage from stream for accurate rate limiting.
        
        Raises:
            CapacityExhaustedError: If all keys exhausted and wait time exceeds limit
            RuntimeError: If scheduler is shutting down
        """
        
        # Check shutdown
        if self._shutdown.is_shutting_down:
            raise RuntimeError("Scheduler is shutting down")
        
        # Acquire semaphore for concurrency control with timeout
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.max_wait_time,
            )
        except asyncio.TimeoutError:
            self._queue_rejections += 1
            raise CapacityExhaustedError(
                f"Queue full for streaming. Wait time exceeded {self.config.max_wait_time}s."
            )
        
        self._active_requests += 1
        try:
            async for chunk in self._stream_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            ):
                yield chunk
            self._total_requests += 1
        except Exception:
            self._total_failures += 1
            raise
        finally:
            self._active_requests -= 1
            self._semaphore.release()
    
    async def _stream_request(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming request with edge case handling."""
        
        input_tokens = self._estimate_tokens(messages, model)
        output_tokens = self._estimate_output(input_tokens, max_tokens)
        total_estimate = int((input_tokens + output_tokens) * self.config.token_buffer)
        
        key_state, wait_time = self.rate_limiter.select_key(
            model=model,
            estimated_tokens=total_estimate,
            strategy=self.config.strategy,
            circuit_threshold=self.config.circuit_breaker_threshold,
            circuit_timeout=self.config.circuit_breaker_timeout,
        )
        
        if key_state is None:
            raise CapacityExhaustedError(
                f"No available keys for model: {model}. All keys exhausted."
            )
        
        # EDGE CASE: Max wait time for streaming
        if wait_time > self.config.max_wait_time:
            raise CapacityExhaustedError(
                f"Wait time ({wait_time:.1f}s) exceeds max_wait_time for streaming.",
                wait_time=wait_time,
            )
        
        if wait_time > 0:
            try:
                await asyncio.wait_for(
                    asyncio.sleep(wait_time),
                    timeout=self.config.max_wait_time,
                )
            except asyncio.TimeoutError:
                raise CapacityExhaustedError(
                    f"Timeout waiting for capacity for streaming"
                )
        
        # Reserve estimated tokens upfront
        self.rate_limiter.consume(key_state.key_config.key_id, total_estimate)
        
        litellm_model = self._get_litellm_model(model, key_state.key_config.provider)
        
        request_start = time.time()
        output_content = []
        
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=litellm_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=key_state.key_config.api_key,
                    stream=True,
                    **kwargs,
                ),
                timeout=self.config.request_timeout,
            )
            
            async for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content = delta.content
                        output_content.append(content)
                        yield content
                    
                    # Check for final usage in stream
                    if hasattr(chunk, "usage") and chunk.usage:
                        actual_tokens = chunk.usage.total_tokens
                        # Adjust if actual differs from estimate
                        diff = actual_tokens - total_estimate
                        if diff > 0:
                            self.rate_limiter.consume(key_state.key_config.key_id, diff)
            
            request_latency = time.time() - request_start
            self.rate_limiter.record_success(
                key_state.key_config.key_id,
                total_estimate,
                latency=request_latency,
            )
            
        except Exception:
            self.rate_limiter.record_failure(key_state.key_config.key_id)
            raise
    
    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the scheduler.
        
        Waits for in-flight requests to complete.
        
        Args:
            timeout: Max seconds to wait for pending requests
        """
        await self._shutdown.initiate_shutdown(timeout)
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        self._shutdown.setup_signal_handlers()
    
    def _get_litellm_model(self, model: str, provider: str) -> str:
        """Get LiteLLM model string with provider prefix."""
        if "/" in model:
            return model
        
        if provider == "groq":
            return f"groq/{model}"
        elif provider == "openrouter":
            return f"openrouter/{model}"
        elif provider == "anthropic":
            return f"anthropic/{model}"
        elif provider == "together":
            return f"together_ai/{model}"
        else:
            return model
    
    def status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "strategy": self.config.strategy,
            "capacity": self.rate_limiter.capacity(),
        }
