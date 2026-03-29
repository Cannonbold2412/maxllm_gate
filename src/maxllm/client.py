"""MAXLLM Client - Simple SDK interface."""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, AsyncGenerator

from maxllm.config import MAXLLMConfig, KeyConfig
from maxllm.scheduler import Scheduler
from maxllm.rate_limiter import RateLimiter
from maxllm.validation import validate_chat_request


@dataclass
class ChatResponse:
    """Response from chat completion."""
    
    content: str
    model: str
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    id: str | None = None
    
    # Timing info
    latency: float | None = None      # Total client-side latency
    llm_latency: float | None = None  # LLM provider latency only
    queue_time: float | None = None
    key_used: str | None = None
    
    def __str__(self) -> str:
        return self.content


@dataclass
class Message:
    """Chat message."""
    
    role: str
    content: str
    
    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class MAXLLM:
    """
    MAXLLM - Intelligent LLM client with built-in rate limiting.
    
    This client automatically:
    - Manages multiple API keys across providers
    - Tracks TPM/RPM limits per key
    - Routes requests to available keys
    - Queues and defers requests when at capacity
    - Retries on transient failures
    
    Usage:
        # From config file
        client = MAXLLM.from_config("config.yaml")
        
        # Simple chat
        response = client.chat("gpt-4o-mini", "Hello!")
        print(response.content)
        
        # With messages
        response = client.chat("gpt-4o-mini", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi!"},
        ])
    """
    
    def __init__(
        self,
        config: MAXLLMConfig | None = None,
        keys: list[dict] | None = None,
        **kwargs,
    ):
        """
        Initialize MAXLLM client.
        
        Args:
            config: MAXLLMConfig object
            keys: List of key configurations (alternative to config)
            **kwargs: Additional config options
        """
        if config is not None:
            self.config = config
        elif keys is not None:
            key_configs = [KeyConfig.from_dict(k) for k in keys]
            self.config = MAXLLMConfig(keys=key_configs, **kwargs)
        else:
            self.config = MAXLLMConfig(**kwargs)
        
        self._rate_limiter = RateLimiter()
        self._scheduler = Scheduler(self.config, self._rate_limiter)
        self._initialized = False
    
    @classmethod
    def from_config(cls, path: str | Path) -> "MAXLLM":
        """
        Create client from config file.
        
        Args:
            path: Path to YAML or JSON config file
            
        Returns:
            Configured MAXLLM client
        """
        config = MAXLLMConfig.from_file(path)
        return cls(config=config)
    
    @classmethod
    def from_env(cls) -> "MAXLLM":
        """Create client from environment variables."""
        config = MAXLLMConfig.from_env()
        return cls(config=config)
    
    def _ensure_initialized(self) -> None:
        """Initialize rate limiter with keys."""
        if self._initialized:
            return
        
        for key in self.config.keys:
            self._rate_limiter.register_key(key)
        
        self._initialized = True
    
    def chat(
        self,
        model: str,
        messages: str | list[dict[str, str]] | list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        priority: str = "medium",
        timeout: float = 120.0,
        validate: bool = True,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a chat completion request.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini", "mixtral-8x7b-32768")
            messages: Either a string (converted to user message) or list of messages
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            priority: Request priority (high/medium/low)
            timeout: Request timeout in seconds
            validate: Whether to validate inputs (default True)
            **kwargs: Additional parameters passed to LiteLLM
            
        Returns:
            ChatResponse with content and metadata
            
        Raises:
            ValueError: If validation fails
            RuntimeError: If no available keys or request fails
        """
        self._ensure_initialized()
        
        # Normalize messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif messages and isinstance(messages[0], Message):
            messages = [m.to_dict() for m in messages]
        
        # Validate if enabled
        if validate:
            request = validate_chat_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                priority=priority,
                timeout=timeout,
                **kwargs,
            )
            max_tokens = request.max_tokens
            temperature = request.temperature
        else:
            # Set defaults
            max_tokens = max_tokens or self.config.default_max_tokens
            temperature = temperature if temperature is not None else self.config.default_temperature
        
        # Run async in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # We're already in an async context - use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._chat_async(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    priority=priority,
                    timeout=timeout,
                    **kwargs,
                ),
                loop,
            )
            return future.result(timeout=timeout + 5)
        else:
            # No running loop - create one
            return asyncio.run(
                self._chat_async(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    priority=priority,
                    timeout=timeout,
                    **kwargs,
                )
            )
    
    async def _chat_async(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        priority: str,
        timeout: float,
        **kwargs,
    ) -> ChatResponse:
        """Async implementation of chat."""
        start_time = time.time()
        
        # Schedule and execute
        result = await self._scheduler.schedule(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            timeout=timeout,
            **kwargs,
        )
        
        latency = time.time() - start_time
        
        # Extract content
        content = ""
        finish_reason = None
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if "message" in choice:
                content = choice["message"].get("content", "")
            finish_reason = choice.get("finish_reason")
        
        return ChatResponse(
            content=content,
            model=result.get("model", model),
            usage=result.get("usage"),
            finish_reason=finish_reason,
            id=result.get("id"),
            latency=latency,
            llm_latency=result.get("_latency"),
            key_used=result.get("_key_used"),
        )
    
    def chat_stream(
        self,
        model: str,
        messages: str | list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        validate: bool = True,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion synchronously.
        
        Yields:
            Content chunks as they arrive from the LLM
            
        Example:
            for chunk in client.chat_stream("gpt-4", "Write a story"):
                print(chunk, end="", flush=True)
        """
        self._ensure_initialized()
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Validate if enabled
        if validate:
            request = validate_chat_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            max_tokens = request.max_tokens
            temperature = request.temperature
        else:
            max_tokens = max_tokens or self.config.default_max_tokens
            temperature = temperature if temperature is not None else self.config.default_temperature
        
        # Create a new event loop for streaming
        # This handles both sync and nested async contexts
        import queue
        import threading
        
        chunk_queue: queue.Queue[str | None | Exception] = queue.Queue()
        
        async def _stream_to_queue():
            try:
                async for chunk in self._scheduler.schedule_stream(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                ):
                    chunk_queue.put(chunk)
                chunk_queue.put(None)  # Signal completion
            except Exception as e:
                chunk_queue.put(e)
        
        def _run_stream():
            asyncio.run(_stream_to_queue())
        
        # Start streaming in background thread
        thread = threading.Thread(target=_run_stream, daemon=True)
        thread.start()
        
        # Yield chunks from queue
        while True:
            item = chunk_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        
        thread.join(timeout=1.0)
    
    def add_key(
        self,
        api_key: str,
        provider: str,
        models: list[str],
        tpm_limit: int = 100000,
        rpm_limit: int = 60,
        key_id: str | None = None,
    ) -> None:
        """
        Add an API key at runtime.
        
        Args:
            api_key: The API key
            provider: Provider name (openai, groq, openrouter, etc.)
            models: List of supported models
            tpm_limit: Tokens per minute limit
            rpm_limit: Requests per minute limit
            key_id: Optional identifier for the key
        """
        key = KeyConfig(
            api_key=api_key,
            provider=provider,
            models=models,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit,
            key_id=key_id,
        )
        self.config.keys.append(key)
        self._rate_limiter.register_key(key)
    
    def status(self) -> dict[str, Any]:
        """Get current status of all keys and queues."""
        return self._scheduler.status()
    
    def capacity(self) -> dict[str, Any]:
        """Get current capacity across all keys."""
        return self._rate_limiter.capacity()
    
    def latency(self) -> dict[str, Any]:
        """
        Get latency statistics per key.
        
        Returns:
            Dict with latency stats for each key including:
            - avg_ms: Average latency in milliseconds
            - min_ms: Minimum latency
            - max_ms: Maximum latency  
            - p50_ms: 50th percentile (median)
            - p99_ms: 99th percentile
            - samples: Number of measurements
        """
        capacity = self._rate_limiter.capacity()
        return {
            key_id: key_data.get("latency", {})
            for key_id, key_data in capacity.get("keys", {}).items()
        }
    
    def scores(self) -> dict[str, dict[str, float]]:
        """
        Get balanced scores for all keys.
        
        Useful for understanding routing decisions when using 'balanced' strategy.
        Lower score = better (more likely to be selected).
        
        Returns:
            Dict with score breakdown per key:
            - total_score: Combined weighted score (0-1)
            - utilization: Capacity usage (0-1)
            - latency_normalized: Normalized latency (0-1)
            - latency_avg_ms: Actual average latency in ms
            - error_penalty: Penalty for recent errors (0-1)
            - freshness: Time since last use (0-1, lower = used recently)
            
        Example:
            {
                "groq-1": {
                    "total_score": 0.23,
                    "utilization": 0.15,
                    "latency_normalized": 0.08,
                    "latency_avg_ms": 245.5,
                    "error_penalty": 0.0,
                    "freshness": 0.85
                },
                ...
            }
        """
        return self._rate_limiter.scores()
    
    def queue_stats(self) -> dict[str, Any]:
        """
        Get current queue statistics.
        
        Returns:
            Dict with queue stats:
            - active_requests: Number of currently processing requests
            - max_concurrent: Maximum concurrent requests allowed
            - queue_utilization: Current usage ratio (0.0 to 1.0)
            - total_processed: Total requests completed
            - total_failures: Total failed requests  
            - queue_rejections: Requests rejected due to queue full
            - can_accept: Whether new requests can be accepted immediately
        """
        return self._scheduler.queue_stats()
    
    def can_accept_request(self) -> bool:
        """
        Check if a new request can be accepted without waiting.
        
        Useful for implementing load balancing or graceful degradation.
        
        Returns:
            True if request can be processed immediately, False otherwise
        """
        return self._scheduler.can_accept_request()
    
    def __repr__(self) -> str:
        return f"MAXLLM(keys={len(self.config.keys)}, strategy={self.config.strategy})"
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the client.
        
        Waits for in-flight requests to complete before returning.
        
        Args:
            timeout: Max seconds to wait for pending requests
        """
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                self._scheduler.shutdown(timeout),
                loop,
            ).result(timeout=timeout + 5)
        except RuntimeError:
            asyncio.run(self._scheduler.shutdown(timeout))
    
    def __enter__(self) -> "MAXLLM":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures graceful shutdown."""
        self.shutdown()


class MAXLLMAsync(MAXLLM):
    """
    Async version of MAXLLM client.
    
    Usage:
        async with MAXLLMAsync.from_config("config.yaml") as client:
            response = await client.chat("gpt-4o-mini", "Hello!")
    """
    
    async def chat(
        self,
        model: str,
        messages: str | list[dict[str, str]] | list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        priority: str = "medium",
        timeout: float = 120.0,
        validate: bool = True,
        **kwargs,
    ) -> ChatResponse:
        """Async chat completion with validation."""
        self._ensure_initialized()
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif messages and isinstance(messages[0], Message):
            messages = [m.to_dict() for m in messages]
        
        if validate:
            request = validate_chat_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                priority=priority,
                timeout=timeout,
                **kwargs,
            )
            max_tokens = request.max_tokens
            temperature = request.temperature
        else:
            max_tokens = max_tokens or self.config.default_max_tokens
            temperature = temperature if temperature is not None else self.config.default_temperature
        
        return await self._chat_async(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            timeout=timeout,
            **kwargs,
        )
    
    async def chat_stream(
        self,
        model: str,
        messages: str | list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        validate: bool = True,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming chat completion.
        
        Example:
            async for chunk in await client.chat_stream("gpt-4", "Hi"):
                print(chunk, end="", flush=True)
        """
        self._ensure_initialized()
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        if validate:
            request = validate_chat_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            max_tokens = request.max_tokens
            temperature = request.temperature
        else:
            max_tokens = max_tokens or self.config.default_max_tokens
            temperature = temperature if temperature is not None else self.config.default_temperature
        
        async for chunk in self._scheduler.schedule_stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield chunk
    
    async def shutdown(self, timeout: float = 30.0) -> None:
        """Async graceful shutdown."""
        await self._scheduler.shutdown(timeout)
    
    async def __aenter__(self) -> "MAXLLMAsync":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
