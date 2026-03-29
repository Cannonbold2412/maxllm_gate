"""Main scheduling engine."""

import asyncio
import time
from typing import Any

from llm_scheduler.config import settings
from llm_scheduler.core.queue_manager import QueueManager, QueuedRequest
from llm_scheduler.core.dispatcher import Dispatcher
from llm_scheduler.core.token_estimator import token_estimator
from llm_scheduler.rate_limiting.key_manager import KeyManager
from llm_scheduler.observability.logging import get_logger
from llm_scheduler.observability.metrics import metrics


class SchedulerError(Exception):
    """Scheduler-level error."""
    pass


class Scheduler:
    """
    Main scheduling engine for LLM requests.
    
    This is the core differentiator - an intelligent scheduler that:
    1. Estimates tokens before execution
    2. Checks ALL available keys before deciding to wait
    3. Routes to least-utilized or best-fit key
    4. Defers execution when capacity exhausted
    5. Never blindly hits 429 errors
    
    Architecture:
        Client → Scheduler → Queue → Dispatcher → LiteLLM → Provider
    """
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.queue_manager = QueueManager(
            max_size=settings.max_queue_size,
            num_workers=10,
        )
        self.dispatcher = Dispatcher(self.key_manager)
        
        self._running = False
        self._logger = get_logger()
    
    async def start(self) -> None:
        """Initialize and start the scheduler."""
        # Load API keys
        self.key_manager.load_from_config()
        
        # Start queue with our processor
        await self.queue_manager.start(self._process_request)
        
        self._running = True
        self._logger.info("Scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        await self.queue_manager.stop()
        self._logger.info("Scheduler stopped")
    
    async def schedule(
        self,
        model: str,
        messages: list[dict[str, Any]],
        priority: str = "medium",
        max_tokens: int | None = None,
        temperature: float = 0.7,
        **extra_params,
    ) -> dict[str, Any]:
        """
        Schedule a completion request.
        
        This is the main entry point. The scheduler will:
        1. Estimate token usage
        2. Find best available key (or defer)
        3. Queue the request
        4. Return result when complete
        
        Args:
            model: Model name (e.g., "mixtral", "gpt-4o-mini")
            messages: Chat messages
            priority: "high", "medium", or "low"
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            **extra_params: Additional LiteLLM params
            
        Returns:
            LiteLLM response dict
        """
        if not self._running:
            raise SchedulerError("Scheduler not running")
        
        start_time = time.time()
        
        # Step 1: Estimate tokens
        input_tokens, output_tokens, total_estimated = token_estimator.estimate_total_tokens(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            buffer_multiplier=settings.token_estimation_buffer,
        )
        
        self._logger.debug(
            "Token estimation",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_estimated=total_estimated,
        )
        
        # Step 2: Select key (checks ALL keys, defers only if all exhausted)
        key_selection = self.key_manager.select_key(
            model=model,
            estimated_tokens=total_estimated,
            strategy=settings.default_strategy,
        )
        
        if key_selection is None:
            raise SchedulerError(f"No available keys for model: {model}")
        
        # Step 3: Determine scheduling time
        scheduled_at = time.time()
        if key_selection.is_deferred:
            scheduled_at += key_selection.wait_time
            self._logger.info(
                "Request deferred",
                model=model,
                wait_seconds=key_selection.wait_time,
                key_id=key_selection.key_id,
            )
            metrics.deferred_requests.inc()
        
        # Step 4: Reserve capacity
        self.key_manager.reserve_capacity(
            key_selection.key_id,
            total_estimated,
        )
        
        # Step 5: Create and queue request
        request = QueuedRequest.create(
            model=model,
            messages=messages,
            estimated_tokens=total_estimated,
            priority=priority,
            scheduled_at=scheduled_at,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_params={
                **extra_params,
                "_key_selection": key_selection,  # Attach selection
            },
        )
        
        await self.queue_manager.enqueue(request)
        
        metrics.requests_enqueued.inc()
        metrics.queue_size.set(self.queue_manager.queue_size())
        
        # Step 6: Wait for result
        try:
            result = await request.future
            
            # Record metrics
            total_time = time.time() - start_time
            metrics.request_latency.observe(total_time)
            metrics.requests_completed.inc()
            
            return result
            
        except Exception:
            metrics.requests_failed.inc()
            raise
    
    async def _process_request(self, request: QueuedRequest) -> dict[str, Any]:
        """Process a queued request (called by queue workers)."""
        key_selection = request.extra_params.pop("_key_selection", None)
        
        if key_selection is None:
            # Re-select key if not attached (e.g., after requeue)
            key_selection = self.key_manager.select_key(
                model=request.model,
                estimated_tokens=request.estimated_tokens,
                strategy=settings.default_strategy,
            )
            
            if key_selection is None:
                raise SchedulerError(f"No available keys for: {request.model}")
        
        # Dispatch to LiteLLM (always streaming)
        chunks = []
        async for chunk in self.dispatcher.dispatch_stream(request, key_selection):
            chunks.append(chunk)
        
        return {
            "choices": [
                {
                    "message": {"content": "".join(chunks)},
                    "finish_reason": None,
                }
            ],
            "content": "".join(chunks),
            "model": request.model,
            "stream": True,
        }
    
    async def schedule_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Schedule multiple requests concurrently.
        
        Args:
            requests: List of request dicts with same params as schedule()
            
        Returns:
            List of results (in same order)
        """
        tasks = [
            self.schedule(**req)
            for req in requests
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "queue": self.queue_manager.get_stats(),
            "keys": self.key_manager.get_status(),
            "strategy": settings.default_strategy,
        }
