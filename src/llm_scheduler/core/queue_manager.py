"""Priority queue manager with deferred execution support."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Awaitable

from llm_scheduler.observability.logging import get_logger


class Priority(IntEnum):
    """Request priority levels (lower = higher priority)."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(order=True)
class QueuedRequest:
    """A request in the queue."""
    
    # Sort key: (priority, scheduled_time, created_time)
    sort_key: tuple = field(compare=True, repr=False)
    
    # Request data (not used for comparison)
    request_id: str = field(compare=False)
    model: str = field(compare=False)
    messages: list[dict[str, Any]] = field(compare=False)
    priority: Priority = field(compare=False)
    estimated_tokens: int = field(compare=False)
    max_tokens: int | None = field(compare=False, default=None)
    temperature: float = field(compare=False, default=0.7)
    extra_params: dict[str, Any] = field(compare=False, default_factory=dict)
    
    # Timing
    created_at: float = field(compare=False, default_factory=time.time)
    scheduled_at: float = field(compare=False, default_factory=time.time)
    
    # Callbacks
    future: asyncio.Future = field(compare=False, default=None, repr=False)
    
    # Retry tracking
    attempts: int = field(compare=False, default=0)
    last_error: str | None = field(compare=False, default=None)
    
    @classmethod
    def create(
        cls,
        model: str,
        messages: list[dict[str, Any]],
        estimated_tokens: int,
        priority: str | Priority = "medium",
        scheduled_at: float | None = None,
        **kwargs,
    ) -> "QueuedRequest":
        """Factory method to create a queued request."""
        if isinstance(priority, str):
            priority = Priority[priority.upper()]
        
        request_id = str(uuid.uuid4())
        created_at = time.time()
        scheduled_at = scheduled_at or created_at
        
        # Sort key: priority first, then scheduled time, then creation time
        sort_key = (priority.value, scheduled_at, created_at)
        
        return cls(
            sort_key=sort_key,
            request_id=request_id,
            model=model,
            messages=messages,
            priority=priority,
            estimated_tokens=estimated_tokens,
            created_at=created_at,
            scheduled_at=scheduled_at,
            future=asyncio.get_event_loop().create_future(),
            **kwargs,
        )
    
    def reschedule(self, new_time: float) -> None:
        """Update scheduled time (for deferred execution)."""
        self.scheduled_at = new_time
        self.sort_key = (self.priority.value, new_time, self.created_at)
    
    def is_ready(self) -> bool:
        """Check if request is ready to execute (scheduled time passed)."""
        return time.time() >= self.scheduled_at
    
    def wait_time(self) -> float:
        """Get remaining wait time until scheduled execution."""
        return max(0, self.scheduled_at - time.time())
    
    def queue_time(self) -> float:
        """Get total time spent in queue."""
        return time.time() - self.created_at


class QueueManager:
    """
    Manages request queues with priority and deferred execution.
    
    Features:
    - Priority-based ordering (high > medium > low)
    - Deferred execution (schedule for future time)
    - Async worker pool for processing
    - Request timeout and cancellation
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        num_workers: int = 10,
    ):
        self.max_size = max_size
        self.num_workers = num_workers
        
        self._queue: asyncio.PriorityQueue = None
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._processor: Callable[[QueuedRequest], Awaitable[Any]] = None
        
        # Metrics
        self._total_enqueued = 0
        self._total_processed = 0
        self._total_failed = 0
        
        self._logger = get_logger()
    
    async def start(
        self,
        processor: Callable[[QueuedRequest], Awaitable[Any]],
    ) -> None:
        """Start the queue with worker pool."""
        self._queue = asyncio.PriorityQueue(maxsize=self.max_size)
        self._processor = processor
        self._running = True
        
        # Start worker tasks
        for i in range(self.num_workers):
            worker = asyncio.create_task(
                self._worker(f"worker-{i}"),
                name=f"queue-worker-{i}",
            )
            self._workers.append(worker)
        
        self._logger.info("Queue manager started", workers=self.num_workers)
    
    async def stop(self) -> None:
        """Stop the queue and workers."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers = []
        self._logger.info("Queue manager stopped")
    
    async def enqueue(self, request: QueuedRequest) -> None:
        """Add a request to the queue."""
        if not self._running:
            raise RuntimeError("Queue not running")
        
        await self._queue.put(request)
        self._total_enqueued += 1
        
        self._logger.debug(
            "Request enqueued",
            request_id=request.request_id,
            priority=request.priority.name,
            scheduled_in=request.wait_time(),
        )
    
    async def _worker(self, worker_id: str) -> None:
        """Worker coroutine that processes requests."""
        while self._running:
            try:
                # Get next request
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                
                # Check if we need to wait for scheduled time
                wait_time = request.wait_time()
                if wait_time > 0:
                    # Put back and wait
                    await self._queue.put(request)
                    await asyncio.sleep(min(wait_time, 0.1))
                    continue
                
                # Process the request
                try:
                    result = await self._processor(request)
                    
                    if not request.future.done():
                        request.future.set_result(result)
                    
                    self._total_processed += 1
                    
                    self._logger.debug(
                        "Request processed",
                        request_id=request.request_id,
                        worker=worker_id,
                        queue_time=request.queue_time(),
                    )
                    
                except Exception as e:
                    self._total_failed += 1
                    
                    if not request.future.done():
                        request.future.set_exception(e)
                    
                    self._logger.error(
                        "Request processing failed",
                        request_id=request.request_id,
                        error=str(e),
                    )
                
                finally:
                    self._queue.task_done()
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Worker error", worker=worker_id, error=str(e))
                await asyncio.sleep(0.1)
    
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize() if self._queue else 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_size": self.queue_size(),
            "max_size": self.max_size,
            "num_workers": self.num_workers,
            "total_enqueued": self._total_enqueued,
            "total_processed": self._total_processed,
            "total_failed": self._total_failed,
            "running": self._running,
        }
