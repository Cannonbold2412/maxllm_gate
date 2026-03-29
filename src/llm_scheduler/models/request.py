"""Request models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class RequestStatus(Enum):
    """Status of an LLM request."""
    
    PENDING = "pending"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LLMRequest:
    """
    Represents an LLM completion request.
    
    This is the internal representation used for tracking
    and processing requests.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    
    # Parameters
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float | None = None
    
    # Scheduling
    priority: str = "medium"
    status: RequestStatus = RequestStatus.PENDING
    
    # Token estimation
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0
    
    # Actual usage (filled after completion)
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    actual_total_tokens: int | None = None
    
    # Timing
    created_at: float = field(default_factory=time.time)
    queued_at: float | None = None
    scheduled_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    
    # Execution
    assigned_key_id: str | None = None
    attempts: int = 0
    last_error: str | None = None
    
    # Extra parameters
    extra_params: dict[str, Any] = field(default_factory=dict)
    
    def queue_time(self) -> float | None:
        """Time spent waiting in queue."""
        if self.queued_at and self.started_at:
            return self.started_at - self.queued_at
        return None
    
    def execution_time(self) -> float | None:
        """Time spent executing."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def total_time(self) -> float | None:
        """Total time from creation to completion."""
        if self.completed_at:
            return self.completed_at - self.created_at
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model": self.model,
            "status": self.status.value,
            "priority": self.priority,
            "estimated_tokens": self.estimated_total_tokens,
            "actual_tokens": self.actual_total_tokens,
            "attempts": self.attempts,
            "assigned_key": self.assigned_key_id,
            "queue_time": self.queue_time(),
            "execution_time": self.execution_time(),
            "total_time": self.total_time(),
        }
