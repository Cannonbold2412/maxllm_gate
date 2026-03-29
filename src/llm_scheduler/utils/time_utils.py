"""Time and window utilities for rate limiting."""

import time
from dataclasses import dataclass
from typing import Generator
from contextlib import contextmanager


@dataclass
class TimeWindow:
    """Represents a rate limit time window."""
    
    start: float
    duration: float  # seconds
    
    @property
    def end(self) -> float:
        return self.start + self.duration
    
    @property
    def remaining(self) -> float:
        """Seconds remaining in window."""
        return max(0, self.end - time.time())
    
    @property
    def elapsed(self) -> float:
        """Seconds elapsed since window start."""
        return time.time() - self.start
    
    @property
    def progress(self) -> float:
        """Progress through window (0.0 to 1.0)."""
        return min(1.0, self.elapsed / self.duration)
    
    def is_expired(self) -> bool:
        """Check if window has expired."""
        return time.time() >= self.end
    
    def next_window(self) -> "TimeWindow":
        """Create the next consecutive window."""
        return TimeWindow(start=self.end, duration=self.duration)


def rate_limit_window(duration: float = 60.0) -> TimeWindow:
    """Create a new rate limit window starting now."""
    return TimeWindow(start=time.time(), duration=duration)


class SlidingWindow:
    """
    Sliding window rate limiter.
    
    Tracks events within a sliding time window.
    """
    
    def __init__(self, window_size: float = 60.0):
        self.window_size = window_size
        self._events: list[tuple[float, int]] = []  # (timestamp, count)
    
    def add(self, count: int = 1) -> None:
        """Add an event to the window."""
        now = time.time()
        self._events.append((now, count))
        self._cleanup(now)
    
    def _cleanup(self, now: float) -> None:
        """Remove expired events."""
        cutoff = now - self.window_size
        self._events = [
            (ts, count) for ts, count in self._events
            if ts > cutoff
        ]
    
    def count(self) -> int:
        """Get total count in current window."""
        self._cleanup(time.time())
        return sum(count for _, count in self._events)
    
    def can_add(self, limit: int, count: int = 1) -> bool:
        """Check if adding count would exceed limit."""
        return self.count() + count <= limit


@contextmanager
def timed_operation(name: str = "operation") -> Generator[dict, None, None]:
    """
    Context manager for timing operations.
    
    Usage:
        with timed_operation("api_call") as timing:
            do_something()
        print(f"Took {timing['duration']:.2f}s")
    """
    result = {"name": name, "start": time.time(), "duration": 0.0}
    try:
        yield result
    finally:
        result["duration"] = time.time() - result["start"]


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"
