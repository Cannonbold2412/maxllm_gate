"""Rate limit tracker for API keys."""

import time
from dataclasses import dataclass, field
from typing import Any

from llm_scheduler.rate_limiting.token_bucket import TokenBucket


@dataclass
class RateLimitState:
    """Current rate limit state for an API key."""
    
    key_id: str
    provider: str
    
    # TPM tracking
    tpm_bucket: TokenBucket
    
    # RPM tracking  
    rpm_bucket: TokenBucket
    
    # Statistics
    total_requests: int = 0
    total_tokens_used: int = 0
    failed_requests: int = 0
    last_request_time: float = field(default_factory=time.time)
    last_error_time: float | None = None
    last_error_code: int | None = None
    consecutive_errors: int = 0
    
    @classmethod
    def from_config(
        cls,
        key_id: str,
        provider: str,
        tpm_limit: int,
        rpm_limit: int,
    ) -> "RateLimitState":
        """Create state from configuration."""
        return cls(
            key_id=key_id,
            provider=provider,
            tpm_bucket=TokenBucket.from_per_minute(tpm_limit),
            rpm_bucket=TokenBucket.from_per_minute(rpm_limit),
        )
    
    def can_handle(self, estimated_tokens: int) -> bool:
        """Check if this key can handle a request with estimated tokens."""
        return (
            self.rpm_bucket.can_consume(1) and
            self.tpm_bucket.can_consume(estimated_tokens)
        )
    
    def time_until_available(self, estimated_tokens: int) -> float:
        """Calculate when this key will have capacity."""
        rpm_wait = self.rpm_bucket.time_until_available(1)
        tpm_wait = self.tpm_bucket.time_until_available(estimated_tokens)
        return max(rpm_wait, tpm_wait)
    
    def consume(self, estimated_tokens: int) -> bool:
        """Consume capacity for a request."""
        if not self.can_handle(estimated_tokens):
            return False
        
        self.rpm_bucket.consume(1)
        self.tpm_bucket.consume(estimated_tokens)
        self.last_request_time = time.time()
        return True
    
    def reserve(self, estimated_tokens: int) -> float:
        """Reserve capacity, returning total wait time."""
        rpm_wait = self.rpm_bucket.reserve(1)
        tpm_wait = self.tpm_bucket.reserve(estimated_tokens)
        self.last_request_time = time.time()
        return max(rpm_wait, tpm_wait)
    
    def record_success(self, actual_tokens: int) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.total_tokens_used += actual_tokens
        self.consecutive_errors = 0
    
    def record_failure(self, error_code: int | None = None) -> None:
        """Record a failed request."""
        self.failed_requests += 1
        self.consecutive_errors += 1
        self.last_error_time = time.time()
        self.last_error_code = error_code
    
    def refund_tokens(self, tokens: int) -> None:
        """Refund tokens (e.g., if request failed before execution)."""
        self.tpm_bucket.add_tokens(tokens)
        self.rpm_bucket.add_tokens(1)
    
    def utilization(self) -> float:
        """Get overall utilization (max of TPM and RPM)."""
        return max(
            self.tpm_bucket.utilization(),
            self.rpm_bucket.utilization()
        )
    
    def is_healthy(self) -> bool:
        """Check if key is healthy (not in error backoff)."""
        if self.consecutive_errors >= 5:
            # Check if enough time has passed for recovery
            if self.last_error_time:
                backoff = min(60, 2 ** self.consecutive_errors)
                return time.time() - self.last_error_time > backoff
            return False
        return True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "key_id": self.key_id,
            "provider": self.provider,
            "tpm_available": int(self.tpm_bucket.available()),
            "tpm_capacity": self.tpm_bucket.capacity,
            "rpm_available": int(self.rpm_bucket.available()),
            "rpm_capacity": self.rpm_bucket.capacity,
            "utilization": self.utilization(),
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "failed_requests": self.failed_requests,
            "is_healthy": self.is_healthy(),
        }


class RateLimitTracker:
    """
    Tracks rate limit state across all API keys.
    
    Provides a unified view of capacity across all keys
    and helps find optimal keys for requests.
    """
    
    def __init__(self):
        self._states: dict[str, RateLimitState] = {}
    
    def register_key(
        self,
        key_id: str,
        provider: str,
        tpm_limit: int,
        rpm_limit: int,
    ) -> None:
        """Register an API key for tracking."""
        self._states[key_id] = RateLimitState.from_config(
            key_id=key_id,
            provider=provider,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit,
        )
    
    def get_state(self, key_id: str) -> RateLimitState | None:
        """Get state for a specific key."""
        return self._states.get(key_id)
    
    def get_all_states(self) -> list[RateLimitState]:
        """Get all tracked states."""
        return list(self._states.values())
    
    def get_available_keys(
        self,
        estimated_tokens: int,
        provider: str | None = None,
    ) -> list[RateLimitState]:
        """
        Get keys that can handle a request immediately.
        
        Args:
            estimated_tokens: Estimated token usage
            provider: Optional provider filter
            
        Returns:
            List of keys with available capacity
        """
        available = []
        
        for state in self._states.values():
            if provider and state.provider != provider:
                continue
            
            if state.is_healthy() and state.can_handle(estimated_tokens):
                available.append(state)
        
        return available
    
    def get_best_key(
        self,
        estimated_tokens: int,
        provider: str | None = None,
        strategy: str = "least_utilized",
    ) -> RateLimitState | None:
        """
        Get the best available key for a request.
        
        Args:
            estimated_tokens: Estimated token usage
            provider: Optional provider filter
            strategy: Selection strategy
            
        Returns:
            Best key state, or None if none available
        """
        available = self.get_available_keys(estimated_tokens, provider)
        
        if not available:
            return None
        
        if strategy == "least_utilized":
            return min(available, key=lambda s: s.utilization())
        elif strategy == "round_robin":
            # Sort by last request time (oldest first)
            return min(available, key=lambda s: s.last_request_time)
        else:
            return available[0]
    
    def get_earliest_available_key(
        self,
        estimated_tokens: int,
        provider: str | None = None,
    ) -> tuple[RateLimitState | None, float]:
        """
        Find key with earliest availability for deferred execution.
        
        Returns:
            Tuple of (state, wait_time_seconds)
        """
        candidates = []
        
        for state in self._states.values():
            if provider and state.provider != provider:
                continue
            
            if not state.is_healthy():
                continue
            
            wait_time = state.time_until_available(estimated_tokens)
            candidates.append((state, wait_time))
        
        if not candidates:
            return None, float("inf")
        
        # Find minimum wait time
        best = min(candidates, key=lambda x: x[1])
        return best
    
    def total_capacity(self, provider: str | None = None) -> dict[str, int]:
        """Get total TPM/RPM capacity across all keys."""
        tpm = 0
        rpm = 0
        
        for state in self._states.values():
            if provider and state.provider != provider:
                continue
            tpm += state.tpm_bucket.capacity
            rpm += state.rpm_bucket.capacity
        
        return {"tpm": tpm, "rpm": rpm}
    
    def available_capacity(self, provider: str | None = None) -> dict[str, int]:
        """Get available TPM/RPM across all keys."""
        tpm = 0
        rpm = 0
        
        for state in self._states.values():
            if provider and state.provider != provider:
                continue
            tpm += int(state.tpm_bucket.available())
            rpm += int(state.rpm_bucket.available())
        
        return {"tpm": tpm, "rpm": rpm}
    
    def to_dict(self) -> dict[str, Any]:
        """Get summary of all tracked keys."""
        return {
            key_id: state.to_dict()
            for key_id, state in self._states.items()
        }
