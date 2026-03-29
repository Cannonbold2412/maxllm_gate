"""Token bucket algorithm for rate limiting."""

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter implementation.
    
    Supports both TPM (tokens per minute) and RPM (requests per minute).
    Allows burst handling while maintaining average rate.
    
    The bucket refills continuously based on elapsed time, providing
    smooth rate limiting without hard window boundaries.
    """
    
    capacity: int  # Maximum tokens in bucket (TPM or RPM limit)
    refill_rate: float  # Tokens added per second
    tokens: float = field(default=0.0, init=False)
    last_update: float = field(default_factory=time.time, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    
    def __post_init__(self):
        # Start with full bucket
        self.tokens = float(self.capacity)
    
    @classmethod
    def from_per_minute(cls, per_minute: int) -> "TokenBucket":
        """Create a bucket from a per-minute rate limit."""
        return cls(
            capacity=per_minute,
            refill_rate=per_minute / 60.0,  # Convert to per-second
        )
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on elapsed time
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_update = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def can_consume(self, tokens: int = 1) -> bool:
        """Check if tokens are available without consuming."""
        with self._lock:
            self._refill()
            return self.tokens >= tokens
    
    def available(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self.tokens
    
    def time_until_available(self, tokens: int) -> float:
        """
        Calculate time until requested tokens will be available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds until tokens available (0 if already available)
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate
    
    def reserve(self, tokens: int) -> float:
        """
        Reserve tokens, returning wait time if needed.
        
        Unlike consume(), this always "reserves" the tokens by
        decrementing the bucket (even going negative) and returns
        the wait time needed before the request should execute.
        
        Args:
            tokens: Number of tokens to reserve
            
        Returns:
            Wait time in seconds (0 if immediate execution OK)
        """
        with self._lock:
            self._refill()
            
            wait_time = 0.0
            if self.tokens < tokens:
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
            
            self.tokens -= tokens
            return wait_time
    
    def add_tokens(self, tokens: int) -> None:
        """
        Add tokens back to bucket (e.g., for refunds).
        
        Args:
            tokens: Number of tokens to add
        """
        with self._lock:
            self._refill()
            self.tokens = min(self.capacity, self.tokens + tokens)
    
    def reset(self) -> None:
        """Reset bucket to full capacity."""
        with self._lock:
            self.tokens = float(self.capacity)
            self.last_update = time.time()
    
    def utilization(self) -> float:
        """Get current utilization (0.0 to 1.0)."""
        with self._lock:
            self._refill()
            return 1.0 - (self.tokens / self.capacity)
    
    def __repr__(self) -> str:
        return (
            f"TokenBucket(capacity={self.capacity}, "
            f"available={self.available():.1f}, "
            f"utilization={self.utilization():.1%})"
        )
