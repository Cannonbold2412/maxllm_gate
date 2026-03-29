"""Rate limiter with token bucket per key."""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from maxllm.config import KeyConfig


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0, init=False)
    last_update: float = field(default_factory=time.time, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    
    def __post_init__(self):
        self.tokens = float(self.capacity)
    
    @classmethod
    def from_per_minute(cls, per_minute: int) -> "TokenBucket":
        return cls(capacity=per_minute, refill_rate=per_minute / 60.0)
    
    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now
    
    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def can_consume(self, tokens: int = 1) -> bool:
        with self._lock:
            self._refill()
            return self.tokens >= tokens
    
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self.tokens
    
    def time_until_available(self, tokens: int) -> float:
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            return (tokens - self.tokens) / self.refill_rate
    
    def utilization(self) -> float:
        with self._lock:
            self._refill()
            return 1.0 - (self.tokens / self.capacity)


@dataclass
class LatencyTracker:
    """Tracks latency with exponential moving average."""
    
    avg_latency: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0
    p50_latency: float = 0.0
    p99_latency: float = 0.0
    sample_count: int = 0
    _recent: list = field(default_factory=list)
    _alpha: float = 0.1  # EMA smoothing factor
    
    def record(self, latency: float) -> None:
        """Record a latency measurement."""
        self.sample_count += 1
        
        # Update min/max
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)
        
        # Exponential moving average
        if self.avg_latency == 0:
            self.avg_latency = latency
        else:
            self.avg_latency = self._alpha * latency + (1 - self._alpha) * self.avg_latency
        
        # Keep recent samples for percentiles (last 100)
        self._recent.append(latency)
        if len(self._recent) > 100:
            self._recent.pop(0)
        
        # Update percentiles
        if self._recent:
            sorted_recent = sorted(self._recent)
            self.p50_latency = sorted_recent[len(sorted_recent) // 2]
            self.p99_latency = sorted_recent[int(len(sorted_recent) * 0.99)]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_ms": round(self.avg_latency * 1000, 2),
            "min_ms": round(self.min_latency * 1000, 2) if self.min_latency != float("inf") else None,
            "max_ms": round(self.max_latency * 1000, 2),
            "p50_ms": round(self.p50_latency * 1000, 2),
            "p99_ms": round(self.p99_latency * 1000, 2),
            "samples": self.sample_count,
        }


@dataclass
class KeyState:
    """State tracking for a single API key."""
    
    key_config: KeyConfig
    tpm_bucket: TokenBucket
    rpm_bucket: TokenBucket
    
    total_requests: int = 0
    total_tokens: int = 0
    last_request: float = field(default_factory=time.time)
    last_error: float | None = None
    consecutive_errors: int = 0
    latency: LatencyTracker = field(default_factory=LatencyTracker)
    
    @classmethod
    def from_config(cls, config: KeyConfig) -> "KeyState":
        return cls(
            key_config=config,
            tpm_bucket=TokenBucket.from_per_minute(config.tpm_limit),
            rpm_bucket=TokenBucket.from_per_minute(config.rpm_limit),
        )
    
    def can_handle(self, estimated_tokens: int) -> bool:
        return (
            self.rpm_bucket.can_consume(1) and
            self.tpm_bucket.can_consume(estimated_tokens) and
            self.is_healthy()
        )
    
    def time_until_available(self, estimated_tokens: int) -> float:
        rpm_wait = self.rpm_bucket.time_until_available(1)
        tpm_wait = self.tpm_bucket.time_until_available(estimated_tokens)
        return max(rpm_wait, tpm_wait)
    
    def consume(self, estimated_tokens: int) -> bool:
        if not self.can_handle(estimated_tokens):
            return False
        self.rpm_bucket.consume(1)
        self.tpm_bucket.consume(estimated_tokens)
        self.last_request = time.time()
        return True
    
    def record_success(self, actual_tokens: int, latency: float | None = None) -> None:
        self.total_requests += 1
        self.total_tokens += actual_tokens
        self.consecutive_errors = 0
        if latency is not None:
            self.latency.record(latency)
    
    def record_failure(self) -> None:
        self.consecutive_errors += 1
        self.last_error = time.time()
    
    def is_healthy(self, circuit_threshold: int = 5, circuit_timeout: float = 60.0) -> bool:
        """
        Check if key is healthy using circuit breaker pattern.
        
        Args:
            circuit_threshold: Consecutive failures to open circuit
            circuit_timeout: Seconds before retry after circuit opens
        """
        if self.consecutive_errors < circuit_threshold:
            return True
        if self.last_error is None:
            return True
        # Circuit is open - check if timeout elapsed
        return time.time() - self.last_error > circuit_timeout
    
    def utilization(self) -> float:
        return max(self.tpm_bucket.utilization(), self.rpm_bucket.utilization())
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_config.key_id,
            "provider": self.key_config.provider,
            "models": self.key_config.models,
            "tpm_available": int(self.tpm_bucket.available()),
            "tpm_capacity": self.tpm_bucket.capacity,
            "rpm_available": int(self.rpm_bucket.available()),
            "rpm_capacity": self.rpm_bucket.capacity,
            "utilization": self.utilization(),
            "total_requests": self.total_requests,
            "is_healthy": self.is_healthy(),
            "latency": self.latency.to_dict(),
        }


class RateLimiter:
    """Manages rate limits across multiple API keys."""
    
    def __init__(self):
        self._keys: dict[str, KeyState] = {}
        self._model_index: dict[str, list[str]] = {}
    
    def register_key(self, config: KeyConfig) -> None:
        """Register an API key."""
        state = KeyState.from_config(config)
        self._keys[config.key_id] = state
        
        # Index by model
        for model in config.models:
            if model not in self._model_index:
                self._model_index[model] = []
            self._model_index[model].append(config.key_id)
    
    def get_keys_for_model(self, model: str) -> list[str]:
        """Get key IDs that support a model."""
        if model in self._model_index:
            return self._model_index[model]
        
        # Try without provider prefix
        if "/" in model:
            base = model.split("/", 1)[1]
            if base in self._model_index:
                return self._model_index[base]
        
        # Return all keys as fallback
        return list(self._keys.keys())
    
    def select_key(
        self,
        model: str,
        estimated_tokens: int,
        strategy: str = "least_utilized",
        exclude_keys: set[str] | None = None,
        circuit_threshold: int = 5,
        circuit_timeout: float = 60.0,
    ) -> tuple[KeyState | None, float]:
        """
        Select best key for request.
        
        Args:
            model: Model name to route to
            estimated_tokens: Estimated total tokens for request
            strategy: Selection strategy
            exclude_keys: Set of key IDs to exclude (for fallback)
            circuit_threshold: Consecutive failures to open circuit breaker
            circuit_timeout: Seconds before retrying after circuit opens
        
        Returns:
            (key_state, wait_time) - wait_time is 0 if immediate
        """
        exclude_keys = exclude_keys or set()
        candidate_ids = self.get_keys_for_model(model)
        
        if not candidate_ids:
            return None, 0.0
        
        # Filter out excluded keys and check health
        candidate_ids = [
            k for k in candidate_ids 
            if k not in exclude_keys
        ]
        
        if not candidate_ids:
            return None, 0.0
        
        # Find immediately available keys
        available = []
        for key_id in candidate_ids:
            state = self._keys.get(key_id)
            if state and state.is_healthy(circuit_threshold, circuit_timeout):
                if state.can_handle(estimated_tokens):
                    available.append(state)
        
        if available:
            if strategy == "least_utilized":
                return min(available, key=lambda s: s.utilization()), 0.0
            elif strategy == "round_robin":
                return min(available, key=lambda s: s.last_request), 0.0
            elif strategy == "latency_aware":
                # Prefer keys with lower average latency
                # If no latency data yet, fall back to least_utilized
                def latency_score(s: KeyState) -> float:
                    if s.latency.sample_count < 3:
                        return s.utilization()  # Not enough data, use utilization
                    return s.latency.avg_latency
                return min(available, key=latency_score), 0.0
            elif strategy == "balanced":
                # Combined score: utilization + latency + health
                # Lower score = better
                return min(available, key=lambda s: self._balanced_score(s)), 0.0
            else:
                return available[0], 0.0
        
        # All exhausted - find earliest availability
        best_state = None
        best_wait = float("inf")
        
        for key_id in candidate_ids:
            state = self._keys.get(key_id)
            if state and state.is_healthy(circuit_threshold, circuit_timeout):
                wait = state.time_until_available(estimated_tokens)
                if wait < best_wait:
                    best_wait = wait
                    best_state = state
        
        return best_state, best_wait
    
    def _balanced_score(self, state: KeyState) -> float:
        """
        Calculate balanced score combining multiple factors.
        
        Score components (lower = better):
        - Utilization (0-1): How much capacity is used
        - Latency (normalized): Average response time
        - Error penalty: Recent failures increase score
        - Freshness bonus: Keys not used recently get slight preference
        
        Weights can be tuned based on priorities.
        """
        # Weights (sum to 1.0 for interpretability)
        W_UTILIZATION = 0.40   # Capacity is important
        W_LATENCY = 0.35       # Speed matters
        W_ERRORS = 0.15        # Avoid error-prone keys
        W_FRESHNESS = 0.10     # Slight round-robin effect
        
        # 1. Utilization score (0-1, lower is better)
        utilization_score = state.utilization()
        
        # 2. Latency score (normalized to 0-1 range)
        # Assume typical latency range 0.1s - 5s
        if state.latency.sample_count >= 3:
            # Normalize: 0.1s -> 0, 5s -> 1
            latency_score = min(1.0, max(0.0, (state.latency.avg_latency - 0.1) / 4.9))
        else:
            # No data yet, assume average
            latency_score = 0.5
        
        # 3. Error penalty (0-1, based on consecutive errors)
        # 0 errors = 0, 5+ errors = 1
        error_score = min(1.0, state.consecutive_errors / 5.0)
        
        # 4. Freshness score (prefer keys not used recently)
        # Normalize based on time since last request (0-60 seconds)
        time_since_last = time.time() - state.last_request
        freshness_score = max(0.0, 1.0 - (time_since_last / 60.0))
        
        # Combined weighted score
        total_score = (
            W_UTILIZATION * utilization_score +
            W_LATENCY * latency_score +
            W_ERRORS * error_score +
            W_FRESHNESS * freshness_score
        )
        
        return total_score
    
    def consume(self, key_id: str, tokens: int) -> bool:
        """Consume capacity from a key."""
        state = self._keys.get(key_id)
        if state:
            return state.consume(tokens)
        return False
    
    def record_success(self, key_id: str, tokens: int, latency: float | None = None) -> None:
        """Record successful request with latency."""
        state = self._keys.get(key_id)
        if state:
            state.record_success(tokens, latency)
    
    def record_failure(self, key_id: str) -> None:
        """Record failed request."""
        state = self._keys.get(key_id)
        if state:
            state.record_failure()
    
    def capacity(self) -> dict[str, Any]:
        """Get total capacity."""
        total_tpm = sum(s.tpm_bucket.capacity for s in self._keys.values())
        total_rpm = sum(s.rpm_bucket.capacity for s in self._keys.values())
        avail_tpm = sum(int(s.tpm_bucket.available()) for s in self._keys.values())
        avail_rpm = sum(int(s.rpm_bucket.available()) for s in self._keys.values())
        
        return {
            "total_tpm": total_tpm,
            "available_tpm": avail_tpm,
            "total_rpm": total_rpm,
            "available_rpm": avail_rpm,
            "keys": {k: v.to_dict() for k, v in self._keys.items()},
        }
    
    def scores(self) -> dict[str, dict[str, float]]:
        """
        Get balanced scores for all keys (for debugging/transparency).
        
        Returns breakdown of each score component.
        """
        result = {}
        
        for key_id, state in self._keys.items():
            # Calculate individual components
            utilization = state.utilization()
            
            if state.latency.sample_count >= 3:
                latency_norm = min(1.0, max(0.0, (state.latency.avg_latency - 0.1) / 4.9))
            else:
                latency_norm = 0.5
            
            error_score = min(1.0, state.consecutive_errors / 5.0)
            
            time_since_last = time.time() - state.last_request
            freshness = max(0.0, 1.0 - (time_since_last / 60.0))
            
            total = self._balanced_score(state)
            
            result[key_id] = {
                "total_score": round(total, 4),
                "utilization": round(utilization, 4),
                "latency_normalized": round(latency_norm, 4),
                "latency_avg_ms": round(state.latency.avg_latency * 1000, 2),
                "error_penalty": round(error_score, 4),
                "freshness": round(freshness, 4),
                "consecutive_errors": state.consecutive_errors,
                "samples": state.latency.sample_count,
            }
        
        return result
