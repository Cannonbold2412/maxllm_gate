"""Redis backend for distributed rate limiting and state persistence."""

import json
import time
from dataclasses import dataclass
from typing import Any

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    
    url: str = "redis://localhost:6379"
    prefix: str = "maxllm_gate:"
    state_ttl: int = 120  # TTL for state keys (2 minutes, covers rate window)
    lock_timeout: int = 5  # Distributed lock timeout
    connection_timeout: float = 5.0
    max_connections: int = 20


class RedisBackend:
    """
    Redis backend for distributed maxllm_gate state.
    
    Provides:
    - Distributed rate limit state (tokens used, request counts)
    - Persistent latency tracking
    - Distributed locks for atomic operations
    - Pub/sub for multi-instance coordination
    
    Usage:
        backend = RedisBackend(RedisConfig(url="redis://localhost:6379"))
        await backend.connect()
        
        # Update token usage atomically
        await backend.update_tokens("key-1", tokens_used=500)
        
        # Get current state
        state = await backend.get_state("key-1")
    """
    
    def __init__(self, config: RedisConfig | None = None):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis support requires redis-py: pip install redis"
            )
        
        self.config = config or RedisConfig()
        self._client: aioredis.Redis | None = None
        self._pubsub = None
        self._connected = False
    
    @property
    def prefix(self) -> str:
        return self.config.prefix
    
    def _key(self, *parts: str) -> str:
        """Build a Redis key with prefix."""
        return f"{self.prefix}{':'.join(parts)}"
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return
        
        self._client = await aioredis.from_url(
            self.config.url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=self.config.connection_timeout,
            max_connections=self.config.max_connections,
        )
        
        # Test connection
        await self._client.ping()
        self._connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._connected = False
    
    async def update_tokens(
        self,
        key_id: str,
        tokens_used: int,
        window_seconds: int = 60,
    ) -> dict[str, Any]:
        """
        Atomically update token usage for a key.
        
        Uses Redis transactions to ensure consistency across instances.
        
        Args:
            key_id: API key identifier
            tokens_used: Tokens consumed by this request
            window_seconds: Rate limit window
            
        Returns:
            Updated state dict with current totals
        """
        state_key = self._key("state", key_id)
        now = time.time()
        window_start = int(now // window_seconds) * window_seconds
        
        # Lua script for atomic update
        script = """
        local key = KEYS[1]
        local tokens = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local ttl = tonumber(ARGV[3])
        
        -- Get current state
        local current = redis.call('HGETALL', key)
        local state = {}
        for i = 1, #current, 2 do
            state[current[i]] = current[i + 1]
        end
        
        -- Check if we need to reset window
        local stored_window = tonumber(state['window_start'] or 0)
        if stored_window < window_start then
            -- New window - reset counters
            redis.call('HMSET', key,
                'tokens_used', tokens,
                'request_count', 1,
                'window_start', window_start,
                'last_update', ARGV[4]
            )
        else
            -- Same window - increment
            redis.call('HINCRBY', key, 'tokens_used', tokens)
            redis.call('HINCRBY', key, 'request_count', 1)
            redis.call('HSET', key, 'last_update', ARGV[4])
        end
        
        redis.call('EXPIRE', key, ttl)
        
        return redis.call('HGETALL', key)
        """
        
        result = await self._client.eval(
            script,
            1,
            state_key,
            tokens_used,
            window_start,
            self.config.state_ttl,
            now,
        )
        
        # Parse result
        state = {}
        for i in range(0, len(result), 2):
            key = result[i]
            val = result[i + 1]
            if key in ("tokens_used", "request_count"):
                state[key] = int(val)
            elif key in ("window_start", "last_update"):
                state[key] = float(val)
            else:
                state[key] = val
        
        return state
    
    async def get_state(self, key_id: str) -> dict[str, Any] | None:
        """Get current state for a key."""
        state_key = self._key("state", key_id)
        data = await self._client.hgetall(state_key)
        
        if not data:
            return None
        
        # Parse numeric fields
        result = {}
        for k, v in data.items():
            if k in ("tokens_used", "request_count"):
                result[k] = int(v)
            elif k in ("window_start", "last_update"):
                result[k] = float(v)
            else:
                result[k] = v
        
        return result
    
    async def get_all_states(self, key_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get states for multiple keys efficiently."""
        if not key_ids:
            return {}
        
        pipe = self._client.pipeline()
        for key_id in key_ids:
            pipe.hgetall(self._key("state", key_id))
        
        results = await pipe.execute()
        
        states = {}
        for key_id, data in zip(key_ids, results):
            if data:
                state = {}
                for k, v in data.items():
                    if k in ("tokens_used", "request_count"):
                        state[k] = int(v)
                    elif k in ("window_start", "last_update"):
                        state[k] = float(v)
                    else:
                        state[k] = v
                states[key_id] = state
        
        return states
    
    async def record_latency(
        self,
        key_id: str,
        latency_ms: float,
        max_samples: int = 100,
    ) -> None:
        """
        Record a latency measurement.
        
        Stores recent samples for percentile calculation.
        """
        latency_key = self._key("latency", key_id)
        
        # Add to sorted set with timestamp as score (for cleanup)
        # and sample list for percentiles
        pipe = self._client.pipeline()
        
        # Store in a list (capped)
        samples_key = self._key("latency_samples", key_id)
        pipe.lpush(samples_key, latency_ms)
        pipe.ltrim(samples_key, 0, max_samples - 1)
        
        # Update running stats
        pipe.hincrbyfloat(latency_key, "total", latency_ms)
        pipe.hincrby(latency_key, "count", 1)
        
        # Set min/max
        script = """
        local key = KEYS[1]
        local value = tonumber(ARGV[1])
        local current_min = tonumber(redis.call('HGET', key, 'min') or value)
        local current_max = tonumber(redis.call('HGET', key, 'max') or value)
        
        if value < current_min then
            redis.call('HSET', key, 'min', value)
        end
        if value > current_max then
            redis.call('HSET', key, 'max', value)
        end
        """
        
        await pipe.execute()
        await self._client.eval(script, 1, latency_key, latency_ms)
        
        # Set TTL
        await self._client.expire(latency_key, 3600)  # 1 hour
        await self._client.expire(samples_key, 3600)
    
    async def get_latency_stats(self, key_id: str) -> dict[str, float]:
        """Get latency statistics for a key."""
        latency_key = self._key("latency", key_id)
        samples_key = self._key("latency_samples", key_id)
        
        pipe = self._client.pipeline()
        pipe.hgetall(latency_key)
        pipe.lrange(samples_key, 0, -1)
        
        stats_data, samples_data = await pipe.execute()
        
        if not stats_data:
            return {}
        
        count = int(stats_data.get("count", 0))
        if count == 0:
            return {}
        
        total = float(stats_data.get("total", 0))
        
        result = {
            "avg_ms": total / count,
            "min_ms": float(stats_data.get("min", 0)),
            "max_ms": float(stats_data.get("max", 0)),
            "samples": count,
        }
        
        # Calculate percentiles from samples
        if samples_data:
            samples = sorted([float(s) for s in samples_data])
            n = len(samples)
            result["p50_ms"] = samples[n // 2]
            result["p99_ms"] = samples[min(int(n * 0.99), n - 1)]
        
        return result
    
    async def record_error(self, key_id: str) -> int:
        """Record an error for a key. Returns consecutive error count."""
        error_key = self._key("errors", key_id)
        count = await self._client.incr(error_key)
        await self._client.expire(error_key, 300)  # 5 minute window
        return count
    
    async def clear_errors(self, key_id: str) -> None:
        """Clear error count on success."""
        error_key = self._key("errors", key_id)
        await self._client.delete(error_key)
    
    async def get_error_count(self, key_id: str) -> int:
        """Get current error count for a key."""
        error_key = self._key("errors", key_id)
        count = await self._client.get(error_key)
        return int(count) if count else 0
    
    async def acquire_lock(
        self,
        name: str,
        timeout: float | None = None,
    ) -> bool:
        """
        Acquire a distributed lock.
        
        Returns True if lock acquired, False otherwise.
        """
        timeout = timeout or self.config.lock_timeout
        lock_key = self._key("lock", name)
        
        # SET NX with expiry
        acquired = await self._client.set(
            lock_key,
            "1",
            nx=True,
            ex=int(timeout),
        )
        
        return acquired is not None
    
    async def release_lock(self, name: str) -> None:
        """Release a distributed lock."""
        lock_key = self._key("lock", name)
        await self._client.delete(lock_key)
    
    async def check_capacity(
        self,
        key_id: str,
        tokens_needed: int,
        tpm_limit: int,
        rpm_limit: int,
    ) -> tuple[bool, float]:
        """
        Check if a key has capacity for a request.
        
        Returns:
            Tuple of (has_capacity, wait_time_if_not)
        """
        state = await self.get_state(key_id)
        
        if state is None:
            return True, 0.0
        
        now = time.time()
        window_start = state.get("window_start", 0)
        
        # Check if window expired
        if now - window_start >= 60:
            return True, 0.0
        
        tokens_used = state.get("tokens_used", 0)
        request_count = state.get("request_count", 0)
        
        # Check limits
        if tokens_used + tokens_needed > tpm_limit:
            wait_time = 60 - (now - window_start)
            return False, max(0, wait_time)
        
        if request_count >= rpm_limit:
            wait_time = 60 - (now - window_start)
            return False, max(0, wait_time)
        
        return True, 0.0
    
    async def save_state_snapshot(self, key_ids: list[str]) -> None:
        """Save full state snapshot for recovery."""
        snapshot_key = self._key("snapshot")
        states = await self.get_all_states(key_ids)
        
        await self._client.set(
            snapshot_key,
            json.dumps(states),
            ex=300,  # 5 minutes
        )
    
    async def load_state_snapshot(self) -> dict[str, Any] | None:
        """Load state snapshot if available."""
        snapshot_key = self._key("snapshot")
        data = await self._client.get(snapshot_key)
        
        if data:
            return json.loads(data)
        return None


class HybridRateLimiter:
    """
    Rate limiter that uses Redis when available, falls back to in-memory.
    
    Provides seamless transition between local development and production.
    """
    
    def __init__(
        self,
        redis_url: str | None = None,
        fallback_to_memory: bool = True,
    ):
        self._redis_url = redis_url
        self._fallback = fallback_to_memory
        self._backend: RedisBackend | None = None
        self._memory_state: dict[str, dict[str, Any]] = {}
        self._using_redis = False
    
    async def initialize(self) -> None:
        """Initialize backend (Redis if available, else memory)."""
        if self._redis_url and REDIS_AVAILABLE:
            try:
                config = RedisConfig(url=self._redis_url)
                self._backend = RedisBackend(config)
                await self._backend.connect()
                self._using_redis = True
                return
            except Exception as e:
                if not self._fallback:
                    raise
                # Fall back to memory
                import logging
                logging.warning(f"Redis unavailable, using in-memory: {e}")
        
        # Use in-memory
        self._using_redis = False
    
    @property
    def is_distributed(self) -> bool:
        """Return True if using Redis backend."""
        return self._using_redis
    
    async def update_tokens(
        self,
        key_id: str,
        tokens_used: int,
    ) -> dict[str, Any]:
        """Update token usage."""
        if self._using_redis:
            return await self._backend.update_tokens(key_id, tokens_used)
        
        # In-memory fallback
        now = time.time()
        window_start = int(now // 60) * 60
        
        state = self._memory_state.get(key_id, {})
        
        if state.get("window_start", 0) < window_start:
            state = {
                "tokens_used": tokens_used,
                "request_count": 1,
                "window_start": window_start,
            }
        else:
            state["tokens_used"] = state.get("tokens_used", 0) + tokens_used
            state["request_count"] = state.get("request_count", 0) + 1
        
        self._memory_state[key_id] = state
        return state
    
    async def get_state(self, key_id: str) -> dict[str, Any] | None:
        """Get state for a key."""
        if self._using_redis:
            return await self._backend.get_state(key_id)
        
        return self._memory_state.get(key_id)
    
    async def record_latency(self, key_id: str, latency_ms: float) -> None:
        """Record latency measurement."""
        if self._using_redis:
            await self._backend.record_latency(key_id, latency_ms)
        # In-memory latency tracking handled by RateLimiter
    
    async def close(self) -> None:
        """Close connections."""
        if self._backend:
            await self._backend.disconnect()
