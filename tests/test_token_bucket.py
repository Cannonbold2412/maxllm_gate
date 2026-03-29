"""Tests for token bucket rate limiter."""

import time
import pytest

from llm_scheduler.rate_limiting.token_bucket import TokenBucket


class TestTokenBucket:
    """Test suite for TokenBucket."""
    
    def test_initial_state(self):
        """Bucket starts full."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        assert bucket.available() == 100
        assert bucket.utilization() == 0.0
    
    def test_consume_success(self):
        """Can consume available tokens."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        assert bucket.consume(50) is True
        assert bucket.available() == 50
        assert bucket.utilization() == 0.5
    
    def test_consume_failure(self):
        """Cannot consume more than available."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        assert bucket.consume(150) is False
        assert bucket.available() == 100  # Unchanged
    
    def test_can_consume(self):
        """Check without consuming."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        assert bucket.can_consume(50) is True
        assert bucket.can_consume(150) is False
        assert bucket.available() == 100  # Unchanged
    
    def test_refill(self):
        """Tokens refill over time."""
        bucket = TokenBucket(capacity=100, refill_rate=100.0)  # 100/sec
        
        bucket.consume(50)
        assert bucket.available() == 50
        
        time.sleep(0.1)  # Wait 100ms
        
        # Should have refilled ~10 tokens
        available = bucket.available()
        assert 55 <= available <= 65
    
    def test_refill_cap(self):
        """Refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=100, refill_rate=1000.0)
        
        bucket.consume(10)
        time.sleep(0.1)
        
        assert bucket.available() <= 100
    
    def test_time_until_available(self):
        """Calculate wait time correctly."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)  # 10/sec
        
        bucket.consume(100)  # Empty
        
        wait = bucket.time_until_available(50)
        assert 4.5 <= wait <= 5.5  # ~5 seconds for 50 tokens at 10/sec
    
    def test_time_until_available_immediate(self):
        """No wait when tokens available."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        wait = bucket.time_until_available(50)
        assert wait == 0.0
    
    def test_reserve(self):
        """Reserve returns wait time and decrements."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        bucket.consume(80)
        
        # Reserve 50 tokens (only 20 available)
        wait = bucket.reserve(50)
        assert 2.5 <= wait <= 3.5  # ~3 seconds for 30 tokens
        
        # Bucket should now be negative
        assert bucket.available() < 0
    
    def test_add_tokens(self):
        """Can add tokens (refund)."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        bucket.consume(60)
        assert bucket.available() == 40
        
        bucket.add_tokens(30)
        assert bucket.available() == 70
    
    def test_add_tokens_cap(self):
        """Adding tokens doesn't exceed capacity."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        bucket.add_tokens(50)
        assert bucket.available() == 100
    
    def test_reset(self):
        """Reset fills bucket."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        bucket.consume(100)
        assert bucket.available() == 0
        
        bucket.reset()
        assert bucket.available() == 100
    
    def test_from_per_minute(self):
        """Create bucket from per-minute rate."""
        bucket = TokenBucket.from_per_minute(60)
        
        assert bucket.capacity == 60
        assert bucket.refill_rate == 1.0  # 60/60 = 1/sec
