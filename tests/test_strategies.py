"""Tests for scheduling strategies."""

import pytest
import time

from llm_scheduler.strategies.least_utilized import LeastUtilizedStrategy
from llm_scheduler.strategies.round_robin import RoundRobinStrategy
from llm_scheduler.strategies.token_aware import TokenAwareStrategy
from llm_scheduler.strategies.fallback import FallbackStrategy
from llm_scheduler.rate_limiting.tracker import RateLimitState


def create_test_state(
    key_id: str,
    provider: str = "openai",
    tpm: int = 10000,
    rpm: int = 100,
    utilization: float = 0.0,
) -> RateLimitState:
    """Create a test rate limit state."""
    state = RateLimitState.from_config(
        key_id=key_id,
        provider=provider,
        tpm_limit=tpm,
        rpm_limit=rpm,
    )
    
    # Simulate utilization by consuming tokens
    if utilization > 0:
        tokens_to_consume = int(tpm * utilization)
        state.tpm_bucket.consume(tokens_to_consume)
    
    return state


class TestLeastUtilizedStrategy:
    """Tests for least-utilized strategy."""
    
    def test_selects_least_utilized(self):
        """Selects key with lowest utilization."""
        strategy = LeastUtilizedStrategy()
        
        candidates = [
            create_test_state("key-1", utilization=0.5),
            create_test_state("key-2", utilization=0.2),
            create_test_state("key-3", utilization=0.8),
        ]
        
        selected = strategy.select(candidates, estimated_tokens=100)
        
        assert selected is not None
        assert selected.key_id == "key-2"
    
    def test_empty_candidates(self):
        """Returns None for empty list."""
        strategy = LeastUtilizedStrategy()
        
        selected = strategy.select([], estimated_tokens=100)
        assert selected is None
    
    def test_name(self):
        """Strategy has correct name."""
        assert LeastUtilizedStrategy().name == "least_utilized"


class TestRoundRobinStrategy:
    """Tests for round-robin strategy."""
    
    def test_selects_oldest_used(self):
        """Selects key used longest ago."""
        strategy = RoundRobinStrategy()
        
        candidates = [
            create_test_state("key-1"),
            create_test_state("key-2"),
            create_test_state("key-3"),
        ]
        
        # Simulate different last request times
        candidates[0].last_request_time = time.time() - 10
        candidates[1].last_request_time = time.time() - 30  # Oldest
        candidates[2].last_request_time = time.time() - 5
        
        selected = strategy.select(candidates, estimated_tokens=100)
        
        assert selected is not None
        assert selected.key_id == "key-2"
    
    def test_name(self):
        """Strategy has correct name."""
        assert RoundRobinStrategy().name == "round_robin"


class TestTokenAwareStrategy:
    """Tests for token-aware strategy."""
    
    def test_selects_best_fit(self):
        """Selects key with best token fit."""
        strategy = TokenAwareStrategy()
        
        candidates = [
            create_test_state("key-1", tpm=10000),  # Lots of headroom
            create_test_state("key-2", tpm=200),    # Close fit
            create_test_state("key-3", tpm=5000),   # Medium headroom
        ]
        
        # Need 150 tokens
        selected = strategy.select(candidates, estimated_tokens=150)
        
        assert selected is not None
        assert selected.key_id == "key-2"  # Best fit
    
    def test_name(self):
        """Strategy has correct name."""
        assert TokenAwareStrategy().name == "token_aware"


class TestFallbackStrategy:
    """Tests for fallback strategy."""
    
    def test_prefers_priority_provider(self):
        """Selects from priority provider first."""
        strategy = FallbackStrategy(provider_order=["groq", "openai"])
        
        candidates = [
            create_test_state("key-1", provider="openai"),
            create_test_state("key-2", provider="groq"),
        ]
        
        selected = strategy.select(candidates, estimated_tokens=100)
        
        assert selected is not None
        assert selected.key_id == "key-2"  # Groq preferred
    
    def test_falls_back_to_next(self):
        """Falls back when priority provider unavailable."""
        strategy = FallbackStrategy(provider_order=["anthropic", "openai"])
        
        candidates = [
            create_test_state("key-1", provider="openai"),
            create_test_state("key-2", provider="groq"),
        ]
        
        selected = strategy.select(candidates, estimated_tokens=100)
        
        assert selected is not None
        assert selected.key_id == "key-1"  # OpenAI as fallback
    
    def test_name(self):
        """Strategy has correct name."""
        assert FallbackStrategy().name == "fallback"
