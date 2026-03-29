"""Multi-provider fallback strategy."""

from llm_scheduler.strategies.base import SchedulingStrategy
from llm_scheduler.rate_limiting.tracker import RateLimitState


class FallbackStrategy(SchedulingStrategy):
    """
    Fallback to alternative providers when primary is exhausted.
    
    Provider priority order can be configured. Falls back to
    next provider when current provider has no capacity.
    """
    
    def __init__(self, provider_order: list[str] | None = None):
        self.provider_order = provider_order or [
            "groq",     # Fast, free tier
            "openrouter",  # Good variety
            "openai",   # Fallback
        ]
    
    @property
    def name(self) -> str:
        return "fallback"
    
    def select(
        self,
        candidates: list[RateLimitState],
        estimated_tokens: int,
        **context,
    ) -> RateLimitState | None:
        if not candidates:
            return None
        
        # Group by provider
        by_provider: dict[str, list[RateLimitState]] = {}
        for state in candidates:
            if state.provider not in by_provider:
                by_provider[state.provider] = []
            by_provider[state.provider].append(state)
        
        # Try providers in order
        for provider in self.provider_order:
            if provider in by_provider:
                # Return least utilized from this provider
                return min(
                    by_provider[provider],
                    key=lambda s: s.utilization()
                )
        
        # If no preferred providers available, use any
        return min(candidates, key=lambda s: s.utilization())
