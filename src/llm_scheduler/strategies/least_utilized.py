"""Least-utilized key selection strategy."""

from llm_scheduler.strategies.base import SchedulingStrategy
from llm_scheduler.rate_limiting.tracker import RateLimitState


class LeastUtilizedStrategy(SchedulingStrategy):
    """
    Select the key with lowest current utilization.
    
    This strategy maximizes throughput by spreading load across
    all available keys, preventing any single key from becoming
    a bottleneck.
    """
    
    @property
    def name(self) -> str:
        return "least_utilized"
    
    def select(
        self,
        candidates: list[RateLimitState],
        estimated_tokens: int,
        **context,
    ) -> RateLimitState | None:
        if not candidates:
            return None
        
        # Sort by utilization (ascending)
        return min(candidates, key=lambda s: s.utilization())
