"""Round-robin key selection strategy."""

from llm_scheduler.strategies.base import SchedulingStrategy
from llm_scheduler.rate_limiting.tracker import RateLimitState


class RoundRobinStrategy(SchedulingStrategy):
    """
    Select keys in round-robin order based on last use time.
    
    This strategy ensures fair distribution across keys and
    is useful when keys have similar capacities.
    """
    
    @property
    def name(self) -> str:
        return "round_robin"
    
    def select(
        self,
        candidates: list[RateLimitState],
        estimated_tokens: int,
        **context,
    ) -> RateLimitState | None:
        if not candidates:
            return None
        
        # Sort by last request time (ascending = oldest first)
        return min(candidates, key=lambda s: s.last_request_time)
