"""Token-aware routing strategy."""

from llm_scheduler.strategies.base import SchedulingStrategy
from llm_scheduler.rate_limiting.tracker import RateLimitState


class TokenAwareStrategy(SchedulingStrategy):
    """
    Select key based on available token capacity.
    
    This strategy considers the estimated token usage and
    selects the key with the best fit - enough capacity
    without too much waste.
    """
    
    @property
    def name(self) -> str:
        return "token_aware"
    
    def select(
        self,
        candidates: list[RateLimitState],
        estimated_tokens: int,
        **context,
    ) -> RateLimitState | None:
        if not candidates:
            return None
        
        # Filter to keys that can handle the request
        viable = [
            s for s in candidates
            if s.tpm_bucket.available() >= estimated_tokens
        ]
        
        if not viable:
            # If no perfect fit, use least utilized
            return min(candidates, key=lambda s: s.utilization())
        
        # Find best fit: minimize waste (available - needed)
        def fit_score(state: RateLimitState) -> float:
            available = state.tpm_bucket.available()
            waste = available - estimated_tokens
            # Prefer keys with less waste but still enough capacity
            return waste
        
        return min(viable, key=fit_score)
