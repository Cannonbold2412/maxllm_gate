"""Base strategy interface."""

from abc import ABC, abstractmethod

from llm_scheduler.rate_limiting.tracker import RateLimitState


class SchedulingStrategy(ABC):
    """Base class for scheduling strategies."""
    
    @abstractmethod
    def select(
        self,
        candidates: list[RateLimitState],
        estimated_tokens: int,
        **context,
    ) -> RateLimitState | None:
        """
        Select the best candidate key from available options.
        
        Args:
            candidates: List of keys with available capacity
            estimated_tokens: Token estimate for the request
            **context: Additional context (model, priority, etc.)
            
        Returns:
            Selected key state, or None if no suitable candidate
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


class StrategyRegistry:
    """Registry for scheduling strategies."""
    
    _strategies: dict[str, SchedulingStrategy] = {}
    
    @classmethod
    def register(cls, strategy: SchedulingStrategy) -> None:
        """Register a strategy."""
        cls._strategies[strategy.name] = strategy
    
    @classmethod
    def get(cls, name: str) -> SchedulingStrategy | None:
        """Get strategy by name."""
        return cls._strategies.get(name)
    
    @classmethod
    def list(cls) -> list[str]:
        """List registered strategy names."""
        return list(cls._strategies.keys())
