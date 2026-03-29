"""API Key pool manager."""

from typing import Any
from dataclasses import dataclass

from llm_scheduler.config import settings, APIKeyConfig
from llm_scheduler.rate_limiting.tracker import RateLimitTracker, RateLimitState


@dataclass
class KeySelection:
    """Result of key selection."""
    
    key_id: str
    api_key: str
    provider: str
    state: RateLimitState
    wait_time: float = 0.0
    is_deferred: bool = False
    
    def to_litellm_kwargs(self, model: str) -> dict[str, Any]:
        """Convert to kwargs for LiteLLM."""
        kwargs = {
            "api_key": self.api_key,
        }
        
        # Add provider-specific configuration
        if self.provider == "groq":
            kwargs["model"] = f"groq/{model}"
        elif self.provider == "openrouter":
            kwargs["model"] = f"openrouter/{model}"
            kwargs["api_base"] = "https://openrouter.ai/api/v1"
        elif self.provider == "openai":
            kwargs["model"] = model
        else:
            kwargs["model"] = model
        
        return kwargs


class KeyManager:
    """
    Manages pool of API keys with intelligent selection.
    
    Handles:
    - Key registration and configuration
    - Finding best key for requests
    - Multi-key capacity checking
    - Deferred execution scheduling
    """
    
    def __init__(self):
        self.tracker = RateLimitTracker()
        self._key_configs: dict[str, APIKeyConfig] = {}
        self._model_to_keys: dict[str, list[str]] = {}
        self._provider_to_keys: dict[str, list[str]] = {}
    
    def load_from_config(self) -> None:
        """Load API keys from application configuration."""
        api_keys = settings.get_api_keys()
        
        for key_id, config in api_keys.items():
            self.register_key(config)
    
    def register_key(self, config: APIKeyConfig) -> None:
        """Register an API key."""
        self._key_configs[config.key_id] = config
        
        # Register with rate tracker
        self.tracker.register_key(
            key_id=config.key_id,
            provider=config.provider,
            tpm_limit=config.tpm_limit,
            rpm_limit=config.rpm_limit,
        )
        
        # Build model index
        for model in config.models:
            if model not in self._model_to_keys:
                self._model_to_keys[model] = []
            self._model_to_keys[model].append(config.key_id)
        
        # Build provider index
        if config.provider not in self._provider_to_keys:
            self._provider_to_keys[config.provider] = []
        self._provider_to_keys[config.provider].append(config.key_id)
    
    def get_keys_for_model(self, model: str) -> list[str]:
        """Get key IDs that support a model."""
        # Direct match
        if model in self._model_to_keys:
            return self._model_to_keys[model]
        
        # Try without provider prefix (e.g., "groq/mixtral" -> "mixtral")
        if "/" in model:
            base_model = model.split("/", 1)[1]
            if base_model in self._model_to_keys:
                return self._model_to_keys[base_model]
        
        # Return all keys as fallback
        return list(self._key_configs.keys())
    
    def select_key(
        self,
        model: str,
        estimated_tokens: int,
        strategy: str = "least_utilized",
        preferred_provider: str | None = None,
    ) -> KeySelection | None:
        """
        Select the best available key for a request.
        
        CRITICAL: Checks ALL available keys before deciding to defer.
        Only defers if NO key has sufficient capacity.
        
        Args:
            model: Requested model
            estimated_tokens: Estimated total tokens
            strategy: Selection strategy
            preferred_provider: Optional provider preference
            
        Returns:
            KeySelection with selected key and wait time, or None if impossible
        """
        candidate_key_ids = self.get_keys_for_model(model)
        
        if not candidate_key_ids:
            return None
        
        # Filter by provider if specified
        if preferred_provider:
            provider_keys = self._provider_to_keys.get(preferred_provider, [])
            candidate_key_ids = [
                k for k in candidate_key_ids if k in provider_keys
            ]
        
        if not candidate_key_ids:
            # Fallback to any key for the model
            candidate_key_ids = self.get_keys_for_model(model)
        
        # Step 1: Try to find a key with immediate capacity
        best_available: RateLimitState | None = None
        best_utilization = float("inf")
        
        for key_id in candidate_key_ids:
            state = self.tracker.get_state(key_id)
            if state is None:
                continue
            
            if not state.is_healthy():
                continue
            
            if state.can_handle(estimated_tokens):
                if strategy == "least_utilized":
                    if state.utilization() < best_utilization:
                        best_utilization = state.utilization()
                        best_available = state
                elif strategy == "round_robin":
                    if best_available is None or \
                       state.last_request_time < best_available.last_request_time:
                        best_available = state
                else:
                    best_available = state
                    break
        
        if best_available:
            config = self._key_configs[best_available.key_id]
            return KeySelection(
                key_id=best_available.key_id,
                api_key=config.api_key,
                provider=config.provider,
                state=best_available,
                wait_time=0.0,
                is_deferred=False,
            )
        
        # Step 2: All keys exhausted - find earliest availability
        earliest_state: RateLimitState | None = None
        earliest_wait = float("inf")
        
        for key_id in candidate_key_ids:
            state = self.tracker.get_state(key_id)
            if state is None or not state.is_healthy():
                continue
            
            wait_time = state.time_until_available(estimated_tokens)
            if wait_time < earliest_wait:
                earliest_wait = wait_time
                earliest_state = state
        
        if earliest_state:
            config = self._key_configs[earliest_state.key_id]
            return KeySelection(
                key_id=earliest_state.key_id,
                api_key=config.api_key,
                provider=config.provider,
                state=earliest_state,
                wait_time=earliest_wait,
                is_deferred=True,
            )
        
        return None
    
    def consume_capacity(
        self,
        key_id: str,
        estimated_tokens: int,
    ) -> bool:
        """Consume capacity from a key."""
        state = self.tracker.get_state(key_id)
        if state is None:
            return False
        return state.consume(estimated_tokens)
    
    def reserve_capacity(
        self,
        key_id: str,
        estimated_tokens: int,
    ) -> float:
        """Reserve capacity and return wait time."""
        state = self.tracker.get_state(key_id)
        if state is None:
            return 0.0
        return state.reserve(estimated_tokens)
    
    def record_success(self, key_id: str, actual_tokens: int) -> None:
        """Record successful request."""
        state = self.tracker.get_state(key_id)
        if state:
            state.record_success(actual_tokens)
    
    def record_failure(self, key_id: str, error_code: int | None = None) -> None:
        """Record failed request."""
        state = self.tracker.get_state(key_id)
        if state:
            state.record_failure(error_code)
    
    def refund_capacity(self, key_id: str, tokens: int) -> None:
        """Refund capacity to a key."""
        state = self.tracker.get_state(key_id)
        if state:
            state.refund_tokens(tokens)
    
    def get_status(self) -> dict[str, Any]:
        """Get status of all managed keys."""
        return {
            "keys": self.tracker.to_dict(),
            "total_capacity": self.tracker.total_capacity(),
            "available_capacity": self.tracker.available_capacity(),
            "models_supported": list(self._model_to_keys.keys()),
            "providers": list(self._provider_to_keys.keys()),
        }
