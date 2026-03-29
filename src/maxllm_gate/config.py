"""maxllm_gate Configuration handling."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KeyConfig:
    """Configuration for a single API key."""
    
    api_key: str
    provider: str
    models: list[str]
    tpm_limit: int = 100000
    rpm_limit: int = 60
    key_id: str | None = None
    
    def __post_init__(self):
        if self.key_id is None:
            # Generate key_id from provider and hash
            self.key_id = f"{self.provider}-{hash(self.api_key) % 10000:04d}"
    
    @classmethod
    def from_dict(cls, data: dict[str, Any], key_id: str | None = None) -> "KeyConfig":
        """Create from dictionary."""
        return cls(
            key_id=key_id or data.get("key_id"),
            api_key=data["api_key"],
            provider=data["provider"],
            models=data.get("models", []),
            tpm_limit=data.get("tpm_limit", 100000),
            rpm_limit=data.get("rpm_limit", 60),
        )


@dataclass
class maxllm_gate_config:
    """Main configuration for maxllm_gate."""
    
    keys: list[KeyConfig] = field(default_factory=list)
    
    # Scheduling
    strategy: str = "least_utilized"
    
    # Defaults
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    token_buffer: float = 1.2  # Increased buffer for safety
    
    # Retry
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    
    # Queue
    max_queue_size: int = 10000
    
    # Concurrency
    max_concurrent_requests: int = 100  # Max parallel LLM requests
    
    # Timeout & Limits (NEW - for edge cases)
    max_wait_time: float = 60.0   # Max seconds to wait for capacity
    request_timeout: float = 120.0  # Total request timeout
    
    # Circuit Breaker (NEW)
    circuit_breaker_threshold: int = 5  # Consecutive failures to open circuit
    circuit_breaker_timeout: float = 60.0  # Seconds before retry after circuit opens
    
    # Server (for API mode)
    host: str = "0.0.0.0"
    port: int = 8000
    
    @classmethod
    def from_file(cls, path: str | Path) -> "maxllm_gate_config":
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        content = path.read_text()
        
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML required for YAML config: pip install pyyaml")
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            # Try JSON first, then YAML
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except ImportError:
                    raise ValueError(f"Unknown config format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "maxllm_gate_config":
        """Create from dictionary."""
        keys = []
        
        # Handle keys as list or dict
        keys_data = data.get("keys", [])
        if isinstance(keys_data, dict):
            for key_id, key_config in keys_data.items():
                keys.append(KeyConfig.from_dict(key_config, key_id=key_id))
        elif isinstance(keys_data, list):
            for key_config in keys_data:
                keys.append(KeyConfig.from_dict(key_config))
        
        return cls(
            keys=keys,
            strategy=data.get("strategy", "least_utilized"),
            default_max_tokens=data.get("default_max_tokens", 1024),
            default_temperature=data.get("default_temperature", 0.7),
            token_buffer=data.get("token_buffer", 1.2),
            max_retries=data.get("max_retries", 3),
            retry_base_delay=data.get("retry_base_delay", 1.0),
            retry_max_delay=data.get("retry_max_delay", 60.0),
            max_queue_size=data.get("max_queue_size", 10000),
            max_concurrent_requests=data.get("max_concurrent_requests", 100),
            max_wait_time=data.get("max_wait_time", 60.0),
            request_timeout=data.get("request_timeout", 120.0),
            circuit_breaker_threshold=data.get("circuit_breaker_threshold", 5),
            circuit_breaker_timeout=data.get("circuit_breaker_timeout", 60.0),
            host=data.get("host", "0.0.0.0"),
            port=data.get("port", 8000),
        )
    
    @classmethod
    def from_env(cls) -> "maxllm_gate_config":
        """Load configuration from environment variables."""
        keys_json = os.environ.get("maxllm_gate_KEYS", "{}")
        
        try:
            keys_data = json.loads(keys_json)
        except json.JSONDecodeError:
            keys_data = {}
        
        keys = []
        for key_id, key_config in keys_data.items():
            keys.append(KeyConfig.from_dict(key_config, key_id=key_id))
        
        return cls(
            keys=keys,
            strategy=os.environ.get("maxllm_gate_STRATEGY", "least_utilized"),
            default_max_tokens=int(os.environ.get("maxllm_gate_MAX_TOKENS", "1024")),
            max_retries=int(os.environ.get("maxllm_gate_MAX_RETRIES", "3")),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "keys": {
                k.key_id: {
                    "api_key": k.api_key,
                    "provider": k.provider,
                    "models": k.models,
                    "tpm_limit": k.tpm_limit,
                    "rpm_limit": k.rpm_limit,
                }
                for k in self.keys
            },
            "strategy": self.strategy,
            "default_max_tokens": self.default_max_tokens,
            "default_temperature": self.default_temperature,
            "token_buffer": self.token_buffer,
            "max_retries": self.max_retries,
            "max_queue_size": self.max_queue_size,
            "max_concurrent_requests": self.max_concurrent_requests,
        }
