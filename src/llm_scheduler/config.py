"""Configuration management for LLM Rate Limit Scheduler."""

import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIKeyConfig:
    """Configuration for a single API key."""
    
    def __init__(
        self,
        key_id: str,
        api_key: str,
        provider: str,
        models: list[str],
        tpm_limit: int,
        rpm_limit: int,
    ):
        self.key_id = key_id
        self.api_key = api_key
        self.provider = provider
        self.models = models
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
    
    def __repr__(self) -> str:
        return f"APIKeyConfig(key_id={self.key_id}, provider={self.provider})"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API Keys (JSON string)
    api_keys_config: str = Field(default="{}")
    
    # Scheduling
    default_strategy: str = Field(default="least_utilized")
    
    # Token estimation
    default_max_tokens: int = Field(default=1024)
    token_estimation_buffer: float = Field(default=1.1)
    
    # Retry
    max_retries: int = Field(default=3)
    retry_base_delay: float = Field(default=1.0)
    retry_max_delay: float = Field(default=60.0)
    
    # Queue
    max_queue_size: int = Field(default=10000)
    default_priority: str = Field(default="medium")
    
    # Redis (optional)
    redis_url: str | None = Field(default=None)
    use_redis_queue: bool = Field(default=False)
    
    @field_validator("default_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"least_utilized", "round_robin", "token_aware"}
        if v not in valid:
            raise ValueError(f"Strategy must be one of {valid}")
        return v
    
    @field_validator("default_priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        valid = {"high", "medium", "low"}
        if v not in valid:
            raise ValueError(f"Priority must be one of {valid}")
        return v
    
    def get_api_keys(self) -> dict[str, APIKeyConfig]:
        """Parse API keys configuration into typed objects."""
        try:
            raw_config = json.loads(self.api_keys_config)
        except json.JSONDecodeError:
            return {}
        
        result = {}
        for key_id, config in raw_config.items():
            result[key_id] = APIKeyConfig(
                key_id=key_id,
                api_key=config.get("api_key", ""),
                provider=config.get("provider", ""),
                models=config.get("models", []),
                tpm_limit=config.get("tpm_limit", 10000),
                rpm_limit=config.get("rpm_limit", 60),
            )
        return result
    
    def get_keys_for_model(self, model: str) -> list[APIKeyConfig]:
        """Get all API keys that support a given model."""
        keys = self.get_api_keys()
        return [k for k in keys.values() if model in k.models]
    
    def get_keys_for_provider(self, provider: str) -> list[APIKeyConfig]:
        """Get all API keys for a given provider."""
        keys = self.get_api_keys()
        return [k for k in keys.values() if k.provider == provider]


# Global settings instance
settings = Settings()
