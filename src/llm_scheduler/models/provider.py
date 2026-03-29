"""Provider models."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class Provider(Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    NVIDIA_NIM = "nvidia_nim"
    TOGETHER = "together"
    ANYSCALE = "anyscale"
    CUSTOM = "custom"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    
    provider: Provider
    api_base: str | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    
    # Rate limits (defaults, can be overridden per key)
    default_tpm: int = 100000
    default_rpm: int = 60
    
    # Model mappings (provider model name -> canonical name)
    model_mappings: dict[str, str] = field(default_factory=dict)
    
    # Supported models
    supported_models: list[str] = field(default_factory=list)
    
    @classmethod
    def openai(cls) -> "ProviderConfig":
        """OpenAI configuration."""
        return cls(
            provider=Provider.OPENAI,
            default_tpm=90000,
            default_rpm=500,
            supported_models=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ],
        )
    
    @classmethod
    def groq(cls) -> "ProviderConfig":
        """Groq configuration."""
        return cls(
            provider=Provider.GROQ,
            default_tpm=30000,
            default_rpm=30,
            supported_models=[
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
        )
    
    @classmethod
    def openrouter(cls) -> "ProviderConfig":
        """OpenRouter configuration."""
        return cls(
            provider=Provider.OPENROUTER,
            api_base="https://openrouter.ai/api/v1",
            default_tpm=100000,
            default_rpm=200,
            default_headers={
                "HTTP-Referer": "https://llm-scheduler.local",
            },
            supported_models=[
                "anthropic/claude-3-haiku",
                "anthropic/claude-3-sonnet",
                "meta-llama/llama-3-70b-instruct",
                "mistralai/mixtral-8x7b-instruct",
            ],
        )
    
    def to_litellm_kwargs(self, api_key: str, model: str) -> dict[str, Any]:
        """Convert to LiteLLM kwargs."""
        kwargs = {"api_key": api_key}
        
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Add provider prefix if needed
        if self.provider == Provider.GROQ:
            kwargs["model"] = f"groq/{model}"
        elif self.provider == Provider.OPENROUTER:
            kwargs["model"] = f"openrouter/{model}"
        else:
            kwargs["model"] = model
        
        return kwargs
