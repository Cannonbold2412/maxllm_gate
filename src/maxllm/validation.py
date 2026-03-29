"""Input validation using Pydantic models."""

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    """A single chat message."""
    
    role: Literal["system", "assistant", "user", "function", "tool"]
    content: str = Field(..., min_length=1, max_length=1_000_000)
    name: str | None = None
    function_call: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    
    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v


class ChatRequest(BaseModel):
    """Validated chat completion request."""
    
    model: str = Field(..., min_length=1, max_length=256)
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=1000)
    max_tokens: int = Field(default=1024, ge=1, le=128000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    priority: Literal["high", "medium", "low"] = "medium"
    timeout: float = Field(default=120.0, ge=1.0, le=600.0)
    
    @field_validator("model")
    @classmethod
    def model_valid(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        # Basic sanitization
        if any(c in v for c in ["<", ">", ";", "&", "|"]):
            raise ValueError("Model name contains invalid characters")
        return v.strip()
    
    @model_validator(mode="after")
    def validate_messages(self) -> "ChatRequest":
        """Ensure message sequence is valid."""
        if not self.messages:
            raise ValueError("Messages list cannot be empty")
        
        # First non-system message should typically be user
        non_system = [m for m in self.messages if m.role != "system"]
        if non_system and non_system[-1].role not in ("user", "tool", "function"):
            # Warning but don't block - some use cases send assistant messages
            pass
        
        return self
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for LiteLLM."""
        result = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            result["stop"] = self.stop
        
        return result


class KeyConfigModel(BaseModel):
    """Validated API key configuration."""
    
    api_key: str = Field(..., min_length=1, max_length=512)
    provider: Literal[
        "openai", "azure", "anthropic", "groq", "openrouter",
        "nvidia_nim", "together_ai", "anyscale", "deepinfra",
        "fireworks_ai", "perplexity", "mistral", "cohere", "custom"
    ]
    models: list[str] = Field(..., min_length=1)
    tpm_limit: int = Field(default=100000, ge=1000, le=100_000_000)
    rpm_limit: int = Field(default=60, ge=1, le=100_000)
    key_id: str | None = Field(default=None, max_length=128)
    
    @field_validator("api_key")
    @classmethod
    def api_key_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("API key cannot be empty")
        return v
    
    @field_validator("models")
    @classmethod
    def models_valid(cls, v: list[str]) -> list[str]:
        cleaned = [m.strip() for m in v if m.strip()]
        if not cleaned:
            raise ValueError("At least one model must be specified")
        return cleaned


class ConfigModel(BaseModel):
    """Validated MAXLLM configuration."""
    
    keys: list[KeyConfigModel] = Field(default_factory=list)
    strategy: Literal[
        "least_utilized", "round_robin", "latency_aware", "balanced"
    ] = "balanced"
    default_max_tokens: int = Field(default=1024, ge=1, le=128000)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    token_buffer: float = Field(default=1.1, ge=1.0, le=2.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, ge=0.1, le=30.0)
    retry_max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    max_queue_size: int = Field(default=10000, ge=1, le=1_000_000)
    
    # Redis settings
    redis_url: str | None = None
    redis_prefix: str = "maxllm:"
    
    @model_validator(mode="after")
    def validate_delays(self) -> "ConfigModel":
        if self.retry_base_delay > self.retry_max_delay:
            raise ValueError("retry_base_delay cannot exceed retry_max_delay")
        return self


def validate_chat_request(
    model: str,
    messages: str | list[dict[str, str]],
    max_tokens: int | None = None,
    temperature: float | None = None,
    priority: str = "medium",
    timeout: float = 120.0,
    **kwargs,
) -> ChatRequest:
    """
    Validate and normalize a chat request.
    
    Args:
        model: Model name
        messages: String or list of message dicts
        max_tokens: Max response tokens
        temperature: Sampling temperature
        priority: Request priority
        timeout: Request timeout
        **kwargs: Additional parameters
        
    Returns:
        Validated ChatRequest
        
    Raises:
        ValueError: If validation fails
    """
    # Normalize messages
    if isinstance(messages, str):
        msg_list = [{"role": "user", "content": messages}]
    else:
        msg_list = messages
    
    # Build request
    chat_messages = [ChatMessage(**m) for m in msg_list]
    
    return ChatRequest(
        model=model,
        messages=chat_messages,
        max_tokens=max_tokens or 1024,
        temperature=temperature or 0.7,
        priority=priority,
        timeout=timeout,
        **{k: v for k, v in kwargs.items() if k in ChatRequest.model_fields},
    )
