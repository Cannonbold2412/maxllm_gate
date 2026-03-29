"""Pydantic schemas for API requests and responses."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""
    
    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message sender"
    )
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    
    model: str = Field(
        description="Model name (e.g., 'mixtral', 'gpt-4o-mini', 'llama-3.1-70b')"
    )
    messages: list[Message] = Field(
        description="List of chat messages"
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Request priority for queue ordering"
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "priority": "medium",
                    "max_tokens": 1024,
                    "temperature": 0.7
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response body for /chat endpoint."""
    
    id: str = Field(description="Response ID")
    model: str = Field(description="Model used")
    content: str = Field(description="Generated content")
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage statistics"
    )
    finish_reason: str | None = Field(
        default=None,
        description="Reason for completion"
    )


class BatchRequest(BaseModel):
    """Request body for /batch endpoint."""
    
    requests: list[ChatRequest] = Field(
        description="List of chat requests to process"
    )


class BatchResponse(BaseModel):
    """Response body for /batch endpoint."""
    
    results: list[ChatResponse | dict[str, str]] = Field(
        description="Results for each request (response or error)"
    )
    total: int = Field(description="Total requests")
    successful: int = Field(description="Successful requests")
    failed: int = Field(description="Failed requests")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall health status"
    )
    scheduler_running: bool = Field(description="Whether scheduler is running")
    queue_size: int = Field(description="Current queue size")
    keys_available: int = Field(description="Number of healthy API keys")


class StatusResponse(BaseModel):
    """Response body for /status endpoint."""
    
    running: bool
    queue: dict[str, Any]
    keys: dict[str, Any]
    strategy: str


class KeyStatus(BaseModel):
    """Status of a single API key."""
    
    key_id: str
    provider: str
    tpm_available: int
    tpm_capacity: int
    rpm_available: int
    rpm_capacity: int
    utilization: float
    total_requests: int
    total_tokens_used: int
    is_healthy: bool


class CapacityResponse(BaseModel):
    """Response body for /capacity endpoint."""
    
    total_tpm: int
    available_tpm: int
    total_rpm: int
    available_rpm: int
    keys: list[KeyStatus]
