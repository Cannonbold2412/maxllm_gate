"""Mock LiteLLM for testing."""

from typing import Any
from dataclasses import dataclass


@dataclass
class MockUsage:
    """Mock token usage."""
    
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class MockMessage:
    """Mock message."""
    
    role: str = "assistant"
    content: str = "This is a mock response."


@dataclass
class MockChoice:
    """Mock completion choice."""
    
    index: int = 0
    message: MockMessage = None
    finish_reason: str = "stop"
    
    def __post_init__(self):
        if self.message is None:
            self.message = MockMessage()


@dataclass
class MockResponse:
    """Mock LiteLLM response."""
    
    id: str = "mock-response-id"
    model: str = "mock-model"
    choices: list = None
    usage: MockUsage = None
    
    def __post_init__(self):
        if self.choices is None:
            self.choices = [MockChoice()]
        if self.usage is None:
            self.usage = MockUsage()
    
    def model_dump(self) -> dict[str, Any]:
        """Convert to dict (mimics Pydantic)."""
        return {
            "id": self.id,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


class MockLiteLLM:
    """Mock LiteLLM module for testing."""
    
    def __init__(self):
        self.calls = []
        self.responses = []
        self._response_index = 0
        self.should_fail = False
        self.failure_error = None
    
    async def acompletion(self, **kwargs) -> MockResponse:
        """Mock async completion."""
        self.calls.append(kwargs)
        
        if self.should_fail:
            if self.failure_error:
                raise self.failure_error
            raise Exception("Mock failure")
        
        if self.responses:
            response = self.responses[self._response_index % len(self.responses)]
            self._response_index += 1
            return response
        
        return MockResponse(model=kwargs.get("model", "mock"))
    
    def completion(self, **kwargs) -> MockResponse:
        """Mock sync completion."""
        self.calls.append(kwargs)
        
        if self.should_fail:
            if self.failure_error:
                raise self.failure_error
            raise Exception("Mock failure")
        
        return MockResponse(model=kwargs.get("model", "mock"))
    
    def set_responses(self, responses: list[MockResponse]) -> None:
        """Set sequence of responses to return."""
        self.responses = responses
        self._response_index = 0
    
    def set_failure(self, error: Exception | None = None) -> None:
        """Configure mock to fail."""
        self.should_fail = True
        self.failure_error = error
    
    def reset(self) -> None:
        """Reset mock state."""
        self.calls = []
        self.responses = []
        self._response_index = 0
        self.should_fail = False
        self.failure_error = None


# Singleton mock instance
mock_litellm = MockLiteLLM()
