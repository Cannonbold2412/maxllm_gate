"""
MAXLLM - Intelligent LLM Rate Limit Scheduler

Simple SDK-style interface for using the scheduler.

Usage:
    from maxllm import MAXLLM
    
    # From config file
    client = MAXLLM.from_config("config.yaml")
    
    # Or inline config
    client = MAXLLM(
        keys=[
            {"api_key": "sk-...", "provider": "openai", "models": ["gpt-4o-mini"]}
        ]
    )
    
    # Use like OpenAI client
    response = client.chat("gpt-4o-mini", "Hello, how are you?")
    print(response.content)
    
    # With context manager (auto-shutdown)
    with MAXLLM.from_config("config.yaml") as client:
        response = client.chat("gpt-4", "Hello!")
    
    # Async usage
    async with MAXLLMAsync.from_config("config.yaml") as client:
        response = await client.chat("gpt-4", "Hello!")
        
    # Handle capacity errors
    from maxllm import CapacityExhaustedError
    try:
        response = client.chat("gpt-4", "Hello!")
    except CapacityExhaustedError as e:
        print(f"System overloaded, wait time: {e.wait_time}s")
"""

from maxllm.client import MAXLLM, MAXLLMAsync, ChatResponse, Message
from maxllm.config import MAXLLMConfig, KeyConfig
from maxllm.validation import ChatRequest, ChatMessage, validate_chat_request
from maxllm.scheduler import CapacityExhaustedError, CircuitBreakerOpenError

# Optional Redis backend
try:
    from maxllm.redis_backend import RedisBackend, RedisConfig, HybridRateLimiter
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    RedisBackend = None
    RedisConfig = None
    HybridRateLimiter = None

__version__ = "0.3.0"
__all__ = [
    # Main client
    "MAXLLM",
    "MAXLLMAsync",
    "ChatResponse",
    "Message",
    # Config
    "MAXLLMConfig",
    "KeyConfig",
    # Validation
    "ChatRequest",
    "ChatMessage",
    "validate_chat_request",
    # Exceptions
    "CapacityExhaustedError",
    "CircuitBreakerOpenError",
    # Redis (optional)
    "RedisBackend",
    "RedisConfig",
    "HybridRateLimiter",
    # Version
    "__version__",
]
