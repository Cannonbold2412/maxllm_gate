"""Pytest configuration and fixtures."""

import asyncio
import pytest
from typing import AsyncGenerator

# Try to import server components (optional)
try:
    from llm_scheduler.config import settings, APIKeyConfig
    from llm_scheduler.core.scheduler import Scheduler as ServerScheduler
    from llm_scheduler.rate_limiting.token_bucket import TokenBucket
    from llm_scheduler.rate_limiting.key_manager import KeyManager
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False
    settings = None
    APIKeyConfig = None
    ServerScheduler = None
    TokenBucket = None
    KeyManager = None


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def token_bucket():
    """Create a test token bucket."""
    if not SERVER_AVAILABLE:
        pytest.skip("Server components not installed")
    return TokenBucket.from_per_minute(60)


@pytest.fixture
def key_manager():
    """Create a test key manager with mock keys."""
    if not SERVER_AVAILABLE:
        pytest.skip("Server components not installed")
    
    manager = KeyManager()
    
    manager.register_key(APIKeyConfig(
        key_id="test-key-1",
        api_key="sk-test-1",
        provider="openai",
        models=["gpt-4o-mini", "gpt-4o"],
        tpm_limit=10000,
        rpm_limit=100,
    ))
    
    manager.register_key(APIKeyConfig(
        key_id="test-key-2",
        api_key="sk-test-2",
        provider="groq",
        models=["mixtral-8x7b-32768", "llama-3.1-70b"],
        tpm_limit=30000,
        rpm_limit=30,
    ))
    
    return manager


@pytest.fixture
async def scheduler() -> AsyncGenerator:
    """Create and start a test scheduler."""
    if not SERVER_AVAILABLE:
        pytest.skip("Server components not installed")
    
    sched = ServerScheduler()
    
    sched.key_manager.register_key(APIKeyConfig(
        key_id="test-key-1",
        api_key="sk-test-1",
        provider="openai",
        models=["gpt-4o-mini"],
        tpm_limit=10000,
        rpm_limit=100,
    ))
    
    await sched.start()
    yield sched
    await sched.stop()


@pytest.fixture
def sample_messages() -> list[dict]:
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
