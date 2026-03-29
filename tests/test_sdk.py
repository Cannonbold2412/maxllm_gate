"""Tests for maxllm_gate SDK client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from maxllm_gate import (
    maxllm_gate,
)
from maxllm_gate.client import ChatResponse, Message
from maxllm_gate.config import maxllm_gate_config, KeyConfig
from maxllm_gate.validation import validate_chat_request
from maxllm_gate.validation import ChatMessage, ChatRequest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Create a test config with mock keys."""
    return maxllm_gate_config(
        keys=[
            KeyConfig(
                api_key="test-key-1",
                provider="openai",
                models=["gpt-4o-mini", "gpt-4"],
                tpm_limit=100000,
                rpm_limit=60,
                key_id="test-openai-1",
            ),
            KeyConfig(
                api_key="test-key-2",
                provider="groq",
                models=["mixtral-8x7b-32768"],
                tpm_limit=50000,
                rpm_limit=30,
                key_id="test-groq-1",
            ),
        ],
        strategy="balanced",
        default_max_tokens=1024,
        default_temperature=0.7,
    )


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLLM completion response."""
    return MagicMock(
        model_dump=lambda: {
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25,
            },
        }
    )


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for input validation."""
    
    def test_validate_string_message(self):
        """String messages are converted to user message."""
        request = validate_chat_request(
            model="gpt-4",
            messages="Hello!",
        )
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hello!"
    
    def test_validate_list_messages(self):
        """List of dicts is properly validated."""
        request = validate_chat_request(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
        )
        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"
    
    def test_validate_empty_content_fails(self):
        """Empty content should fail validation."""
        with pytest.raises(ValueError):
            validate_chat_request(
                model="gpt-4",
                messages=[{"role": "user", "content": ""}],
            )
    
    def test_validate_whitespace_content_fails(self):
        """Whitespace-only content should fail validation."""
        with pytest.raises(ValueError):
            validate_chat_request(
                model="gpt-4",
                messages=[{"role": "user", "content": "   "}],
            )
    
    def test_validate_empty_model_fails(self):
        """Empty model name should fail."""
        with pytest.raises(ValueError):
            validate_chat_request(
                model="",
                messages="Hello",
            )
    
    def test_validate_invalid_role_fails(self):
        """Invalid role should fail."""
        with pytest.raises(ValueError):
            validate_chat_request(
                model="gpt-4",
                messages=[{"role": "invalid", "content": "Hello"}],
            )
    
    def test_validate_temperature_range(self):
        """Temperature must be 0-2."""
        with pytest.raises(ValueError):
            validate_chat_request(
                model="gpt-4",
                messages="Hello",
                temperature=2.5,
            )
    
    def test_validate_max_tokens_positive(self):
        """Max tokens must be positive."""
        with pytest.raises(ValueError):
            validate_chat_request(
                model="gpt-4",
                messages="Hello",
                max_tokens=-1,  # Negative value
            )
    
    def test_validate_priority(self):
        """Priority must be high/medium/low."""
        with pytest.raises(ValueError):
            validate_chat_request(
                model="gpt-4",
                messages="Hello",
                priority="urgent",  # Invalid
            )


class TestChatMessage:
    """Tests for ChatMessage model."""
    
    def test_valid_roles(self):
        """All valid roles should work."""
        for role in ["system", "user", "assistant", "function", "tool"]:
            msg = ChatMessage(role=role, content="Test")
            assert msg.role == role
    
    def test_with_optional_fields(self):
        """Optional fields should be accepted."""
        msg = ChatMessage(
            role="assistant",
            content="Hello",
            name="assistant_1",
        )
        assert msg.name == "assistant_1"


# =============================================================================
# Client Tests
# =============================================================================

class Testmaxllm_gateClient:
    """Tests for maxllm_gate client."""
    
    def test_init_with_config(self, mock_config):
        """Client initializes with config object."""
        client = maxllm_gate(config=mock_config)
        assert len(client.config.keys) == 2
        assert client.config.strategy == "balanced"
    
    def test_init_with_keys_list(self):
        """Client initializes with keys list."""
        client = maxllm_gate(
            keys=[
                {
                    "api_key": "test-key",
                    "provider": "openai",
                    "models": ["gpt-4"],
                }
            ]
        )
        assert len(client.config.keys) == 1
    
    def test_repr(self, mock_config):
        """String representation is correct."""
        client = maxllm_gate(config=mock_config)
        rep = repr(client)
        assert "maxllm_gate" in rep
        assert "keys=2" in rep
        assert "strategy=balanced" in rep
    
    def test_add_key_at_runtime(self, mock_config):
        """Keys can be added at runtime."""
        client = maxllm_gate(config=mock_config)
        initial_count = len(client.config.keys)
        
        client.add_key(
            api_key="new-key",
            provider="anthropic",
            models=["claude-3"],
        )
        
        assert len(client.config.keys) == initial_count + 1
    
    @pytest.mark.asyncio
    @patch("maxllm_gate.scheduler.litellm.acompletion")
    async def test_chat_sync(self, mock_acompletion, mock_config, mock_litellm_response):
        """Async chat works correctly."""
        mock_acompletion.return_value = mock_litellm_response
        
        client = maxllm_gate(config=mock_config)
        response = await client.chat("gpt-4o-mini", "Hello!", validate=False)
        
        assert isinstance(response, ChatResponse)
        assert "Hello" in response.content
        assert response.model == "gpt-4o-mini"
        assert response.usage is not None
    
    @pytest.mark.asyncio
    @patch("maxllm_gate.scheduler.litellm.acompletion")
    async def test_chat_with_messages_list(self, mock_acompletion, mock_config, mock_litellm_response):
        """Chat with message list works."""
        mock_acompletion.return_value = mock_litellm_response
        
        client = maxllm_gate(config=mock_config)
        response = await client.chat(
            "gpt-4o-mini",
            [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello!"},
            ],
            validate=False,
        )
        
        assert response.content is not None
    
    @pytest.mark.asyncio
    @patch("maxllm_gate.scheduler.litellm.acompletion")
    async def test_chat_with_message_objects(self, mock_acompletion, mock_config, mock_litellm_response):
        """Chat with Message objects works."""
        mock_acompletion.return_value = mock_litellm_response
        
        client = maxllm_gate(config=mock_config)
        response = await client.chat(
            "gpt-4o-mini",
            [Message(role="user", content="Hello!")],
            validate=False,
        )
        
        assert response.content is not None
    
    def test_status(self, mock_config):
        """Status returns scheduler status."""
        client = maxllm_gate(config=mock_config)
        # Ensure initialized
        client._ensure_initialized()
        
        status = client.status()
        assert "strategy" in status
        assert "capacity" in status
    
    def test_capacity(self, mock_config):
        """Capacity returns rate limiter capacity."""
        client = maxllm_gate(config=mock_config)
        client._ensure_initialized()
        
        capacity = client.capacity()
        assert "keys" in capacity
    
    def test_latency_stats(self, mock_config):
        """Latency returns per-key stats."""
        client = maxllm_gate(config=mock_config)
        client._ensure_initialized()
        
        latency = client.latency()
        # Keys exist but may have empty stats initially
        assert isinstance(latency, dict)
    
    def test_scores(self, mock_config):
        """Scores returns balanced scores."""
        client = maxllm_gate(config=mock_config)
        client._ensure_initialized()
        
        scores = client.scores()
        assert isinstance(scores, dict)
        
        # Check score structure for each key
        for key_id, score_data in scores.items():
            assert "total_score" in score_data
            assert "utilization" in score_data
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_config):
        """Async context manager works for graceful shutdown."""
        async with maxllm_gate(config=mock_config) as client:
            client._ensure_initialized()
            assert len(client.config.keys) == 2
        # Shutdown called automatically


class Testmaxllm_gate_asyncClient:
    """Tests for async maxllm_gate client."""
    
    @pytest.mark.asyncio
    @patch("maxllm_gate.scheduler.litellm.acompletion")
    async def test_async_chat(self, mock_acompletion, mock_config, mock_litellm_response):
        """Async chat works correctly."""
        mock_acompletion.return_value = mock_litellm_response
        
        client = maxllm_gate(config=mock_config)
        response = await client.chat("gpt-4o-mini", "Hello!", validate=False)
        
        assert isinstance(response, ChatResponse)
        assert response.content is not None
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_config):
        """Async context manager works."""
        async with maxllm_gate(config=mock_config) as client:
            client._ensure_initialized()
            assert len(client.config.keys) == 2


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration loading."""
    
    def test_keyconfig_auto_id(self):
        """KeyConfig generates ID if not provided."""
        key = KeyConfig(
            api_key="test-key",
            provider="openai",
            models=["gpt-4"],
        )
        assert key.key_id is not None
        assert key.key_id.startswith("openai-")
    
    def test_keyconfig_custom_id(self):
        """KeyConfig uses provided ID."""
        key = KeyConfig(
            api_key="test-key",
            provider="openai",
            models=["gpt-4"],
            key_id="my-custom-id",
        )
        assert key.key_id == "my-custom-id"
    
    def test_config_from_dict(self):
        """Config loads from dictionary."""
        config = maxllm_gate_config.from_dict({
            "keys": [
                {
                    "api_key": "test-key",
                    "provider": "openai",
                    "models": ["gpt-4"],
                }
            ],
            "strategy": "round_robin",
        })
        
        assert len(config.keys) == 1
        assert config.strategy == "round_robin"
    
    def test_config_to_dict(self, mock_config):
        """Config serializes to dictionary."""
        data = mock_config.to_dict()
        
        assert "keys" in data
        assert "strategy" in data
        assert data["strategy"] == "balanced"


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for rate limiting functionality."""
    
    def test_key_registration(self, mock_config):
        """Keys are properly registered."""
        client = maxllm_gate(config=mock_config)
        client._ensure_initialized()
        
        capacity = client.capacity()
        assert "test-openai-1" in capacity["keys"]
        assert "test-groq-1" in capacity["keys"]
    
    def test_capacity_tracking(self, mock_config):
        """Capacity is properly tracked."""
        client = maxllm_gate(config=mock_config)
        client._ensure_initialized()
        
        # Initial capacity should be full
        capacity = client.capacity()
        key_data = capacity["keys"]["test-openai-1"]
        
        assert key_data["tpm_capacity"] == 100000
        assert key_data["utilization"] == 0.0
    
    def test_strategy_selection(self, mock_config):
        """Strategy is used for key selection."""
        client = maxllm_gate(config=mock_config)
        client._ensure_initialized()
        
        # With balanced strategy, should select based on score
        key_state, wait_time = client._rate_limiter.select_key(
            model="gpt-4o-mini",
            estimated_tokens=100,
            strategy="balanced",
        )
        
        assert key_state is not None
        assert key_state.key_config.key_id == "test-openai-1"


# =============================================================================
# Integration Tests (with mocked LiteLLM)
# =============================================================================

class TestIntegration:
    """Integration tests with mocked LLM calls."""
    
    @pytest.mark.asyncio
    @patch("maxllm_gate.scheduler.litellm.acompletion")
    async def test_full_request_flow(self, mock_acompletion, mock_config, mock_litellm_response):
        """Test complete request flow."""
        mock_acompletion.return_value = mock_litellm_response
        
        async with maxllm_gate(config=mock_config) as client:
            # Make a request
            response = await client.chat(
                model="gpt-4o-mini",
                messages="Hello!",
                max_tokens=100,
                temperature=0.5,
                validate=False,
            )
            
            # Verify response
            assert response.content is not None
            assert response.latency is not None
            assert response.latency > 0
            
            # Verify LiteLLM was called correctly
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.5
    
    @pytest.mark.asyncio
    @patch("maxllm_gate.scheduler.litellm.acompletion")
    async def test_retry_on_failure(self, mock_acompletion, mock_config, mock_litellm_response):
        """Test retry behavior on transient failures."""
        # Fail twice, then succeed
        mock_acompletion.side_effect = [
            Exception("Transient error"),
            Exception("Another error"),
            mock_litellm_response,
        ]
        
        client = maxllm_gate(config=mock_config)
        response = await client.chat("gpt-4o-mini", "Hello!", validate=False)
        
        assert response.content is not None
        assert mock_acompletion.call_count == 3
    
    @pytest.mark.asyncio
    @patch("maxllm_gate.scheduler.litellm.acompletion")
    async def test_provider_prefix(self, mock_acompletion, mock_config, mock_litellm_response):
        """Test that provider prefixes are added correctly."""
        mock_acompletion.return_value = mock_litellm_response
        
        client = maxllm_gate(config=mock_config)
        
        # Request groq model
        response = await client.chat(
            model="mixtral-8x7b-32768",
            messages="Hello!",
            validate=False,
        )
        
        # Check the model passed to LiteLLM
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "groq/mixtral-8x7b-32768"
