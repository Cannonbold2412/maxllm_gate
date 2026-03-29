"""Tests for the scheduler."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from llm_scheduler.core.scheduler import Scheduler, SchedulerError
from llm_scheduler.config import APIKeyConfig


class TestScheduler:
    """Test suite for Scheduler."""
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Scheduler starts and stops cleanly."""
        scheduler = Scheduler()
        
        # Register a mock key
        scheduler.key_manager.register_key(APIKeyConfig(
            key_id="test-key",
            api_key="sk-test",
            provider="openai",
            models=["gpt-4o-mini"],
            tpm_limit=10000,
            rpm_limit=100,
        ))
        
        await scheduler.start()
        assert scheduler._running is True
        
        await scheduler.stop()
        assert scheduler._running is False
    
    @pytest.mark.asyncio
    async def test_schedule_requires_running(self):
        """Schedule raises error if not running."""
        scheduler = Scheduler()
        
        with pytest.raises(SchedulerError, match="not running"):
            await scheduler.schedule(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
            )
    
    @pytest.mark.asyncio
    async def test_schedule_no_keys(self):
        """Schedule raises error if no keys for model."""
        scheduler = Scheduler()
        await scheduler.start()
        
        try:
            with pytest.raises(SchedulerError, match="No available keys"):
                await scheduler.schedule(
                    model="unknown-model",
                    messages=[{"role": "user", "content": "test"}],
                )
        finally:
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_get_status(self):
        """Get status returns expected structure."""
        scheduler = Scheduler()
        
        scheduler.key_manager.register_key(APIKeyConfig(
            key_id="test-key",
            api_key="sk-test",
            provider="openai",
            models=["gpt-4o-mini"],
            tpm_limit=10000,
            rpm_limit=100,
        ))
        
        await scheduler.start()
        
        try:
            status = scheduler.get_status()
            
            assert "running" in status
            assert "queue" in status
            assert "keys" in status
            assert "strategy" in status
            
            assert status["running"] is True
        finally:
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_schedule_with_mock_litellm(self):
        """Schedule successfully with mocked LiteLLM."""
        scheduler = Scheduler()
        
        scheduler.key_manager.register_key(APIKeyConfig(
            key_id="test-key",
            api_key="sk-test",
            provider="openai",
            models=["gpt-4o-mini"],
            tpm_limit=10000,
            rpm_limit=100,
        ))
        
        await scheduler.start()
        
        try:
            # Mock LiteLLM response
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {
                "id": "test-id",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {"content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
                mock.return_value = mock_response
                
                result = await scheduler.schedule(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hi"}],
                )
                
                assert result is not None
                assert "choices" in result
                mock.assert_called_once()
        finally:
            await scheduler.stop()
