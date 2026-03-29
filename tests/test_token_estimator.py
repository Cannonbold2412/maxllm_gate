"""Tests for token estimator."""

import pytest

from llm_scheduler.core.token_estimator import TokenEstimator, token_estimator


class TestTokenEstimator:
    """Test suite for TokenEstimator."""
    
    def test_count_tokens_empty(self):
        """Empty string returns 0."""
        estimator = TokenEstimator()
        assert estimator.count_tokens("", "gpt-4") == 0
    
    def test_count_tokens_simple(self):
        """Count tokens in simple text."""
        estimator = TokenEstimator()
        
        text = "Hello, world!"
        tokens = estimator.count_tokens(text, "gpt-4")
        
        assert tokens > 0
        assert tokens < len(text)  # Should be less than char count
    
    def test_count_tokens_longer(self):
        """Token count scales with text length."""
        estimator = TokenEstimator()
        
        short = "Hello"
        long = "Hello " * 100
        
        short_tokens = estimator.count_tokens(short, "gpt-4")
        long_tokens = estimator.count_tokens(long, "gpt-4")
        
        assert long_tokens > short_tokens
    
    def test_estimate_messages_tokens(self):
        """Estimate tokens for chat messages."""
        estimator = TokenEstimator()
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        
        tokens = estimator.estimate_messages_tokens(messages, "gpt-4")
        
        assert tokens > 0
        assert tokens > 10  # Should include overhead
    
    def test_estimate_messages_multimodal(self):
        """Handle multimodal content."""
        estimator = TokenEstimator()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            },
        ]
        
        tokens = estimator.estimate_messages_tokens(messages, "gpt-4-vision")
        
        assert tokens > 85  # At least image base cost
    
    def test_estimate_output_tokens(self):
        """Estimate output tokens based on input."""
        estimator = TokenEstimator()
        
        output = estimator.estimate_output_tokens(100)
        assert output > 0
        
        # With max_tokens cap
        capped = estimator.estimate_output_tokens(100, max_tokens=50)
        assert capped <= 50
    
    def test_estimate_output_task_types(self):
        """Different task types affect estimation."""
        estimator = TokenEstimator()
        
        chat = estimator.estimate_output_tokens(100, task_type="chat")
        code = estimator.estimate_output_tokens(100, task_type="code")
        summary = estimator.estimate_output_tokens(100, task_type="summarization")
        
        assert code > chat  # Code tends to be longer
        assert summary < chat  # Summaries are shorter
    
    def test_estimate_total_tokens(self):
        """Estimate total with buffer."""
        estimator = TokenEstimator()
        
        messages = [
            {"role": "user", "content": "Write a poem about coding."},
        ]
        
        input_t, output_t, total = estimator.estimate_total_tokens(
            messages=messages,
            model="gpt-4",
            buffer_multiplier=1.1,
        )
        
        assert input_t > 0
        assert output_t > 0
        assert total >= input_t + output_t  # Buffer applied
    
    def test_global_instance(self):
        """Global instance works."""
        tokens = token_estimator.count_tokens("Hello", "gpt-4")
        assert tokens > 0
