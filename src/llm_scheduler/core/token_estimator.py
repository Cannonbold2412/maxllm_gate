"""Token estimation using tiktoken and heuristics."""

from typing import Any
import tiktoken


class TokenEstimator:
    """
    Estimates token count for requests before execution.
    
    Uses tiktoken for accurate OpenAI model estimation and
    applies heuristics for other providers.
    """
    
    # Approximate tokens per character for different model families
    CHARS_PER_TOKEN = {
        "gpt": 4.0,
        "claude": 3.5,
        "llama": 3.8,
        "mixtral": 3.8,
        "gemma": 4.0,
        "default": 4.0,
    }
    
    # Output token multipliers based on task type (conservative estimates)
    OUTPUT_MULTIPLIERS = {
        "chat": 1.5,  # Response ~1.5x input for conversational
        "code": 2.0,  # Code generation tends to be longer
        "summarization": 0.3,  # Summaries are shorter
        "default": 1.5,
    }
    
    def __init__(self):
        self._encoders: dict[str, tiktoken.Encoding] = {}
    
    def _get_encoder(self, model: str) -> tiktoken.Encoding | None:
        """Get tiktoken encoder for a model, with caching."""
        if model in self._encoders:
            return self._encoders[model]
        
        try:
            encoder = tiktoken.encoding_for_model(model)
            self._encoders[model] = encoder
            return encoder
        except KeyError:
            # Model not supported by tiktoken
            try:
                # Fall back to cl100k_base (GPT-4, GPT-3.5-turbo)
                encoder = tiktoken.get_encoding("cl100k_base")
                self._encoders[model] = encoder
                return encoder
            except Exception:
                return None
    
    def _estimate_by_chars(self, text: str, model: str) -> int:
        """Estimate tokens using character count heuristic."""
        model_lower = model.lower()
        
        chars_per_token = self.CHARS_PER_TOKEN["default"]
        for prefix, ratio in self.CHARS_PER_TOKEN.items():
            if prefix in model_lower:
                chars_per_token = ratio
                break
        
        return max(1, int(len(text) / chars_per_token))
    
    def count_tokens(self, text: str, model: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: The text to count tokens for
            model: Model name for tokenizer selection
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        encoder = self._get_encoder(model)
        if encoder:
            return len(encoder.encode(text))
        
        return self._estimate_by_chars(text, model)
    
    def estimate_messages_tokens(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> int:
        """
        Estimate tokens for a list of chat messages.
        
        Accounts for message formatting overhead.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            
        Returns:
            Estimated token count
        """
        total = 0
        
        # Per-message overhead (role, formatting)
        per_message_overhead = 4
        
        for message in messages:
            total += per_message_overhead
            
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content, model)
            elif isinstance(content, list):
                # Handle multi-modal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""), model)
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        # Approximate image token cost
                        total += 85  # Base cost for low-detail
            
            # Role tokens
            role = message.get("role", "")
            total += self.count_tokens(role, model)
        
        # Conversation overhead
        total += 3  # Priming tokens
        
        return total
    
    def estimate_output_tokens(
        self,
        input_tokens: int,
        max_tokens: int | None = None,
        task_type: str = "default",
    ) -> int:
        """
        Estimate expected output tokens.
        
        Uses a conservative estimate based on input size and task type.
        
        Args:
            input_tokens: Number of input tokens
            max_tokens: Maximum tokens if specified
            task_type: Type of task (chat, code, summarization)
            
        Returns:
            Estimated output tokens
        """
        multiplier = self.OUTPUT_MULTIPLIERS.get(
            task_type, 
            self.OUTPUT_MULTIPLIERS["default"]
        )
        
        estimated = int(input_tokens * multiplier)
        
        # Apply max_tokens cap if specified
        if max_tokens is not None:
            estimated = min(estimated, max_tokens)
        
        # Ensure minimum of 1 token
        return max(1, estimated)
    
    def estimate_total_tokens(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int | None = None,
        task_type: str = "default",
        buffer_multiplier: float = 1.1,
    ) -> tuple[int, int, int]:
        """
        Estimate total tokens (input + expected output) for a request.
        
        Args:
            messages: Chat messages
            model: Model name
            max_tokens: Max output tokens
            task_type: Type of task
            buffer_multiplier: Safety buffer (default 10%)
            
        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens)
        """
        input_tokens = self.estimate_messages_tokens(messages, model)
        output_tokens = self.estimate_output_tokens(
            input_tokens, 
            max_tokens, 
            task_type
        )
        
        # Apply buffer for safety margin
        total = int((input_tokens + output_tokens) * buffer_multiplier)
        
        return input_tokens, output_tokens, total


# Global estimator instance
token_estimator = TokenEstimator()
