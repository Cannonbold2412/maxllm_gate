"""
maxllm_gate - Intelligent LLM Rate Limit Scheduler

Async-only interface for rate-limited LLM requests.

Usage:
    from maxllm_gate import maxllm_gate
    
    # Async usage
    async with maxllm_gate.from_config("config.yaml") as client:
        response = await client.chat("gpt-4", "Hello!")
        print(response.content)
"""

from maxllm_gate.client import maxllm_gate_async as maxllm_gate

__version__ = "0.3.0"
__all__ = ["maxllm_gate"]
