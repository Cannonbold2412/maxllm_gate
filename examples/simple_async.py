"""
Simple async-only example for maxllm_gate.

Shows the minimal interface: just import maxllm_gate and use async.
"""

import asyncio
from maxllm_gate import maxllm_gate


async def main():
    """Simple async chat example."""
    
    # Initialize with config file
    async with maxllm_gate.from_config("config.yaml") as client:
        # Simple chat
        response = await client.chat("gpt-4o-mini", "Hello, how are you?")
        print(f"Response: {response.content}")
        print(f"Latency: {response.latency:.2f}s")
        
        # With structured messages
        response = await client.chat(
            "gpt-4o-mini",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=100
        )
        print(f"\nStructured response: {response.content}")
        
        # Streaming
        print("\nStreaming response:")
        async for chunk in client.chat_stream("gpt-4o-mini", "Tell me a short joke"):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    print("maxllm_gate - Async-only Interface")
    print("=" * 50)
    asyncio.run(main())
