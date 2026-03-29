"""Basic usage example for LLM Rate Limit Scheduler."""

import asyncio
import httpx


async def main():
    """Demonstrate basic API usage."""
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # Check health
        print("Checking health...")
        health = await client.get(f"{base_url}/health")
        print(f"Health: {health.json()}")
        
        # Check capacity
        print("\nChecking capacity...")
        capacity = await client.get(f"{base_url}/capacity")
        print(f"Capacity: {capacity.json()}")
        
        # Send a chat request
        print("\nSending chat request...")
        response = await client.post(
            f"{base_url}/chat",
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                "priority": "medium",
                "max_tokens": 100,
                "temperature": 0.7,
            },
            timeout=60.0,
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['content']}")
            print(f"Model: {result['model']}")
            print(f"Usage: {result.get('usage')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Send high priority request
        print("\nSending high priority request...")
        response = await client.post(
            f"{base_url}/chat",
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "Quick! What's 2+2?"},
                ],
                "priority": "high",
                "max_tokens": 20,
            },
            timeout=60.0,
        )
        
        if response.status_code == 200:
            print(f"High priority response: {response.json()['content']}")
        
        # Check status
        print("\nChecking scheduler status...")
        status = await client.get(f"{base_url}/status")
        status_data = status.json()
        print(f"Scheduler running: {status_data['running']}")
        print(f"Queue size: {status_data['queue']['queue_size']}")


if __name__ == "__main__":
    print("LLM Rate Limit Scheduler - Basic Usage Example")
    print("=" * 50)
    print("Make sure the scheduler is running: uvicorn llm_scheduler.main:app")
    print("=" * 50 + "\n")
    
    asyncio.run(main())
