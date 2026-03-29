"""Example: Using priority requests."""

import asyncio
import httpx
import time


async def demo_priorities():
    """Demonstrate priority-based scheduling."""
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        print("Sending requests with different priorities...")
        print("High priority requests should complete before low priority.\n")
        
        # Create a mix of requests
        requests = [
            ("low", "Low priority request 1"),
            ("low", "Low priority request 2"),
            ("medium", "Medium priority request 1"),
            ("high", "HIGH PRIORITY - Urgent request"),
            ("low", "Low priority request 3"),
            ("medium", "Medium priority request 2"),
            ("high", "HIGH PRIORITY - Critical task"),
            ("low", "Low priority request 4"),
        ]
        
        async def send_request(priority: str, message: str, index: int):
            start = time.time()
            
            response = await client.post(
                f"{base_url}/chat",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": message}],
                    "priority": priority,
                    "max_tokens": 50,
                },
                timeout=120.0,
            )
            
            elapsed = time.time() - start
            
            if response.status_code == 200:
                print(f"[{priority.upper():6}] Request {index}: completed in {elapsed:.2f}s")
            else:
                print(f"[{priority.upper():6}] Request {index}: FAILED - {response.status_code}")
            
            return priority, elapsed
        
        # Send all requests concurrently
        tasks = [
            send_request(priority, message, i + 1)
            for i, (priority, message) in enumerate(requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        print("\n" + "=" * 50)
        print("PRIORITY ANALYSIS")
        print("=" * 50)
        
        by_priority = {"high": [], "medium": [], "low": []}
        for priority, elapsed in results:
            by_priority[priority].append(elapsed)
        
        for priority in ["high", "medium", "low"]:
            times = by_priority[priority]
            if times:
                avg = sum(times) / len(times)
                print(f"{priority.upper():8}: avg={avg:.2f}s, count={len(times)}")
        
        print("\nNote: High priority requests are scheduled before lower priority")
        print("ones, but actual completion time depends on LLM provider latency.")


async def demo_queue_behavior():
    """Demonstrate queue behavior under load."""
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        print("\nDemonstrating queue behavior under load...")
        print("Sending 20 requests rapidly to fill the queue.\n")
        
        async def send(i: int, priority: str):
            start = time.time()
            try:
                response = await client.post(
                    f"{base_url}/chat",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": f"Say 'Request {i}'"}],
                        "priority": priority,
                        "max_tokens": 20,
                    },
                    timeout=180.0,
                )
                elapsed = time.time() - start
                return f"Request {i} ({priority}): {elapsed:.2f}s"
            except Exception as e:
                return f"Request {i} ({priority}): ERROR - {e}"
        
        # Create mixed priority batch
        tasks = []
        for i in range(20):
            if i % 5 == 0:
                priority = "high"
            elif i % 3 == 0:
                priority = "low"
            else:
                priority = "medium"
            tasks.append(send(i + 1, priority))
        
        # Check queue before
        status_before = await client.get(f"{base_url}/status")
        print(f"Queue before: {status_before.json()['queue']['queue_size']}")
        
        # Send all
        results = await asyncio.gather(*tasks)
        
        # Check queue after
        status_after = await client.get(f"{base_url}/status")
        print(f"Queue after: {status_after.json()['queue']['queue_size']}")
        
        print("\nCompleted requests:")
        for result in sorted(results):
            print(f"  {result}")


if __name__ == "__main__":
    print("LLM Rate Limit Scheduler - Priority Requests Demo")
    print("=" * 50)
    print("Make sure the scheduler is running with API keys configured.")
    print("=" * 50 + "\n")
    
    asyncio.run(demo_priorities())
    asyncio.run(demo_queue_behavior())
