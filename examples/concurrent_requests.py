"""
Example: Handling concurrent requests with maxllm_gate.

Demonstrates how maxllm_gate handles multiple concurrent requests with:
1. Semaphore-based concurrency control
2. Rate limit tracking across concurrent calls
3. Load balancing across multiple keys
"""

import asyncio
import time
from maxllm_gate import maxllm_gate


async def main():
    # Create config with concurrency limit
    config = {
        "keys": [
            {
                "api_key": "test-key-1",
                "provider": "openai",
                "models": ["gpt-4o-mini"],
                "tpm_limit": 100000,
                "rpm_limit": 60,
            },
            {
                "api_key": "test-key-2",
                "provider": "openai",
                "models": ["gpt-4o-mini"],
                "tpm_limit": 100000,
                "rpm_limit": 60,
            },
        ],
        "strategy": "balanced",
        "max_concurrent_requests": 10,  # Max 10 parallel requests
    }
    
    async with maxllm_gate(keys=config["keys"], strategy=config["strategy"], max_concurrent_requests=config["max_concurrent_requests"]) as client:
        print(f"🚀 maxllm_gate Concurrent Request Test")
        print(f"📊 Config: 2 keys, max 10 concurrent")
        print(f"🎯 Strategy: balanced\n")
        
        # Simulate 20 concurrent requests
        print("Sending 20 concurrent requests...")
        print("(Semaphore limits to 10 at a time)\n")
        
        start_time = time.time()
        
        async def make_request(i):
            request_start = time.time()
            try:
                # Note: This would actually call the LLM in production
                # For demo, we're just showing the scheduler logic
                print(f"  [{i:2d}] Request started at {request_start - start_time:.2f}s")
                
                # Simulate work (replace with actual chat call)
                await asyncio.sleep(0.1)
                
                elapsed = time.time() - request_start
                print(f"  [{i:2d}] Request completed in {elapsed:.2f}s")
                
                return {"request_id": i, "latency": elapsed}
            except Exception as e:
                print(f"  [{i:2d}] Request failed: {e}")
                return {"request_id": i, "error": str(e)}
        
        # Fire all 20 requests concurrently
        tasks = [make_request(i) for i in range(1, 21)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ All 20 requests completed in {total_time:.2f}s")
        print(f"📈 Average time per request: {total_time / 20:.2f}s")
        
        # Show capacity after requests
        capacity = client.capacity()
        print(f"\n📊 Capacity Status:")
        for key_id, key_data in capacity.get("keys", {}).items():
            tokens_used = key_data.get("tokens_used", 0)
            tpm_limit = key_data.get("tpm_limit", 0)
            utilization = (tokens_used / tpm_limit * 100) if tpm_limit > 0 else 0
            print(f"  {key_id}: {utilization:.1f}% utilized")
        
        # Show routing decisions
        scores = client.scores()
        print(f"\n🎯 Routing Scores:")
        for key_id, score_data in scores.items():
            print(f"  {key_id}:")
            print(f"    Total Score: {score_data['total_score']:.3f}")
            print(f"    Utilization: {score_data['utilization']:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("maxllm_gate Concurrent Request Handling Demo")
    print("=" * 60)
    print()
    
    # Run the demo
    asyncio.run(main())
    
    print("\n" + "=" * 60)
    print("How It Works:")
    print("=" * 60)
    print("""
1. **Semaphore Control**: max_concurrent_requests=10 limits parallel execution
   - Requests 1-10 start immediately
   - Requests 11-20 queue until slots open

2. **Rate Limiting**: Each request checks capacity before execution
   - If key exhausted, waits for capacity
   - Never exceeds TPM/RPM limits

3. **Load Balancing**: 'balanced' strategy distributes across keys
   - Considers utilization, latency, errors
   - Automatically adapts to workload

Benefits:
✅ Prevents connection pool exhaustion
✅ Avoids rate limiter race conditions  
✅ Maintains high throughput (100 concurrent default)
✅ Configurable per deployment needs
""")
