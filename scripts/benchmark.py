"""Benchmark throughput under rate limits."""

import asyncio
import time
import argparse
from dataclasses import dataclass

import httpx


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    
    url: str
    model: str
    duration_seconds: int
    target_rps: float
    warmup_seconds: int = 5


async def benchmark_throughput(config: BenchmarkConfig):
    """Run throughput benchmark."""
    print(f"Warming up for {config.warmup_seconds}s...")
    
    async with httpx.AsyncClient() as client:
        # Warmup
        warmup_start = time.time()
        while time.time() - warmup_start < config.warmup_seconds:
            try:
                await client.post(
                    f"{config.url}/chat",
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 10,
                    },
                    timeout=30.0,
                )
            except Exception:
                pass
            await asyncio.sleep(1.0 / config.target_rps)
        
        print(f"Running benchmark for {config.duration_seconds}s at ~{config.target_rps} RPS...")
        
        # Benchmark
        completed = 0
        failed = 0
        total_latency = 0.0
        
        start = time.time()
        request_interval = 1.0 / config.target_rps
        
        while time.time() - start < config.duration_seconds:
            request_start = time.time()
            
            try:
                response = await client.post(
                    f"{config.url}/chat",
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": "Count to 3"}],
                        "max_tokens": 50,
                    },
                    timeout=60.0,
                )
                
                latency = time.time() - request_start
                
                if response.status_code == 200:
                    completed += 1
                    total_latency += latency
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
            
            # Rate limiting
            elapsed = time.time() - request_start
            sleep_time = request_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        duration = time.time() - start
        
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Duration:           {duration:.1f}s")
        print(f"Completed:          {completed}")
        print(f"Failed:             {failed}")
        print(f"Actual RPS:         {completed / duration:.2f}")
        print(f"Target RPS:         {config.target_rps}")
        
        if completed > 0:
            print(f"Avg Latency:        {total_latency / completed:.3f}s")
        
        print(f"Success Rate:       {completed / (completed + failed) * 100:.1f}%")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Benchmark throughput")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rps", type=float, default=2.0)
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        url=args.url,
        model=args.model,
        duration_seconds=args.duration,
        target_rps=args.rps,
    )
    
    asyncio.run(benchmark_throughput(config))


if __name__ == "__main__":
    main()
