"""Load simulation script for testing rate limiting."""

import asyncio
import time
import argparse
import statistics
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class RequestResult:
    """Result of a single request."""
    
    success: bool
    status_code: int
    latency: float
    tokens_used: int | None = None
    error: str | None = None


@dataclass
class LoadTestResults:
    """Aggregate load test results."""
    
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def requests_per_second(self) -> float:
        if self.duration <= 0:
            return 0
        return self.total_requests / self.duration
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful / self.total_requests
    
    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0
        return statistics.mean(self.latencies)
    
    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0
        return statistics.median(self.latencies)
    
    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    priority: str,
    message: str,
) -> RequestResult:
    """Send a single chat request."""
    start = time.time()
    
    try:
        response = await client.post(
            f"{base_url}/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": message}
                ],
                "priority": priority,
                "max_tokens": 100,
            },
            timeout=120.0,
        )
        
        latency = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            return RequestResult(
                success=True,
                status_code=200,
                latency=latency,
                tokens_used=data.get("usage", {}).get("total_tokens"),
            )
        else:
            return RequestResult(
                success=False,
                status_code=response.status_code,
                latency=latency,
                error=response.text,
            )
            
    except Exception as e:
        return RequestResult(
            success=False,
            status_code=0,
            latency=time.time() - start,
            error=str(e),
        )


async def run_load_test(
    base_url: str,
    num_requests: int,
    concurrency: int,
    model: str,
    priority_distribution: dict[str, float],
) -> LoadTestResults:
    """Run the load test."""
    results = LoadTestResults()
    results.start_time = time.time()
    
    # Generate request configs
    requests_config = []
    for i in range(num_requests):
        # Select priority based on distribution
        import random
        rand = random.random()
        cumulative = 0
        priority = "medium"
        for p, weight in priority_distribution.items():
            cumulative += weight
            if rand < cumulative:
                priority = p
                break
        
        requests_config.append({
            "priority": priority,
            "message": f"Test request {i+1}: Tell me a very short joke.",
        })
    
    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(config):
            async with semaphore:
                return await send_request(
                    client,
                    base_url,
                    model,
                    config["priority"],
                    config["message"],
                )
        
        tasks = [limited_request(cfg) for cfg in requests_config]
        
        print(f"Starting {num_requests} requests with concurrency {concurrency}...")
        
        request_results = await asyncio.gather(*tasks)
        
        for result in request_results:
            results.total_requests += 1
            if result.success:
                results.successful += 1
                results.latencies.append(result.latency)
            else:
                results.failed += 1
                print(f"  Failed: {result.error[:100] if result.error else 'Unknown'}")
    
    results.end_time = time.time()
    return results


def print_results(results: LoadTestResults):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Duration:              {results.duration:.2f}s")
    print(f"Total Requests:        {results.total_requests}")
    print(f"Successful:            {results.successful}")
    print(f"Failed:                {results.failed}")
    print(f"Success Rate:          {results.success_rate:.1%}")
    print(f"Requests/Second:       {results.requests_per_second:.2f}")
    print("-" * 60)
    print("LATENCY (successful requests)")
    print("-" * 60)
    if results.latencies:
        print(f"Average:               {results.avg_latency:.3f}s")
        print(f"P50 (median):          {results.p50_latency:.3f}s")
        print(f"P99:                   {results.p99_latency:.3f}s")
        print(f"Min:                   {min(results.latencies):.3f}s")
        print(f"Max:                   {max(results.latencies):.3f}s")
    else:
        print("No successful requests")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Load test the LLM Rate Limit Scheduler"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the scheduler",
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=100,
        help="Number of requests to send",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Concurrent requests",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="Model to use",
    )
    parser.add_argument(
        "--high-priority",
        type=float,
        default=0.1,
        help="Fraction of high priority requests",
    )
    parser.add_argument(
        "--low-priority",
        type=float,
        default=0.2,
        help="Fraction of low priority requests",
    )
    
    args = parser.parse_args()
    
    priority_dist = {
        "high": args.high_priority,
        "medium": 1.0 - args.high_priority - args.low_priority,
        "low": args.low_priority,
    }
    
    print(f"Load Test Configuration:")
    print(f"  URL: {args.url}")
    print(f"  Requests: {args.requests}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Model: {args.model}")
    print(f"  Priority distribution: {priority_dist}")
    
    results = asyncio.run(run_load_test(
        base_url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency,
        model=args.model,
        priority_distribution=priority_dist,
    ))
    
    print_results(results)


if __name__ == "__main__":
    main()
