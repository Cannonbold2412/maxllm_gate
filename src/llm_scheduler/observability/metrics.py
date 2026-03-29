"""Prometheus metrics definitions."""

from prometheus_client import Counter, Histogram, Gauge, Info


class Metrics:
    """Application metrics."""
    
    def __init__(self):
        # Request counters
        self.requests_enqueued = Counter(
            "llm_scheduler_requests_enqueued_total",
            "Total requests enqueued",
        )
        
        self.requests_completed = Counter(
            "llm_scheduler_requests_completed_total",
            "Total requests completed successfully",
        )
        
        self.requests_failed = Counter(
            "llm_scheduler_requests_failed_total",
            "Total requests failed",
        )
        
        self.deferred_requests = Counter(
            "llm_scheduler_deferred_requests_total",
            "Requests that were deferred due to rate limits",
        )
        
        # Latency histograms
        self.request_latency = Histogram(
            "llm_scheduler_request_latency_seconds",
            "End-to-end request latency",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )
        
        self.queue_wait_time = Histogram(
            "llm_scheduler_queue_wait_seconds",
            "Time spent waiting in queue",
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
        )
        
        self.dispatch_latency = Histogram(
            "llm_scheduler_dispatch_latency_seconds",
            "LiteLLM dispatch latency",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )
        
        # Gauges
        self.queue_size = Gauge(
            "llm_scheduler_queue_size",
            "Current queue size",
        )
        
        self.active_requests = Gauge(
            "llm_scheduler_active_requests",
            "Currently processing requests",
        )
        
        # Token tracking
        self.tokens_used = Counter(
            "llm_scheduler_tokens_used_total",
            "Total tokens used",
            ["key_id", "provider"],
        )
        
        self.tokens_available = Gauge(
            "llm_scheduler_tokens_available",
            "Available tokens per key",
            ["key_id"],
        )
        
        # Retry tracking
        self.retries = Counter(
            "llm_scheduler_retries_total",
            "Total retry attempts",
            ["key_id", "reason"],
        )
        
        # Info
        self.info = Info(
            "llm_scheduler",
            "Scheduler information",
        )
        self.info.info({
            "version": "0.1.0",
        })


# Global metrics instance
metrics = Metrics()
