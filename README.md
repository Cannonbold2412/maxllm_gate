# MAXLLM

> **Production-ready** intelligent LLM client with built-in rate limiting, smart routing, and distributed state support. Maximizes throughput and prevents 429 errors.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/yourusername/maxllm)

## Overview

**MAXLLM** is a production-ready LLM client that automatically manages rate limits across multiple API keys and providers. It works on top of [LiteLLM](https://github.com/BerriAI/litellm) as an intelligent scheduling and optimization layer.

```python
from maxllm import MAXLLM

# Load config and go
client = MAXLLM.from_config("config.yaml")

# Use like OpenAI client - rate limiting is automatic
response = client.chat("gpt-4o-mini", "Explain quantum computing")
print(response.content)
```

### What it does automatically:
- ✅ **Multi-key management** - Manages multiple API keys across providers (Groq, OpenAI, OpenRouter, etc.)
- ✅ **Real-time rate limiting** - Tracks TPM/RPM limits per key with token bucket algorithm
- ✅ **Smart routing** - 4 strategies: least_utilized, round_robin, latency_aware, **balanced** (NEW)
- ✅ **No 429 errors** - Defers requests when capacity exhausted instead of failing
- ✅ **Auto-retry** - Exponential backoff on transient failures
- ✅ **Streaming support** - Real async/sync streaming with proper token tracking
- ✅ **Input validation** - Pydantic models validate all inputs
- ✅ **Graceful shutdown** - Context manager support with proper cleanup
- ✅ **Production-ready** - Optional Redis backend for distributed state

## Installation

```bash
# Base installation
pip install maxllm

# With YAML config support
pip install maxllm[yaml]

# With Redis backend (for production/distributed deployments)
pip install maxllm[redis]

# With server mode (optional FastAPI gateway)
pip install maxllm[server]

# Everything (recommended for production)
pip install maxllm[all]
```

## Quick Start

### 1. Create config file

```yaml
# config.yaml
keys:
  groq-1:
    api_key: "gsk_your_groq_key"
    provider: groq
    models: [llama-3.1-70b-versatile, mixtral-8x7b-32768]
    tpm_limit: 30000
    rpm_limit: 30
    
  openai-1:
    api_key: "sk-your_openai_key"
    provider: openai
    models: [gpt-4o-mini, gpt-4o]
    tpm_limit: 90000
    rpm_limit: 500

# Routing strategy (see strategies section below)
strategy: balanced  # NEW: Smart combined routing
```

### 2. Use it

```python
from maxllm import MAXLLM

# Context manager ensures graceful shutdown
with MAXLLM.from_config("config.yaml") as client:
    # Simple chat
    response = client.chat("gpt-4o-mini", "Hello!")
    print(response.content)
    
    # With messages list
    response = client.chat("mixtral-8x7b-32768", [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about Python."},
    ])
    
    # Streaming
    for chunk in client.chat_stream("gpt-4o-mini", "Tell me a story"):
        print(chunk, end="", flush=True)
    
    # Check capacity and scores
    print(client.capacity())
    print(client.scores())  # See routing decisions
```

### Async Usage

```python
from maxllm import MAXLLMAsync
import asyncio

async def main():
    # Async context manager
    async with MAXLLMAsync.from_config("config.yaml") as client:
        # Single request
        response = await client.chat("gpt-4o-mini", "Hello!")
        print(response.content)
        
        # Concurrent requests - automatically load balanced
        tasks = [
            client.chat("gpt-4o-mini", f"Question {i}")
            for i in range(10)
        ]
        responses = await asyncio.gather(*tasks)
        
        # Async streaming
        async for chunk in client.chat_stream("gpt-4o-mini", "Tell a story"):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

## Configuration

### YAML Config (recommended)

```yaml
keys:
  # Multiple keys per provider for more capacity
  groq-1:
    api_key: "gsk_key_1"
    provider: groq
    models: [llama-3.1-70b-versatile]
    tpm_limit: 30000
    rpm_limit: 30
    
  groq-2:
    api_key: "gsk_key_2" 
    provider: groq
    models: [llama-3.1-70b-versatile]
    tpm_limit: 30000
    rpm_limit: 30

# Routing Strategy
strategy: balanced  # least_utilized | round_robin | latency_aware | balanced

# Defaults
default_max_tokens: 1024
default_temperature: 0.7
token_buffer: 1.1  # Safety margin for token estimation

# Retry Configuration
max_retries: 3
retry_base_delay: 1.0
retry_max_delay: 60.0

# Redis (optional - for production/distributed deployments)
redis_url: "redis://localhost:6379"
redis_prefix: "maxllm:"
```

### Routing Strategies

MAXLLM supports 4 routing strategies:

| Strategy | Best For | How It Works |
|----------|----------|--------------|
| **`balanced`** ⭐ | Production use | Combines utilization (40%), latency (35%), errors (15%), freshness (10%) into a weighted score. Automatically adapts to your workload. |
| `least_utilized` | Max throughput | Routes to key with most available capacity (TPM/RPM). |
| `round_robin` | Fair distribution | Cycles through keys evenly. |
| `latency_aware` | Low latency | Prefers keys with fastest response times. |

**Recommended:** Use `balanced` for production - it intelligently combines all factors.

```python
# See routing decisions in real-time
scores = client.scores()
print(scores)
# {
#   "groq-1": {
#     "total_score": 0.23,      # Lower = better
#     "utilization": 0.15,      # 15% capacity used
#     "latency_normalized": 0.08,
#     "latency_avg_ms": 245.5,
#     "error_penalty": 0.0,     # No recent errors
#     "freshness": 0.85
#   },
#   ...
# }
```

### Inline Config

```python
from maxllm import MAXLLM

client = MAXLLM(keys=[
    {
        "api_key": "gsk_...",
        "provider": "groq",
        "models": ["mixtral-8x7b-32768"],
        "tpm_limit": 30000,
        "rpm_limit": 30,
    },
    {
        "api_key": "sk-...",
        "provider": "openai", 
        "models": ["gpt-4o-mini"],
        "tpm_limit": 90000,
        "rpm_limit": 500,
    },
])
```

### Environment Variables

```bash
export MAXLLM_KEYS='{
  "groq-1": {"api_key": "gsk_...", "provider": "groq", "models": ["mixtral-8x7b-32768"], "tpm_limit": 30000, "rpm_limit": 30}
}'

# Then in Python
from maxllm import MAXLLM
client = MAXLLM.from_env()
```

## Supported Providers

MAXLLM works with any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers):

| Provider | Config Name | Example Models |
|----------|-------------|----------------|
| OpenAI | `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| Groq | `groq` | `llama-3.1-70b-versatile`, `mixtral-8x7b-32768` |
| OpenRouter | `openrouter` | `anthropic/claude-3-haiku`, `meta-llama/llama-3-70b` |
| Anthropic | `anthropic` | `claude-3-haiku-20240307`, `claude-3-5-sonnet-20241022` |
| Together AI | `together_ai` | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| Anyscale | `anyscale` | `meta-llama/Llama-3-70b-chat-hf` |
| Fireworks | `fireworks_ai` | `accounts/fireworks/models/llama-v3-70b-instruct` |
| NVIDIA NIM | `nvidia_nim` | Any NVIDIA NIM endpoint |
| Azure OpenAI | `azure` | Your Azure deployments |

### Provider Configuration

```yaml
keys:
  # OpenAI (no prefix needed)
  openai-1:
    api_key: "sk-..."
    provider: openai
    models: [gpt-4o-mini]
    
  # Groq
  groq-1:
    api_key: "gsk_..."
    provider: groq
    models: [llama-3.1-70b-versatile]
    
  # OpenRouter
  openrouter-1:
    api_key: "sk-or-..."
    provider: openrouter
    models: [anthropic/claude-3-haiku]
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Code                                │
│   response = client.chat("gpt-4o-mini", "Hello!")               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MAXLLM Client                             │
│  1. Validate inputs (Pydantic)                                   │
│  2. Estimate tokens needed (~50 tokens)                          │
│  3. Select best key using routing strategy                       │
│  4. Check capacity - defer if needed                             │
│  5. Execute via LiteLLM                                          │
│  6. Record latency & update rate limits                          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                          LiteLLM                                 │
│                    (handles provider API)                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │   OpenAI     │
                         └──────────────┘
```

### Key Features

#### 1. Deferred Execution (No 429 Errors)

When ALL keys are at capacity, MAXLLM doesn't fail - it waits:

```python
# If all keys exhausted, request is automatically deferred
# until capacity is available (no 429 errors!)
response = client.chat("gpt-4o-mini", "Hello")  # May wait, then succeeds
```

#### 2. Input Validation

All inputs are validated with Pydantic before execution:

```python
from maxllm import validate_chat_request

# Manual validation
request = validate_chat_request(
    model="gpt-4",
    messages="Hello!",
    temperature=0.7,
)

# Automatic validation (default)
response = client.chat("gpt-4", "Hello!", validate=True)  # ✅ Validated

# Skip validation for performance (not recommended)
response = client.chat("gpt-4", "Hello!", validate=False)
```

Validation checks:
- ✅ Model name is valid (no special characters)
- ✅ Messages are not empty or whitespace-only
- ✅ Temperature is 0-2
- ✅ Max tokens is positive
- ✅ Priority is high/medium/low
- ✅ Roles are valid (system/user/assistant/function/tool)

#### 3. Graceful Shutdown

Use context managers for automatic cleanup:

```python
# Sync
with MAXLLM.from_config("config.yaml") as client:
    response = client.chat(...)
# Waits for in-flight requests, then shuts down

# Async
async with MAXLLMAsync.from_config("config.yaml") as client:
    response = await client.chat(...)
```

Or manual shutdown:

```python
client = MAXLLM.from_config("config.yaml")
try:
    response = client.chat(...)
finally:
    client.shutdown(timeout=30)  # Wait max 30s for pending requests
```

## Production Deployment

### Redis Backend (Recommended for Production)

For distributed deployments or to persist rate limit state across restarts, use Redis:

```bash
pip install maxllm[redis]
```

```yaml
# config.yaml
redis_url: "redis://localhost:6379"
redis_prefix: "maxllm:"

keys:
  # ... your keys
```

**Redis provides:**
- 🔄 **Persistent state** - Rate limits survive restarts
- 🌐 **Distributed coordination** - Multiple instances share state
- 📊 **Centralized metrics** - Latency tracking across all instances
- 🔒 **Distributed locks** - Atomic operations across workers

**Using HybridRateLimiter (auto-fallback):**

```python
from maxllm.redis_backend import HybridRateLimiter
import asyncio

# Tries Redis, falls back to in-memory if unavailable
limiter = HybridRateLimiter(
    redis_url="redis://localhost:6379",
    fallback_to_memory=True,
)

await limiter.initialize()

if limiter.is_distributed:
    print("✅ Using Redis backend")
else:
    print("⚠️ Fallback to in-memory (Redis unavailable)")
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install maxllm[all]

COPY config.yaml .
COPY app.py .

CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  maxllm:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### Monitoring & Observability

```python
from maxllm import MAXLLM

client = MAXLLM.from_config("config.yaml")

# Check capacity across all keys
capacity = client.capacity()
print(f"Total capacity: {capacity['total_capacity']}")

# View latency stats per key
latency = client.latency()
for key_id, stats in latency.items():
    print(f"{key_id}: avg={stats['avg_ms']:.1f}ms, p99={stats['p99_ms']:.1f}ms")

# Debug routing decisions
scores = client.scores()
for key_id, score_data in scores.items():
    print(f"{key_id}: score={score_data['total_score']:.2f}, "
          f"util={score_data['utilization']:.2f}, "
          f"latency={score_data['latency_avg_ms']:.1f}ms")
```

### Health Checks

```python
# For Kubernetes/Docker health checks
def health_check():
    try:
        capacity = client.capacity()
        # Check if any key has capacity
        has_capacity = any(
            key['tokens_remaining'] > 1000 
            for key in capacity['keys'].values()
        )
        return has_capacity
    except Exception:
        return False
```

## API Reference

### ChatResponse

```python
response = client.chat("gpt-4o-mini", "Hello")

response.content       # The generated text
response.model         # Model used
response.usage         # {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
response.finish_reason # "stop", "length", etc.
response.latency       # Total request time in seconds
response.llm_latency   # LLM provider time only (NEW)
response.key_used      # Which API key was used
```

### MAXLLM Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `chat(model, messages, **kwargs)` | Sync chat completion | `ChatResponse` |
| `chat_stream(model, messages, **kwargs)` | Streaming completion | `Generator[str]` |
| `add_key(api_key, provider, models, ...)` | Add key at runtime | `None` |
| `status()` | Get scheduler status | `dict` |
| `capacity()` | Get current capacity | `dict` |
| `latency()` | Get latency stats per key (NEW) | `dict` |
| `scores()` | Get routing scores (NEW) | `dict` |
| `shutdown(timeout)` | Graceful shutdown (NEW) | `None` |

### MAXLLMAsync Methods

Same as `MAXLLM` but all methods are `async`:

```python
async with MAXLLMAsync.from_config("config.yaml") as client:
    response = await client.chat(...)
    
    async for chunk in client.chat_stream(...):
        print(chunk, end="")
```

### Configuration Classes

```python
from maxllm import MAXLLMConfig, KeyConfig

# Programmatic config
config = MAXLLMConfig(
    keys=[
        KeyConfig(
            api_key="sk-...",
            provider="openai",
            models=["gpt-4o-mini"],
            tpm_limit=90000,
            rpm_limit=500,
        )
    ],
    strategy="balanced",
    max_retries=3,
)

client = MAXLLM(config=config)
```

### Validation

```python
from maxllm import validate_chat_request, ChatRequest, ChatMessage

# Validate before sending
request = validate_chat_request(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=1024,
)

# Access validated data
print(request.model)  # "gpt-4"
print(request.messages[0].role)  # "user"
```

## Testing

```bash
# Install dev dependencies
pip install maxllm[dev]

# Run all tests
pytest

# Run specific test file
pytest tests/test_sdk.py

# With coverage report
pytest --cov=maxllm --cov-report=html

# Run only SDK tests (fast, no server deps needed)
pytest tests/test_sdk.py -v
```

### Test Structure

```
tests/
├── test_sdk.py           # SDK client tests (35 tests)
├── test_validation.py    # Input validation tests
├── test_rate_limiter.py  # Rate limiting tests
├── test_scheduler.py     # Scheduler tests
└── conftest.py           # Shared fixtures
```

## Examples

See the [`examples/`](examples/) directory for more:

- `basic_usage.py` - Simple SDK usage
- `async_usage.py` - Async client with concurrency
- `streaming.py` - Streaming responses
- `multi_provider.py` - Multiple providers and keys
- `monitoring.py` - Capacity tracking and monitoring

## Architecture

MAXLLM is built as a scheduling layer **on top of** [LiteLLM](https://github.com/BerriAI/litellm):

```
┌──────────────────────────────────────────┐
│            Your Application               │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│          MAXLLM (Scheduler)              │
│  • Rate limiting (token bucket)          │
│  • Smart routing (4 strategies)          │
│  • Queue management                      │
│  • Latency tracking                      │
│  • Input validation                      │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│        LiteLLM (Execution)               │
│  • Provider abstraction                  │
│  • API key management                    │
│  • Retry logic                           │
└────────────────┬─────────────────────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  ┌─────┐   ┌─────┐     ┌─────┐
  │ GPT │   │Groq │     │ ... │
  └─────┘   └─────┘     └─────┘
```

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## Roadmap

- [ ] Cost tracking and optimization
- [ ] Streaming with backpressure control
- [ ] Request priority queue
- [ ] Web dashboard UI
- [ ] Prometheus metrics export
- [ ] Custom retry strategies

## FAQ

**Q: Why use MAXLLM instead of calling LiteLLM directly?**

A: MAXLLM adds intelligent scheduling, rate limiting, and multi-key management. It prevents 429 errors and maximizes throughput across multiple keys/providers.

**Q: Does this work with OpenAI's official client?**

A: MAXLLM uses LiteLLM under the hood, which supports OpenAI and 100+ other providers. The API is similar but not identical to OpenAI's client.

**Q: What happens when all keys are rate limited?**

A: MAXLLM automatically defers the request and waits for capacity to become available. No 429 errors!

**Q: Can I use this in production?**

A: Yes! Version 0.2.0 is production-ready with input validation, Redis backend support, graceful shutdown, and comprehensive tests.

**Q: How does the balanced strategy work?**

A: It combines utilization (40%), latency (35%), errors (15%), and freshness (10%) into a weighted score. Lower score = better. It automatically adapts to your workload patterns.

**Q: Do I need Redis?**

A: No, Redis is optional. It's recommended for production/distributed deployments but MAXLLM works fine with in-memory state for single-instance deployments.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on top of [LiteLLM](https://github.com/BerriAI/litellm) for provider abstraction
- Token estimation using [tiktoken](https://github.com/openai/tiktoken)
- Input validation with [Pydantic](https://docs.pydantic.dev/)

---

<div align="center">

**MAXLLM v0.2.0** - Maximum LLM throughput with zero 429 errors.

[Documentation](https://github.com/yourusername/maxllm) • [Issues](https://github.com/yourusername/maxllm/issues) • [PyPI](https://pypi.org/project/maxllm/)

Made with ❤️ for the AI community

</div>
# MAXLLM
