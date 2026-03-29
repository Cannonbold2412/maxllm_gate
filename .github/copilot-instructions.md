# maxllm_gate Copilot Instructions

## Project Overview

maxllm_gate is a production-ready LLM client that sits on top of LiteLLM, providing intelligent rate limiting, smart routing, and distributed state support. It manages multiple API keys across providers (OpenAI, Groq, OpenRouter, etc.) to maximize throughput and prevent 429 errors.

### Architecture

The project has two main components:

1. **SDK Client (`src/maxllm_gate_gate/`)** - Simple Python client library for end users
2. **Scheduler Server (`src/llm_scheduler/`)** - Optional FastAPI gateway with advanced scheduling

**Request Flow:**
```
User → maxllm_gate Client → Scheduler → Rate Limiter → LiteLLM → Provider API
                      ↓
                 Queue Manager (if capacity exhausted)
```

**Core Components:**
- `llm_scheduler/core/scheduler.py` - Main scheduling engine that routes requests
- `llm_scheduler/rate_limiting/token_bucket.py` - Token bucket algorithm for rate limiting
- `llm_scheduler/strategies/` - Routing strategies (least_utilized, round_robin, token_aware, balanced)
- `maxllm/client.py` - User-facing SDK (sync/async)
- `maxllm/scheduler.py` - SDK's scheduler (simplified version for client use)

### Key Concepts

**Dual Package Structure:**
- `maxllm` - The SDK package that users import (`from maxllm_gate import maxllm_gate`)
- `llm_scheduler` - Server/API package for FastAPI gateway mode
- Both packages are in `src/` and installed together via `pyproject.toml`

**Rate Limiting Philosophy:**
- Never blindly hit 429 errors
- Estimate tokens BEFORE making requests (using tiktoken)
- Check ALL available keys before deciding to wait
- Use token bucket algorithm for TPM/RPM tracking
- Defer execution when capacity exhausted (queuing instead of failing)

**Routing Strategies:**
- `least_utilized` - Routes to key with most available capacity
- `round_robin` - Cycles through keys evenly
- `token_aware` - Prioritizes keys that can handle the request size
- `balanced` (NEW) - Weighted scoring: utilization (40%), latency (35%), errors (15%), freshness (10%)

## Build, Test, and Lint Commands

### Installation
```bash
# Development setup
pip install -e ".[dev,all]"

# Individual features
pip install -e ".[server]"  # FastAPI server mode
pip install -e ".[yaml]"    # YAML config support
pip install -e ".[redis]"   # Redis backend
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_sdk.py

# Run single test
pytest tests/test_sdk.py::test_chat_basic

# Run async tests only
pytest -k "asyncio"
```

### Linting
```bash
# Run ruff linter
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Type checking
mypy src/
```

### Running the Server
```bash
# Start FastAPI server (requires [server] extras)
maxllm_gate-server

# Or with uvicorn directly
uvicorn llm_scheduler.main:app --host 0.0.0.0 --port 8000

# With Docker
docker-compose up

# Check health
curl http://localhost:8000/health
```

### Running Examples
```bash
python examples/basic_usage.py
python examples/concurrent_requests.py
python examples/multi_key_config.py
```

## Key Conventions

### Config Management

**Two config systems coexist:**
1. SDK Config (`maxllm/config.py`) - Simple YAML/dict for client library
2. Server Config (`llm_scheduler/config.py`) - Pydantic Settings for FastAPI app

Both use similar structure but serve different purposes. Don't confuse them when making changes.

**Config Loading Priority:**
```python
# SDK client
maxllm_gate.from_config("config.yaml")  # YAML file
maxllm_gate.from_env()                  # Environment variables
maxllm_gate(keys=[...])                 # Direct dict

# Server uses Pydantic Settings
settings.get_api_keys()  # Reads from env vars or config
```

### Async/Sync Duality

The SDK provides both sync (`maxllm_gate`) and async (`maxllm_gate_async`) clients. Key patterns:

- Async is preferred for production/high-throughput scenarios
- Sync wrapper uses `asyncio.run()` internally
- Both share the same core logic in `scheduler.py` and `rate_limiter.py`
- Tests use `@pytest.mark.asyncio` for async code

**Implementation pattern:**
```python
# Internal methods are async
async def _execute_request(...):
    ...

# Public API provides both
def chat(self, ...):  # Sync wrapper
    return asyncio.run(self._execute_request(...))

async def chat(self, ...):  # Async version
    return await self._execute_request(...)
```

### Token Estimation

Token counting happens BEFORE requests to avoid hitting rate limits:

```python
# src/llm_scheduler/core/token_estimator.py
estimated_tokens = token_estimator.estimate(messages, max_tokens)
```

- Uses tiktoken for accurate counts
- Adds buffer (default 10%) for safety margin
- Cached encoders per model to avoid repeated initialization
- Estimation errors are conservative (overestimate to be safe)

### Strategy Selection

Strategies are selected by name in config and resolved via registry:

```python
# src/llm_scheduler/strategies/__init__.py
strategy = StrategyRegistry.get(strategy_name)
selected_key = strategy.select(candidates, estimated_tokens)
```

When adding new strategies:
1. Create class in `strategies/` that extends `SchedulingStrategy`
2. Register in `StrategyRegistry` 
3. Add to config validation in `config.py`
4. Add tests in `test_strategies.py`

### Error Handling

**Retry Logic:**
- Transient failures (network, timeout) → automatic retry with exponential backoff
- Rate limit hits (429) → should never happen (that's the point!)
- Auth failures (401) → immediate fail, no retry
- Model not found (404) → immediate fail, no retry

**Key Health Tracking:**
- Each key tracks error rate and latency
- Strategies can use health metrics for routing decisions
- Unhealthy keys are automatically deprioritized
- See `llm_scheduler/rate_limiting/tracker.py`

### Testing Patterns

**Fixtures in conftest.py:**
- `mock_config` - Test config with fake keys
- `key_manager` - Pre-configured KeyManager
- `scheduler` - Running scheduler instance
- `sample_messages` - Standard test messages

**Mocking LiteLLM:**
```python
@patch("litellm.acompletion")
async def test_something(mock_completion):
    mock_completion.return_value = AsyncMock(...)
    # Test code
```

**Testing Rate Limits:**
Use `TokenBucket` directly to test token bucket logic without full scheduler overhead.

### Redis Backend (Optional)

For distributed deployments, rate limit state can be stored in Redis:

```python
# src/maxllm_gate_gate/redis_backend.py
limiter = HybridRateLimiter(
    redis_url="redis://localhost:6379",
    fallback_to_memory=True,  # Graceful degradation
)
```

- Keys stored as `maxllm:ratelimit:{key_id}:tokens`
- Uses Redis EVAL for atomic token consumption
- Falls back to in-memory if Redis unavailable
- Not required for single-instance deployments

### Observability

**Metrics Available:**
- `client.capacity()` - Token/request capacity per key
- `client.latency()` - Latency stats (avg, p50, p99)
- `client.scores()` - Routing decision scores per key

**Prometheus Integration (server mode):**
- Request counts by model/key
- Latency histograms
- Queue depth
- Rate limit hit rate
- Available at `/metrics` endpoint

### Common Pitfalls

1. **Don't confuse the two config systems** - SDK uses `maxllm_gate_config`, server uses Pydantic Settings
2. **Token estimation is approximate** - Always add buffer, never assume exact count
3. **Strategies return None if no capacity** - Handle this case (queue or fail)
4. **Context managers are important** - Use `with maxllm_gate.from_config(...)` for graceful shutdown
5. **Test isolation** - Each test should use fresh scheduler instance (see fixtures)
6. **Provider-specific quirks** - Some providers need special handling in LiteLLM (check docs)

### File Organization

```
src/
  maxllm/           # SDK package (public API)
    client.py       # User-facing maxllm_gate/maxllm_gate_async classes
    scheduler.py    # Client-side scheduler
    config.py       # SDK config models
    rate_limiter.py # Rate limiting for SDK
    validation.py   # Pydantic request validation
    
  llm_scheduler/    # Server package (FastAPI)
    main.py         # FastAPI app entry point
    config.py       # Server settings (Pydantic)
    api/            # FastAPI routes
    core/           # Core scheduling logic
      scheduler.py  # Main scheduler engine
      dispatcher.py # Request dispatcher
      queue_manager.py  # Request queuing
      token_estimator.py  # Token counting
    rate_limiting/  # Rate limit tracking
      token_bucket.py  # Token bucket algorithm
      key_manager.py   # API key management
      tracker.py       # Rate limit state
    strategies/     # Routing strategies
    observability/  # Logging and metrics
```

## Environment Variables

```bash
# For SDK usage
maxllm_gate_KEYS='{"groq-1": {...}, "openai-1": {...}}'

# For server mode (see llm_scheduler/config.py)
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
DEFAULT_STRATEGY=balanced
MAX_QUEUE_SIZE=10000

# Redis (optional)
REDIS_URL=redis://localhost:6379
REDIS_PREFIX=maxllm:

# Provider API keys can also be individual env vars
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
```

## Making Changes

### Adding a New Provider

1. LiteLLM already handles most providers - just add to config
2. Update `config.example.yaml` with example
3. Add provider-specific rate limits (check their docs)
4. Test with `examples/basic_usage.py`

### Adding a New Strategy

1. Create `src/llm_scheduler/strategies/my_strategy.py`
2. Extend `SchedulingStrategy` base class
3. Implement `select()` method
4. Register in `StrategyRegistry` (`strategies/__init__.py`)
5. Add tests in `tests/test_strategies.py`
6. Update README.md strategy table

### Modifying Rate Limiting

Core logic is in `token_bucket.py`. The token bucket algorithm:
- Refills at constant rate (TPM/RPM converted to tokens per second)
- Consumes tokens on each request
- Blocks if insufficient capacity

Be careful changing this - it's mathematically proven and well-tested.

### Changing Token Estimation

Token estimation is in `core/token_estimator.py`. Uses tiktoken under the hood. If changing:
- Keep conservative (overestimate is better than underestimate)
- Cache encoders (they're expensive to create)
- Test with various message lengths
- Consider token_buffer multiplier in config
