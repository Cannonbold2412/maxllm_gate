"""FastAPI route definitions."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from llm_scheduler.api.schemas import (
    ChatRequest,
    ChatResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    StatusResponse,
    CapacityResponse,
    KeyStatus,
)
from llm_scheduler.api.dependencies import get_scheduler
from llm_scheduler.core.scheduler import Scheduler, SchedulerError
from llm_scheduler.observability.logging import get_logger


router = APIRouter()
logger = get_logger()


@router.post("/chat", response_model=ChatResponse, tags=["LLM"])
async def chat(
    request: ChatRequest,
    scheduler: Scheduler = Depends(get_scheduler),
) -> ChatResponse:
    """
    Send a chat completion request.
    
    The request is queued and scheduled based on:
    - Priority (high > medium > low)
    - Available API key capacity
    - Rate limits (TPM/RPM)
    
    If all keys are at capacity, the request is deferred until
    capacity becomes available.
    """
    try:
        messages = [m.model_dump() for m in request.messages]
        
        result = await scheduler.schedule(
            model=request.model,
            messages=messages,
            priority=request.priority,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        # Extract content from response
        content = ""
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if "message" in choice:
                content = choice["message"].get("content", "")
            elif "text" in choice:
                content = choice["text"]
        elif "content" in result:
            content = result["content"]
        
        return ChatResponse(
            id=result.get("id", ""),
            model=result.get("model", request.model),
            content=content,
            usage=result.get("usage"),
            finish_reason=result.get("choices", [{}])[0].get("finish_reason"),
        )
        
    except SchedulerError as e:
        logger.warning("Scheduler error", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Chat request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream", tags=["LLM"])
async def chat_stream(
    request: ChatRequest,
    scheduler: Scheduler = Depends(get_scheduler),
):
    """
    Send a streaming chat completion request.
    
    Returns a Server-Sent Events (SSE) stream of response chunks.
    """
    async def generate():
        try:
            messages = [m.model_dump() for m in request.messages]
            
            # For streaming, we need direct access to dispatcher
            # This is a simplified version
            result = await scheduler.schedule(
                model=request.model,
                messages=messages,
                priority=request.priority,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            content = result.get("content", "")
            # Simulate streaming for collected content
            for chunk in [content[i:i+10] for i in range(0, len(content), 10)]:
                yield f"data: {chunk}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.post("/batch", response_model=BatchResponse, tags=["LLM"])
async def batch(
    request: BatchRequest,
    scheduler: Scheduler = Depends(get_scheduler),
) -> BatchResponse:
    """
    Process multiple chat requests in parallel.
    
    Returns results for all requests, with errors inline.
    """
    results = []
    successful = 0
    failed = 0
    
    batch_requests = [
        {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "priority": req.priority,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        for req in request.requests
    ]
    
    raw_results = await scheduler.schedule_batch(batch_requests)
    
    for raw in raw_results:
        if isinstance(raw, Exception):
            failed += 1
            results.append({"error": str(raw)})
        else:
            successful += 1
            content = ""
            if "choices" in raw and raw["choices"]:
                content = raw["choices"][0].get("message", {}).get("content", "")
            
            results.append(ChatResponse(
                id=raw.get("id", ""),
                model=raw.get("model", ""),
                content=content,
                usage=raw.get("usage"),
                finish_reason=raw.get("choices", [{}])[0].get("finish_reason"),
            ))
    
    return BatchResponse(
        results=results,
        total=len(request.requests),
        successful=successful,
        failed=failed,
    )


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health(
    scheduler: Scheduler = Depends(get_scheduler),
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns overall system health status.
    """
    status_data = scheduler.get_status()
    queue_size = status_data["queue"]["queue_size"]
    
    # Count healthy keys
    keys_data = status_data["keys"].get("keys", {})
    healthy_keys = sum(1 for k in keys_data.values() if k.get("is_healthy", False))
    
    # Determine status
    if not status_data["running"]:
        status = "unhealthy"
    elif healthy_keys == 0:
        status = "unhealthy"
    elif queue_size > status_data["queue"]["max_size"] * 0.9:
        status = "degraded"
    else:
        status = "healthy"
    
    return HealthResponse(
        status=status,
        scheduler_running=status_data["running"],
        queue_size=queue_size,
        keys_available=healthy_keys,
    )


@router.get("/status", response_model=StatusResponse, tags=["System"])
async def status(
    scheduler: Scheduler = Depends(get_scheduler),
) -> StatusResponse:
    """
    Get detailed scheduler status.
    
    Returns queue statistics, key states, and configuration.
    """
    return scheduler.get_status()


@router.get("/capacity", response_model=CapacityResponse, tags=["System"])
async def capacity(
    scheduler: Scheduler = Depends(get_scheduler),
) -> CapacityResponse:
    """
    Get current capacity across all API keys.
    
    Useful for monitoring and understanding rate limit state.
    """
    status_data = scheduler.get_status()
    keys_data = status_data["keys"]
    
    total = keys_data.get("total_capacity", {})
    available = keys_data.get("available_capacity", {})
    
    key_statuses = [
        KeyStatus(**key_info)
        for key_info in keys_data.get("keys", {}).values()
    ]
    
    return CapacityResponse(
        total_tpm=total.get("tpm", 0),
        available_tpm=available.get("tpm", 0),
        total_rpm=total.get("rpm", 0),
        available_rpm=available.get("rpm", 0),
        keys=key_statuses,
    )


@router.get("/metrics", tags=["System"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus exposition format.
    """
    from starlette.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LLM Rate Limit Scheduler",
        "version": "0.1.0",
        "description": (
            "An intelligent scheduling and rate-limit-aware control layer "
            "on top of LiteLLM that maximizes throughput and prevents 429 errors."
        ),
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }
