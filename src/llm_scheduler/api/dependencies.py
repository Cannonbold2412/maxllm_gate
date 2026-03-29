"""FastAPI dependencies."""

from fastapi import Request

from llm_scheduler.core.scheduler import Scheduler


def get_scheduler(request: Request) -> Scheduler:
    """Get scheduler instance from app state."""
    return request.app.state.scheduler
