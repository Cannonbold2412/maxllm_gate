"""Main FastAPI application entry point."""

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_scheduler.config import settings
from llm_scheduler.api.routes import router
from llm_scheduler.core.scheduler import Scheduler
from llm_scheduler.observability.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger = get_logger()
    
    # Startup
    logger.info("Starting LLM Rate Limit Scheduler", version="0.1.0")
    
    # Initialize scheduler
    scheduler = Scheduler()
    await scheduler.start()
    app.state.scheduler = scheduler
    
    logger.info(
        "Scheduler initialized",
        strategy=settings.default_strategy,
        keys_loaded=len(settings.get_api_keys()),
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down scheduler")
    await scheduler.stop()
    logger.info("Scheduler stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging(level=settings.log_level)
    
    app = FastAPI(
        title="LLM Rate Limit Scheduler",
        description=(
            "An intelligent scheduling and rate-limit-aware control layer "
            "on top of LiteLLM that maximizes throughput and prevents 429 errors."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app


app = create_app()


def main():
    """Entry point for the CLI."""
    uvicorn.run(
        "llm_scheduler.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
