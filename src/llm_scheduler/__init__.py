"""LLM Rate Limit Scheduler - Intelligent scheduling layer on top of LiteLLM."""

__version__ = "0.1.0"

from llm_scheduler.config import settings
from llm_scheduler.core.scheduler import Scheduler

__all__ = ["settings", "Scheduler", "__version__"]
