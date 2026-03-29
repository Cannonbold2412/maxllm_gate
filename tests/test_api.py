"""Tests for API routes."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from llm_scheduler.main import app
from llm_scheduler.core.scheduler import Scheduler


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIRoutes:
    """Test suite for API routes."""
    
    def test_root(self, client):
        """Root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert data["name"] == "LLM Rate Limit Scheduler"
        assert "version" in data
    
    def test_health(self, client):
        """Health endpoint works."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "scheduler_running" in data
        assert "queue_size" in data
    
    def test_status(self, client):
        """Status endpoint returns scheduler state."""
        response = client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "running" in data
        assert "queue" in data
        assert "keys" in data
    
    def test_capacity(self, client):
        """Capacity endpoint returns rate limit info."""
        response = client.get("/capacity")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_tpm" in data
        assert "available_tpm" in data
        assert "total_rpm" in data
        assert "available_rpm" in data
    
    def test_metrics(self, client):
        """Metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "llm_scheduler" in response.text
    
    def test_chat_validation(self, client):
        """Chat endpoint validates input."""
        # Missing model
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "hi"}]
        })
        assert response.status_code == 422
        
        # Missing messages
        response = client.post("/chat", json={
            "model": "gpt-4"
        })
        assert response.status_code == 422
        
        # Invalid priority
        response = client.post("/chat", json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "priority": "invalid"
        })
        assert response.status_code == 422
