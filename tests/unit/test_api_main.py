"""
Integration tests for FastAPI main application.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Test FastAPI endpoints."""

    @patch("src.api.main.RetrievalPipeline")
    def test_root_endpoint(self, mock_pipeline_class):
        """Test root endpoint."""
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Advanced RAG System"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"

    @patch("src.api.main.RetrievalPipeline")
    def test_health_endpoint(self, mock_pipeline_class):
        """Test health check endpoint."""
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @patch("src.api.main.RetrievalPipeline")
    def test_index_endpoint_success(self, mock_pipeline_class):
        """Test successful document indexing."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        response = client.post("/index", json={"documents": documents})

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["indexed"] == 3

        # Verify pipeline.index was called
        mock_pipeline.index.assert_called_once_with(documents, None)

    @patch("src.api.main.RetrievalPipeline")
    def test_index_endpoint_with_metadata(self, mock_pipeline_class):
        """Test indexing with metadata."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        documents = ["Doc 1", "Doc 2"]
        metadata = [{"source": "test1"}, {"source": "test2"}]

        response = client.post(
            "/index", json={"documents": documents, "metadata": metadata}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["indexed"] == 2

        # Verify metadata was passed
        mock_pipeline.index.assert_called_once_with(documents, metadata)

    @patch("src.api.main.RetrievalPipeline")
    def test_index_endpoint_error(self, mock_pipeline_class):
        """Test index endpoint with error."""
        # Setup mock to raise exception
        mock_pipeline = Mock()
        mock_pipeline.index.side_effect = Exception("Indexing failed")
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        response = client.post("/index", json={"documents": ["Doc 1"]})

        assert response.status_code == 500
        assert "Indexing failed" in response.json()["detail"]

    @patch("src.api.main.RetrievalPipeline")
    def test_index_endpoint_invalid_request(self, mock_pipeline_class):
        """Test index endpoint with invalid request."""
        from src.api.main import app

        client = TestClient(app)

        # Missing required field
        response = client.post("/index", json={})

        assert response.status_code == 422  # Validation error

    @patch("src.api.main.RetrievalPipeline")
    def test_query_endpoint_success(self, mock_pipeline_class):
        """Test successful query."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline.retrieve.return_value = [
            {"text": "Result 1", "score": 0.95},
            {"text": "Result 2", "score": 0.85},
        ]
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        response = client.post("/query", json={"query": "test query", "top_k": 5})

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["count"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["text"] == "Result 1"

        # Verify pipeline.retrieve was called
        mock_pipeline.retrieve.assert_called_once_with("test query", top_k=5)

    @patch("src.api.main.RetrievalPipeline")
    def test_query_endpoint_default_top_k(self, mock_pipeline_class):
        """Test query with default top_k."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline.retrieve.return_value = []
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        response = client.post("/query", json={"query": "test"})

        assert response.status_code == 200

        # Verify default top_k=5 was used
        mock_pipeline.retrieve.assert_called_once_with("test", top_k=5)

    @patch("src.api.main.RetrievalPipeline")
    def test_query_endpoint_empty_results(self, mock_pipeline_class):
        """Test query with no results."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline.retrieve.return_value = []
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        response = client.post("/query", json={"query": "nonexistent", "top_k": 10})

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["results"] == []

    @patch("src.api.main.RetrievalPipeline")
    def test_query_endpoint_error(self, mock_pipeline_class):
        """Test query endpoint with error."""
        # Setup mock to raise exception
        mock_pipeline = Mock()
        mock_pipeline.retrieve.side_effect = Exception("Query failed")
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        response = client.post("/query", json={"query": "test"})

        assert response.status_code == 500
        assert "Query failed" in response.json()["detail"]

    @patch("src.api.main.RetrievalPipeline")
    def test_query_endpoint_invalid_request(self, mock_pipeline_class):
        """Test query endpoint with invalid request."""
        from src.api.main import app

        client = TestClient(app)

        # Missing required field
        response = client.post("/query", json={})

        assert response.status_code == 422  # Validation error

    @patch("src.api.main.RetrievalPipeline")
    def test_query_endpoint_invalid_top_k(self, mock_pipeline_class):
        """Test query with invalid top_k type."""
        from src.api.main import app

        client = TestClient(app)

        # Invalid top_k type
        response = client.post("/query", json={"query": "test", "top_k": "invalid"})

        assert response.status_code == 422  # Validation error

    @patch("src.api.main.RetrievalPipeline")
    def test_metrics_endpoint_exists(self, mock_pipeline_class):
        """Test that metrics endpoint exists."""
        from src.api.main import app

        client = TestClient(app)

        response = client.get("/metrics")

        # Metrics endpoint should return prometheus metrics
        assert response.status_code == 200
        # Should contain prometheus-style metrics
        assert b"# HELP" in response.content or b"# TYPE" in response.content

    @patch("src.api.main.RetrievalPipeline")
    def test_query_response_model(self, mock_pipeline_class):
        """Test query response follows correct model."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline.retrieve.return_value = [
            {"text": "Test doc", "score": 0.9, "extra_field": "ignored"}
        ]
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        response = client.post("/query", json={"query": "test"})

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "query" in data
        assert "results" in data
        assert "count" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["count"], int)

    @patch("src.api.main.RetrievalPipeline")
    def test_index_empty_documents(self, mock_pipeline_class):
        """Test indexing empty document list."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        response = client.post("/index", json={"documents": []})

        assert response.status_code == 201
        data = response.json()
        assert data["indexed"] == 0

    @patch("src.api.main.RetrievalPipeline")
    def test_concurrent_queries(self, mock_pipeline_class):
        """Test handling multiple concurrent queries."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline.retrieve.return_value = [{"text": "Result", "score": 0.9}]
        mock_pipeline_class.return_value = mock_pipeline

        from src.api.main import app

        client = TestClient(app)

        # Make multiple requests
        for i in range(5):
            response = client.post("/query", json={"query": f"test {i}"})
            assert response.status_code == 200

        # Verify all calls were made
        assert mock_pipeline.retrieve.call_count == 5
