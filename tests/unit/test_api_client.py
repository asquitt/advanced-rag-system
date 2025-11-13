"""
Unit tests for RAGClient.
"""
import pytest
from unittest.mock import Mock, patch
from src.api.client import RAGClient


class TestRAGClient:
    """Test RAGClient class."""

    def test_initialization_default_url(self):
        """Test client initialization with default URL."""
        client = RAGClient()
        assert client.base_url == "http://localhost:8000"

    def test_initialization_custom_url(self):
        """Test client initialization with custom URL."""
        client = RAGClient(base_url="http://custom:9000")
        assert client.base_url == "http://custom:9000"

    @patch("src.api.client.requests.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = RAGClient()
        result = client.health()

        # Verify request
        mock_get.assert_called_once_with("http://localhost:8000/health")
        assert result == {"status": "healthy"}

    @patch("src.api.client.requests.get")
    def test_health_check_raises_on_error(self, mock_get):
        """Test health check raises on HTTP error."""
        # Mock response with error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        client = RAGClient()

        with pytest.raises(Exception, match="HTTP Error"):
            client.health()

    @patch("src.api.client.requests.post")
    def test_index_documents(self, mock_post, sample_texts):
        """Test indexing documents."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"indexed": 5, "status": "success"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = RAGClient()
        result = client.index(sample_texts)

        # Verify request
        mock_post.assert_called_once_with(
            "http://localhost:8000/index", json={"documents": sample_texts}
        )
        assert result == {"indexed": 5, "status": "success"}

    @patch("src.api.client.requests.post")
    def test_index_documents_with_metadata(self, mock_post, sample_texts):
        """Test indexing documents with metadata."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"indexed": 5, "status": "success"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        metadata = [{"source": f"doc_{i}"} for i in range(len(sample_texts))]

        client = RAGClient()
        result = client.index(sample_texts, metadata)

        # Verify request includes metadata
        mock_post.assert_called_once_with(
            "http://localhost:8000/index",
            json={"documents": sample_texts, "metadata": metadata},
        )

    @patch("src.api.client.requests.post")
    def test_index_documents_without_metadata(self, mock_post):
        """Test indexing documents without metadata."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"indexed": 2, "status": "success"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = RAGClient()
        docs = ["Doc 1", "Doc 2"]
        result = client.index(docs, metadata=None)

        # Verify request does not include metadata
        call_args = mock_post.call_args
        assert "metadata" not in call_args[1]["json"]

    @patch("src.api.client.requests.post")
    def test_index_raises_on_error(self, mock_post):
        """Test index raises on HTTP error."""
        # Mock response with error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_post.return_value = mock_response

        client = RAGClient()

        with pytest.raises(Exception, match="HTTP Error"):
            client.index(["doc1", "doc2"])

    @patch("src.api.client.requests.post")
    def test_query_basic(self, mock_post):
        """Test basic query."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": "test query",
            "count": 2,
            "results": [
                {"text": "Result 1", "score": 0.95},
                {"text": "Result 2", "score": 0.85},
            ],
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = RAGClient()
        result = client.query("test query", top_k=5)

        # Verify request
        mock_post.assert_called_once_with(
            "http://localhost:8000/query", json={"query": "test query", "top_k": 5}
        )

        assert result["query"] == "test query"
        assert result["count"] == 2
        assert len(result["results"]) == 2

    @patch("src.api.client.requests.post")
    def test_query_default_top_k(self, mock_post):
        """Test query with default top_k."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": "test",
            "count": 0,
            "results": [],
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = RAGClient()
        result = client.query("test")

        # Verify default top_k=5
        call_args = mock_post.call_args
        assert call_args[1]["json"]["top_k"] == 5

    @patch("src.api.client.requests.post")
    def test_query_custom_top_k(self, mock_post):
        """Test query with custom top_k."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": "test",
            "count": 0,
            "results": [],
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = RAGClient()
        result = client.query("test", top_k=10)

        # Verify custom top_k
        call_args = mock_post.call_args
        assert call_args[1]["json"]["top_k"] == 10

    @patch("src.api.client.requests.post")
    def test_query_raises_on_error(self, mock_post):
        """Test query raises on HTTP error."""
        # Mock response with error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_post.return_value = mock_response

        client = RAGClient()

        with pytest.raises(Exception, match="HTTP Error"):
            client.query("test query")

    @patch("src.api.client.requests.get")
    def test_health_with_custom_base_url(self, mock_get):
        """Test health check with custom base URL."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = RAGClient(base_url="http://custom-server:8080")
        result = client.health()

        # Verify correct URL is used
        mock_get.assert_called_once_with("http://custom-server:8080/health")

    @patch("src.api.client.requests.post")
    def test_query_empty_results(self, mock_post):
        """Test query with no results."""
        # Mock response with empty results
        mock_response = Mock()
        mock_response.json.return_value = {
            "query": "nonexistent query",
            "count": 0,
            "results": [],
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = RAGClient()
        result = client.query("nonexistent query")

        assert result["count"] == 0
        assert result["results"] == []

    @patch("src.api.client.requests.post")
    def test_index_empty_documents(self, mock_post):
        """Test indexing empty document list."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"indexed": 0, "status": "success"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = RAGClient()
        result = client.index([])

        # Should still make the request
        mock_post.assert_called_once_with(
            "http://localhost:8000/index", json={"documents": []}
        )
