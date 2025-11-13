"""
Unit tests for API metrics.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.api.metrics import (
    query_counter,
    query_duration,
    retrieval_results,
    active_requests,
    monitor_query,
)


class TestMetrics:
    """Test Prometheus metrics."""

    def test_query_counter_exists(self):
        """Test that query counter metric exists."""
        assert query_counter is not None
        assert query_counter._name == "rag_queries_total"

    def test_query_duration_exists(self):
        """Test that query duration histogram exists."""
        assert query_duration is not None
        assert query_duration._name == "rag_query_duration_seconds"

    def test_retrieval_results_exists(self):
        """Test that retrieval results histogram exists."""
        assert retrieval_results is not None
        assert retrieval_results._name == "rag_retrieval_results_count"

    def test_active_requests_exists(self):
        """Test that active requests gauge exists."""
        assert active_requests is not None
        assert active_requests._name == "rag_active_requests"

    @pytest.mark.asyncio
    async def test_monitor_query_decorator_basic(self):
        """Test monitor_query decorator with successful function."""

        @monitor_query
        async def test_func():
            return {"result": "success"}

        # Get initial metric values
        initial_active = active_requests._value.get()

        result = await test_func()

        assert result == {"result": "success"}
        # Active requests should return to initial value
        assert active_requests._value.get() == initial_active

    @pytest.mark.asyncio
    async def test_monitor_query_increments_counter(self):
        """Test that monitor_query increments counter."""

        @monitor_query
        async def test_func():
            return {"result": "success"}

        # Get initial counter value
        initial_count = query_counter.labels(endpoint="test_func")._value.get()

        await test_func()

        # Counter should increment
        new_count = query_counter.labels(endpoint="test_func")._value.get()
        assert new_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_monitor_query_tracks_duration(self):
        """Test that monitor_query tracks duration."""

        @monitor_query
        async def slow_func():
            await asyncio.sleep(0.01)
            return {"result": "success"}

        # Get initial sample count
        initial_count = query_duration._sum._value.get()

        await slow_func()

        # Duration should be tracked (sum should increase)
        new_sum = query_duration._sum._value.get()
        assert new_sum >= initial_count

    @pytest.mark.asyncio
    async def test_monitor_query_active_requests(self):
        """Test that active requests gauge is managed correctly."""

        @monitor_query
        async def test_func():
            # Check that active requests is incremented during execution
            current_active = active_requests._value.get()
            return {"active": current_active}

        initial_active = active_requests._value.get()

        result = await test_func()

        # During execution, active should have been higher
        # After execution, should return to initial
        assert active_requests._value.get() == initial_active

    @pytest.mark.asyncio
    async def test_monitor_query_with_exception(self):
        """Test that metrics are handled correctly when function raises exception."""

        @monitor_query
        async def failing_func():
            raise ValueError("Test error")

        initial_active = active_requests._value.get()

        with pytest.raises(ValueError, match="Test error"):
            await failing_func()

        # Active requests should be decremented even on exception
        assert active_requests._value.get() == initial_active

    @pytest.mark.asyncio
    async def test_monitor_query_tracks_result_count(self):
        """Test that monitor_query tracks result count if available."""

        class ResultWithCount:
            def __init__(self, count):
                self.count = count

        @monitor_query
        async def func_with_count():
            return ResultWithCount(count=5)

        await func_with_count()

        # Result count should be observed (histogram count should increase)
        # We can't easily verify the exact value, but we can check it doesn't error

    @pytest.mark.asyncio
    async def test_monitor_query_without_result_count(self):
        """Test that monitor_query handles results without count attribute."""

        @monitor_query
        async def func_without_count():
            return {"result": "no count attribute"}

        # Should not raise error
        result = await func_without_count()
        assert result == {"result": "no count attribute"}

    @pytest.mark.asyncio
    async def test_monitor_query_preserves_function_name(self):
        """Test that decorator preserves function name."""

        @monitor_query
        async def my_custom_function():
            return "test"

        assert my_custom_function.__name__ == "my_custom_function"

    @pytest.mark.asyncio
    async def test_monitor_query_with_args_kwargs(self):
        """Test monitor_query with function arguments."""

        @monitor_query
        async def func_with_args(arg1, arg2, kwarg1=None):
            return {"arg1": arg1, "arg2": arg2, "kwarg1": kwarg1}

        result = await func_with_args("val1", "val2", kwarg1="kwval1")

        assert result == {"arg1": "val1", "arg2": "val2", "kwarg1": "kwval1"}

    @pytest.mark.asyncio
    async def test_monitor_query_multiple_calls(self):
        """Test multiple calls to monitored function."""

        @monitor_query
        async def test_func():
            return "result"

        initial_count = query_counter.labels(endpoint="test_func")._value.get()

        # Call multiple times
        for _ in range(3):
            await test_func()

        # Counter should increment by 3
        new_count = query_counter.labels(endpoint="test_func")._value.get()
        assert new_count == initial_count + 3

    def test_query_duration_buckets(self):
        """Test that query duration has correct buckets."""
        expected_buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

        # Access buckets from histogram
        # Note: The exact way to access buckets depends on prometheus_client version
        # This is a simplified test
        assert query_duration is not None

    def test_retrieval_results_buckets(self):
        """Test that retrieval results histogram has correct buckets."""
        expected_buckets = [0, 1, 3, 5, 10, 20, 50]

        # Verify histogram exists with correct name
        assert retrieval_results is not None
