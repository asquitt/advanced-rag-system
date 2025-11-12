"""
Prometheus metrics for monitoring.
"""
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Define metrics
query_counter = Counter("rag_queries_total", "Total number of queries", ["endpoint"])

query_duration = Histogram(
    "rag_query_duration_seconds",
    "Query processing duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

retrieval_results = Histogram(
    "rag_retrieval_results_count",
    "Number of results retrieved",
    buckets=[0, 1, 3, 5, 10, 20, 50],
)

active_requests = Gauge("rag_active_requests", "Number of active requests")


def monitor_query(func):
    """Decorator to monitor query functions."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        active_requests.inc()
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            query_duration.observe(time.time() - start_time)
            query_counter.labels(endpoint=func.__name__).inc()

            # Track result count if available
            if hasattr(result, "count"):
                retrieval_results.observe(result.count)

            return result
        finally:
            active_requests.dec()

    return wrapper
