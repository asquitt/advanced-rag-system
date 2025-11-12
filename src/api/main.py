"""
FastAPI service with monitoring.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from prometheus_client import make_asgi_app
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.api.metrics import monitor_query

app = FastAPI(
    title="Advanced RAG System",
    description="Hybrid retrieval with re-ranking",
    version="0.1.0"
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize retrieval pipeline
pipeline = RetrievalPipeline(
    collection_name="api_documents",
    use_reranking=True
)


class Document(BaseModel):
    """Document model."""
    text: str
    metadata: Optional[dict] = None


class IndexRequest(BaseModel):
    """Request to index documents."""
    documents: List[str]
    metadata: Optional[List[dict]] = None


class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    results: List[dict]
    count: int


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Advanced RAG System",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/index", status_code=201)
async def index_documents(request: IndexRequest):
    """Index documents."""
    try:
        pipeline.index(request.documents, request.metadata)
        return {
            "status": "success",
            "indexed": len(request.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
@monitor_query
async def query_documents(request: QueryRequest):
    """Query documents with monitoring."""
    try:
        results = pipeline.retrieve(request.query, top_k=request.top_k)
        return QueryResponse(
            query=request.query,
            results=results,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
