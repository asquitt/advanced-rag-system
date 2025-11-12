"""
FastAPI service for RAG system.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.retrieval.retrieval_pipeline import RetrievalPipeline

app = FastAPI(
    title="Advanced RAG System",
    description="Hybrid retrieval with re-ranking",
    version="0.1.0"
)

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
    """
    Index documents.
    
    Args:
        request: IndexRequest with documents and metadata
        
    Returns:
        Status message
    """
    try:
        pipeline.index(request.documents, request.metadata)
        return {
            "status": "success",
            "indexed": len(request.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents.
    
    Args:
        request: QueryRequest with query and top_k
        
    Returns:
        QueryResponse with results
    """
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
