"""
main.py ? Module 3: Support Assistant FastAPI Application
Wraps the LangGraph pipeline in a REST API with POST /ask endpoint.

Run locally:
    cd support_assistant
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

With mock LLM (default, graded):
    MOCK_LLM=1 uvicorn main:app --host 0.0.0.0 --port 8000

With real LLM (optional):
    MOCK_LLM=0 GROQ_API_KEY=your_key uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import ask, ZeptoResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "A RAG-powered customer support assistant grounded in Zepto's policy documents. "
        "Uses LangGraph for orchestration, ChromaDB for retrieval, and all-MiniLM-L6-v2 for embeddings. "
        "MOCK_LLM=1 (default) runs fully offline with deterministic mock responses."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    query: str


class AskResponse(BaseModel):
    """Response model matching ZeptoResponse Pydantic schema."""
    answer: str
    sources: list[str]
    confidence: float


@app.get("/")
def root():
    """Health check endpoint."""
    mock_mode = os.environ.get("MOCK_LLM", "1") != "0"
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "mock_llm": mock_mode,
        "endpoints": {"ask": "POST /ask"},
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """
    Main question-answering endpoint.
    Accepts a customer query and returns a validated JSON response with
    answer, sources (document IDs), and confidence score.

    - If the query contains policy keywords (delivery, return, refund, etc.),
      it triggers retrieval from the ChromaDB policy corpus.
    - Otherwise, it returns a direct canned response (mock mode).
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response: ZeptoResponse = ask(request.query.strip())
        return AskResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
