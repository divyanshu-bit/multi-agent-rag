import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from src.orchestrator import run_rag_pipeline, stream_rag_pipeline

app = FastAPI(
    title="Multi-Agent RAG Orchestrator API",
    version="1.0.0",
    description="Production API with streaming and self-correction reflection loops."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    thread_id: str = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "multi-agent-rag-api"}

@app.post("/api/v1/query")
async def process_query(request: QueryRequest):
    """
    Executes the full RAG pipeline and returns the final state payload.
    """
    try:
        result = await run_rag_pipeline(
            user_query=request.query, 
            thread_id=request.thread_id
        )
        return {
            "thread_id": request.thread_id,
            "response": result.get("generated_response"),
            "fallback_mode": result.get("fallback_mode", False),
            "execution_logs": result.get("execution_logs", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/stream")
async def stream_query(request: QueryRequest):
    """
    Streams step-by-step state events via Server-Sent Events (SSE).
    """
    async def event_generator():
        async for event in stream_rag_pipeline(user_query=request.query, thread_id=request.thread_id):
            yield {
                "event": "state_update",
                "data": json.dumps(event, default=str)
            }

    return EventSourceResponse(event_generator())