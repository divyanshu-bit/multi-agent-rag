from typing import List, Optional, TypedDict
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    relevance_score: float = 0.0

class GraphState(TypedDict):
    original_query: str
    optimized_query: str
    retrieved_documents: List[DocumentChunk]
    generated_response: Optional[str]
    is_context_relevant: bool
    is_hallucination_free: bool
    is_answer_satisfactory: bool
    retry_count: int
    fallback_mode: bool
    execution_logs: List[str]
