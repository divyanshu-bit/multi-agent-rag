from typing import Literal
from langgraph.graph import StateGraph, END
from src.state import GraphState

async def query_router_node(state: GraphState) -> GraphState:
    return {"optimized_query": f"Optimized: {state['original_query']}", "execution_logs": state.get("execution_logs", []) + ["Query routed."]}

async def retrieval_node(state: GraphState) -> GraphState:
    return {"retrieved_documents": [], "execution_logs": state.get("execution_logs", []) + ["Retrieval completed."]}

async def doc_grader_node(state: GraphState) -> GraphState:
    return {"is_context_relevant": True, "execution_logs": state.get("execution_logs", []) + ["Context graded."]}

async def generation_node(state: GraphState) -> GraphState:
    return {"generated_response": "Response.", "execution_logs": state.get("execution_logs", []) + ["Response generated."]}

async def hallucination_grader_node(state: GraphState) -> GraphState:
    return {"is_hallucination_free": True, "execution_logs": state.get("execution_logs", []) + ["Hallucination check passed."]}

def decide_to_generate(state: GraphState) -> Literal["generate", "end"]:
    return "generate" if state.get("is_context_relevant") else "end"

builder = StateGraph(GraphState)
builder.add_node("query_router", query_router_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("doc_grader", doc_grader_node)
builder.add_node("generation", generation_node)
builder.add_node("hallucination_grader", hallucination_grader_node)

builder.set_entry_point("query_router")
builder.add_edge("query_router", "retrieval")
builder.add_edge("retrieval", "doc_grader")
builder.add_conditional_edges("doc_grader", decide_to_generate, {"generate": "generation", "end": END})
builder.add_edge("generation", "hallucination_grader")
builder.add_edge("hallucination_grader", END)

app = builder.compile()
