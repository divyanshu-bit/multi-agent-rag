from typing import Literal
from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.agents.generation import generation_agent_node

# ---------------------------------------------------------------------------
# Fallback Circuit Breaker Node
# ---------------------------------------------------------------------------
async def fallback_node(state: GraphState) -> GraphState:
    """
    Terminates execution when max retries are reached to avoid infinite loops & excessive costs.
    """
    logs = state.get("execution_logs", [])
    logs.append("Circuit Breaker Activated: Maximum retry threshold reached.")
    
    return {
        "fallback_mode": True,
        "generated_response": "Unable to locate sufficient relevant context in the database to answer this request accurately.",
        "execution_logs": logs
    }

# ---------------------------------------------------------------------------
# Conditional Edge Router Functions
# ---------------------------------------------------------------------------
def decide_after_doc_grader(state: GraphState) -> Literal["generate", "rewrite", "fallback"]:
    """
    Evaluates context relevance. Routes to generation if context is valid, 
    otherwise triggers query re-writing or web search fallback if max retries exceeded.
    """
    is_relevant = state.get("is_context_relevant", False)
    retries = state.get("retry_count", 0)
    
    if is_relevant:
        return "generate"
    elif retries >= 3:
        return "fallback"
    else:
        return "rewrite"


def decide_after_hallucination_grader(state: GraphState) -> Literal["end", "generate", "fallback"]:
    """
    Evaluates response groundedness. Ends graph if hallucination-free, 
    otherwise triggers re-generation or fallback if loop limit hit.
    """
    is_grounded = state.get("is_hallucination_free", False)
    retries = state.get("retry_count", 0)
    
    if is_grounded:
        return "end"
    elif retries >= 3:
        return "fallback"
    else:
        return "generate"

# ---------------------------------------------------------------------------
# StateGraph Assembly
# ---------------------------------------------------------------------------
builder = StateGraph(GraphState)

# Add Generation & Fallback Nodes
builder.add_node("generation", generation_agent_node)
builder.add_node("fallback", fallback_node)

# Conditional Edges for Self-Correction Loops
builder.add_conditional_edges(
    "doc_grader",
    decide_after_doc_grader,
    {
        "generate": "generation",
        "rewrite": "query_rewriter",
        "fallback": "fallback"
    }
)

builder.add_conditional_edges(
    "hallucination_grader",
    decide_after_hallucination_grader,
    {
        "end": END,
        "generate": "generation",
        "fallback": "fallback"
    }
)

builder.add_edge("fallback", END)

# Compile Graph
app = builder.compile()