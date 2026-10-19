import asyncio
from src.graph import app

async def test_circuit_breaker():
    test_state = {
        "original_query": "Explain proprietary company policy X.",
        "optimized_query": "policy X",
        "retrieved_documents": [],
        "generated_response": None,
        "is_context_relevant": False,
        "is_hallucination_free": False,
        "is_answer_satisfactory": False,
        "retry_count": 3,
        "fallback_mode": False,
        "execution_logs": []
    }
    
    result = await app.ainvoke(test_state)
    print("--- CIRCUIT BREAKER TEST RESULT ---")
    print("Fallback Activated:", result.get("fallback_mode"))
    print("Final Output:", result.get("generated_response"))
    print("Execution Logs:", result.get("execution_logs"))

if __name__ == "__main__":
    asyncio.run(test_circuit_breaker())