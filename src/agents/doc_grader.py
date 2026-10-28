from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState

class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

async def doc_grader_node(state: GraphState) -> GraphState:
    """
    Evaluates retrieved documents for relevance to the user's query.
    """
    question = state["optimized_query"]
    documents = state.get("retrieved_documents", [])
    logs = state.get("execution_logs", [])

    if not documents:
        logs.append("Doc Grader: No documents found to grade.")
        return {"is_context_relevant": False, "execution_logs": logs}

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(GradeDocuments)

    system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])

    grader_chain = grade_prompt | structured_llm

    relevant_docs = []
    is_any_relevant = False

    for doc in documents:
        content = doc.get("page_content", "") if isinstance(doc, dict) else str(doc)
        res = await grader_chain.ainvoke({"document": content, "question": question})
        if res.binary_score.lower() == "yes":
            is_any_relevant = True
            relevant_docs.append(doc)

    logs.append(f"Doc Grader: Evaluated context relevance -> {is_any_relevant}")

    return {
        "is_context_relevant": is_any_relevant,
        "retrieved_documents": relevant_docs if is_any_relevant else documents,
        "execution_logs": logs
    }