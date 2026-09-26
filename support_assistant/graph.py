"""
graph.py [*] Module 3: Support Assistant
LangGraph StateGraph with 3 nodes:
  1. classify_intent [*] keyword heuristic (mock) or LLM (optional MOCK_LLM=0)
  2. retrieve_and_answer [*] ChromaDB retrieval + canned template (mock) or LLM answer
  3. direct_answer [*] fixed canned string (mock) or direct LLM answer

MOCK_LLM environment variable:
  - Unset or "1" (default, graded baseline): fully deterministic mock mode, no LLM calls
  - "0": optional real LLM mode (requires GROQ_API_KEY or similar)
"""

import os
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from prompt_template import SYSTEM_PROMPT, build_user_prompt

# [*]
# Configuration
# [*]
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
MOCK_LLM_SNIPPET_LEN = 200

# Policy keywords for intent classification heuristic
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking", "track",
    "cancel", "gift card", "support hours", "damaged", "missing",
    "in stock", "fee", "subscription", "pass", "rider", "packed",
    "wallet", "replacement", "tier", "denomination"
]

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"


# [*]
# Pydantic Output Schema
# [*]
class ZeptoResponse(BaseModel):
    """Validated JSON response from the support assistant."""
    answer: str = Field(description="The assistant's answer to the customer query.")
    sources: list[str] = Field(
        default_factory=list,
        description="List of document/chunk IDs used to generate the answer."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )


# [*]
# LangGraph State
# [*]
class AssistantState(TypedDict):
    """State passed between nodes in the LangGraph."""
    query: str
    intent: str                    # "policy_question" or "general_question"
    retrieved_chunks: list[dict]   # Top-K retrieved chunks from ChromaDB
    response: ZeptoResponse | None


# [*]
# Shared Resources (loaded once)
# [*]
_embedding_model = None
_chroma_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


# [*]
# Optional Real-LLM Setup (MOCK_LLM=0 path)
# [*]
def get_llm_client():
    """Returns a Groq client if MOCK_LLM=0 and GROQ_API_KEY is set."""
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        return Groq(api_key=api_key)
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call real LLM (Groq). Only used when MOCK_LLM=0."""
    client = get_llm_client()
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# [*]
# NODE 1: classify_intent
# [*]
def classify_intent(state: AssistantState) -> AssistantState:
    """
    Classify query as 'policy_question' or 'general_question'.
    Mock mode (graded): keyword heuristic, no LLM call.
    Real mode (MOCK_LLM=0): LLM classification.
    """
    query = state["query"]

    if MOCK_LLM:
        # Graded baseline: keyword heuristic
        query_lower = query.lower()
        intent = "policy_question" if any(kw in query_lower for kw in POLICY_KEYWORDS) \
            else "general_question"
        print(f"  [classify_intent] Mock mode [*] intent: {intent}")
    else:
        # Optional real-LLM path
        prompt = (
            f"Classify this customer query as exactly one of: "
            f"'policy_question' or 'general_question'.\n"
            f"Query: {query}\n"
            f"Respond with ONLY the classification label."
        )
        result = call_llm(
            "You are an intent classifier for a customer support system.",
            prompt
        )
        intent = "policy_question" if "policy" in result.lower() else "general_question"
        print(f"  [classify_intent] Real LLM [*] intent: {intent}")

    return {**state, "intent": intent}


# [*]
# NODE 2: retrieve_and_answer
# [*]
def retrieve_and_answer(state: AssistantState) -> AssistantState:
    """
    For policy_question:
    - Always retrieves top-K chunks from ChromaDB (real retrieval in both modes).
    - Mock mode: returns canned template from top chunk snippet.
    - Real mode (MOCK_LLM=0): calls LLM with retrieved context.
    """
    query = state["query"]
    model = get_embedding_model()
    collection = get_chroma_collection()

    # Embed query and retrieve top-K chunks (ALWAYS real, no LLM needed)
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc_text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "id": meta["doc_id"],
            "policy_name": meta["policy_name"],
            "text": doc_text,
            "distance": dist,
        })

    print(f"  [retrieve_and_answer] Retrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"    - {c['id']} ({c['policy_name']}) dist={c['distance']:.4f}")

    # Answer generation
    if MOCK_LLM:
        # Graded baseline: canned template from top chunk
        top_chunk_snippet = chunks[0]["text"][:MOCK_LLM_SNIPPET_LEN] if chunks else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        sources = [c["id"] for c in chunks]
        confidence = 1.0
        print("  [retrieve_and_answer] Mock mode [*] returning canned template answer.")
    else:
        # Optional real-LLM path with retry on validation failure
        context_texts = [c["text"] for c in chunks]
        user_prompt = build_user_prompt(query, context_texts)
        import json

        raw_answer = None
        for attempt in range(3):
            try:
                raw = call_llm(SYSTEM_PROMPT, user_prompt)
                parsed = json.loads(raw)
                response_obj = ZeptoResponse(**parsed)
                answer = response_obj.answer
                sources = response_obj.sources
                confidence = response_obj.confidence
                raw_answer = raw
                break
            except Exception as e:
                print(f"  [retrieve_and_answer] LLM attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    user_prompt += (
                        f"\n\nPrevious response was invalid JSON or failed schema validation. "
                        f"Error: {e}. Please respond ONLY with valid JSON matching the schema."
                    )
                else:
                    answer = "I encountered an error generating a response. Please contact Zepto support."
                    sources = [c["id"] for c in chunks]
                    confidence = 0.0

    response = ZeptoResponse(answer=answer, sources=sources, confidence=confidence)
    return {**state, "retrieved_chunks": chunks, "response": response}


# [*]
# NODE 3: direct_answer
# [*]
def direct_answer(state: AssistantState) -> AssistantState:
    """
    For general_question:
    - Mock mode: fixed canned string.
    - Real mode (MOCK_LLM=0): direct LLM answer with no retrieval.
    """
    if MOCK_LLM:
        answer = (
            "I can only answer questions about Zepto policies right now. "
            "For other questions, please visit https://www.zeptonow.com or "
            "contact Zepto in-app support (available 24/7)."
        )
        sources = []
        confidence = 1.0
        print("  [direct_answer] Mock mode [*] returning fixed canned string.")
    else:
        prompt = (
            f"Answer this customer query about Zepto:\n{state['query']}\n"
            f"Be concise and helpful."
        )
        try:
            answer = call_llm(
                "You are Zara, Zepto's helpful customer support assistant.",
                prompt
            )
            sources = []
            confidence = 0.7
        except Exception as e:
            answer = f"I'm unable to answer that question right now. Error: {e}"
            sources = []
            confidence = 0.0

    response = ZeptoResponse(answer=answer, sources=sources, confidence=confidence)
    return {**state, "response": response}


# [*]
# Conditional Edge Router
# [*]
def route_by_intent(state: AssistantState) -> str:
    """Routes to the correct node based on classified intent."""
    intent = state.get("intent", "general_question")
    return "retrieve_and_answer" if intent == "policy_question" else "direct_answer"


# [*]
# Build the LangGraph
# [*]
def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    builder = StateGraph(AssistantState)

    # Add nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)

    # Entry point
    builder.set_entry_point("classify_intent")

    # Conditional edge from classify_intent
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        }
    )

    # Both answer nodes go to END
    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)

    graph = builder.compile()
    return graph


# [*]
# Public API
# [*]
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(query: str) -> ZeptoResponse:
    """Main entry point: run the graph and return a validated ZeptoResponse."""
    graph = get_graph()
    initial_state: AssistantState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "response": None,
    }
    final_state = graph.invoke(initial_state)
    return final_state["response"]


# [*]
# Demo
# [*]
if __name__ == "__main__":
    print(f"MOCK_LLM = {MOCK_LLM}")
    print("\n=== Example 1: Policy question (should trigger retrieval) ===")
    r1 = ask("What is the delivery fee for orders below INR 149?")
    print(f"Answer: {r1.answer}")
    print(f"Sources: {r1.sources}")
    print(f"Confidence: {r1.confidence}")

    print("\n=== Example 2: General question (should NOT trigger retrieval) ===")
    r2 = ask("What is the weather like today?")
    print(f"Answer: {r2.answer}")
    print(f"Sources: {r2.sources}")
    print(f"Confidence: {r2.confidence}")
