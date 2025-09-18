from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from services.embeddings_service import embed_text
from services.vectordb_service import query_similar
from openai import OpenAI
from core.config import settings

class GraphState(TypedDict, total=False):
    question: str
    contexts: List[Dict]
    answer: str
    top_k: int
    pc_filter: Optional[Dict]

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def retrieve_node(state: GraphState) -> GraphState:
    emb = embed_text(state["question"])
    hits = query_similar(emb, top_k=state.get("top_k", 5), filter=state.get("pc_filter"))
    state["contexts"] = hits
    return state

def generate_node(state: GraphState) -> GraphState:
    ctx = "\n\n".join(c.get("text","") for c in state.get("contexts", []))
    msg = _client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role":"system","content":"Use ONLY the provided sources; if unsure, say you don’t know. Cite like [1],[2]."},
            {"role":"user","content": f"Question: {state['question']}\n\nSources:\n{ctx}"}
        ]
    )
    state["answer"] = msg.choices[0].message.content.strip()
    return state

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
