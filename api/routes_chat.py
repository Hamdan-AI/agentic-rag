from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.graph_service import build_graph

router = APIRouter()
_graph = build_graph()

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    file_id: str | None = None

@router.post("/chat")
def chat(req: ChatRequest):
    pc_filter = {"file_id": {"$eq": req.file_id}} if req.file_id else None
    out = _graph.invoke({"question": req.query, "top_k": req.top_k, "pc_filter": pc_filter})
    return {"answer": out.get("answer",""), "contexts": out.get("contexts", [])}
