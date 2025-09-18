from typing import List, Dict, Optional, Any
from pinecone import Pinecone, ServerlessSpec
from core.config import settings
from utils.logger import get_logger

log = get_logger(__name__)

# Create a Pinecone client using your API key from .env
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index_name = settings.PINECONE_INDEX

def _list_index_names() -> List[str]:
    """Return a simple list of index names (handles object/dict styles)."""
    idxs = pc.list_indexes()
    try:
        return [i.name for i in idxs]  # SDK v7
    except Exception:
        try:
            return [i["name"] for i in idxs]  # fallback
        except Exception:
            return []

def _ensure_index():
    """Create the index if it doesn't exist, then return an Index handle."""
    if index_name not in _list_index_names():
        log.info(f"Creating Pinecone index '{index_name}' (dim={settings.embed_dim})")
        pc.create_index(
            name=index_name,
            dimension=settings.embed_dim,  # 1536 for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,   # e.g., "aws"
                region=settings.PINECONE_REGION  # e.g., "us-east-1"
            ),
        )
    return pc.Index(index_name)

# Global index handle used by our functions
_index = _ensure_index()

def upsert_text_vectors(vectors: List[Dict], embeddings: List[List[float]]) -> None:
    """
    vectors[i] -> {"id": str, "text": str, "metadata": {...}}
    embeddings[i] -> List[float] (same order as vectors)
    """
    payload = []
    for v, emb in zip(vectors, embeddings):
        md = dict(v.get("metadata", {}))
        md["text"] = v.get("text", "")[:1000]  # short preview for dashboard
        payload.append({
            "id": v["id"],
            "values": emb,
            "metadata": md
        })
    _index.upsert(vectors=payload)

def _safe_get_matches(res: Any) -> List[Any]:
    """Normalize .matches from SDK object or dict."""
    if hasattr(res, "matches"):
        return res.matches or []
    if isinstance(res, dict):
        return res.get("matches", []) or []
    return []

def query_similar(embedding: List[float], top_k: int = 5,
                  filter: Optional[Dict] = None) -> List[Dict]:
    res = _index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter or {}
    )
    matches = _safe_get_matches(res)
    out = []
    for m in matches:
        mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
        mscore = getattr(m, "score", None) or (m.get("score") if isinstance(m, dict) else None)
        mmd = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {}) or {}
        out.append({
            "id": mid,
            "score": mscore,
            "text": mmd.get("text", ""),
            "metadata": mmd
        })
    return out

def delete_by_file_id(file_id: str) -> None:
    _index.delete(filter={"file_id": {"$eq": file_id}})
