from typing import List
from core.config import settings

if settings.EMBEDDINGS_PROVIDER == "local":
    # Local embeddings with SentenceTransformer
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(settings.LOCAL_EMBED_MODEL)

    def embed_texts(texts: List[str]) -> List[List[float]]:
        # Normalize whitespace a bit to avoid giant inputs
        texts = [(t or "").strip() for t in texts]
        embs = _model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [e.tolist() for e in embs]

    def embed_text(text: str) -> List[float]:
        return embed_texts([text])[0]

else:
    # OpenAI embeddings (requires billing)
    from openai import OpenAI
    _client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def embed_texts(texts: List[str]) -> List[List[float]]:
        resp = _client.embeddings.create(
            model=settings.OPENAI_EMBED_MODEL,
            input=texts
        )
        return [item.embedding for item in resp.data]

    def embed_text(text: str) -> List[float]:
        return embed_texts([text])[0]
