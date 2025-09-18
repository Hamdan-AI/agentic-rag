import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # LLM (still OpenAI for chat)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embeddings: provider switch
    EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    LOCAL_EMBED_MODEL: str = os.getenv("LOCAL_EMBED_MODEL", "intfloat/e5-small-v2")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "agentic-rag")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

    @property
    def embed_dim(self) -> int:
        if self.EMBEDDINGS_PROVIDER == "local":
            # common local models
            name = self.LOCAL_EMBED_MODEL.lower()
            if "e5-small" in name: return 384
            if "bge-small" in name: return 384
            if "e5-base" in name or "bge-base" in name: return 768
            if "e5-large" in name or "bge-large" in name: return 1024
            # fallback try: default to 384 for small models
            return 384
        # OpenAI models
        m = self.OPENAI_EMBED_MODEL.lower()
        if "3-large" in m: return 3072
        return 1536

settings = Settings()
