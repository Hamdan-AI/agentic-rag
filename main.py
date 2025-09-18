from fastapi import FastAPI
from api.routes_files import router as files_router
from api.routes_chat import router as chat_router

app = FastAPI(title="Agentic RAG (LangGraph + Pinecone + OpenAI)")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(files_router, prefix="")
app.include_router(chat_router, prefix="")
