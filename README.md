# Agentic RAG (LangGraph + Pinecone + OpenAI)

A minimal Retrieval-Augmented Generation (RAG) system over **PDFs only**.
- **Orchestration:** LangGraph (retrieve → generate)
- **Vector DB:** Pinecone (serverless)
- **Embeddings:** OpenAI `text-embedding-3-small` (1536-dim)
- **LLM:** OpenAI (e.g., `gpt-4o-mini`)
- **API:** FastAPI (Uvicorn)

---

## Run

```bash
uvicorn main:app --reload --port 8000
# then open http://127.0.0.1:8000/docs
```

Health check: `GET /health` → `{"status":"ok"}`

---

## Environment (.env)

Create a file named **`.env`** in the project root:

```dotenv
# OpenAI (LLM + Embeddings)
OPENAI_API_KEY=sk-...            # must have credits
OPENAI_MODEL=gpt-4o-mini
EMBEDDINGS_PROVIDER=openai
OPENAI_EMBED_MODEL=text-embedding-3-small   # 1536-dim

# Pinecone (Serverless)
PINECONE_API_KEY=pcsk-...
PINECONE_INDEX=agentic-rag-1536             # use a NEW name if you change embedding dims
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

> If you temporarily need **local embeddings** (for debugging):
> ```
> EMBEDDINGS_PROVIDER=local
> LOCAL_EMBED_MODEL=intfloat/e5-small-v2   # 384-dim
> PINECONE_INDEX=agentic-rag-384
> ```
> and install: `pip install sentence-transformers torch` (CPU wheel is fine).  
> **Note:** The official requirement prefers OpenAI embeddings.

---

## Example Requests (curl)

### 1) Upload PDF — `POST /add_file`
Uploads a PDF, extracts & chunks text, creates embeddings, upserts to Pinecone.

```bash
curl -F "file=@./sample.pdf" http://127.0.0.1:8000/add_file
```

**Response**
```json
{ "file_id": "xxxxxxxx", "chunks": 265, "message": "File indexed successfully" }
```

---

### 2) Chat — `POST /chat`
Retrieves top chunks from Pinecone and uses the OpenAI LLM to answer.

```bash
curl -X POST http://127.0.0.1:8000/chat   -H "Content-Type: application/json"   -d '{"query":"What are the key objectives?","top_k":5,"file_id":"<FILE_ID>"}'
```

**Response (truncated)**
```json
{ "answer": "…", "contexts": [ { "id":"<...::chunk::37>", "score": 0.88, "text": "…"} ] }
```
Tip: omit `"file_id"` to search across **all** uploaded PDFs.

---

### 3) Update existing file — `PUT /update_file/{file_id}`
Deletes old vectors for this `file_id`, re-ingests the new PDF, and upserts again.

```bash
curl -X PUT -F "file=@./new_version.pdf" http://127.0.0.1:8000/update_file/<FILE_ID>
```

**Response**
```json
{ "file_id":"<FILE_ID>", "chunks": 270, "message":"File re-indexed successfully" }
```

---

### 4) Delete by file ID — `DELETE /delete_file/{file_id}`
Removes all vectors for the given `file_id` from Pinecone.

```bash
curl -X DELETE http://127.0.0.1:8000/delete_file/<FILE_ID>
```

**Response**
```json
{ "status":"deleted", "file_id":"<FILE_ID>", "message":"Vectors deleted successfully" }
```

---

## Notes

- **PDF-only**: uploads are validated for PDF content types.
- **Chunking**: page-wise chunking with overlap; ingestion occurs in small batches to keep memory low.
- **LangGraph**: `/chat` runs a simple retrieve → generate graph.
- **Pinecone index dimension** must match your embedding model:
  - OpenAI `text-embedding-3-small` → **1536** (use an index like `agentic-rag-1536`)
  - If switching models/dimensions, use a **new** `PINECONE_INDEX` to avoid “dimension mismatch”.

---

## Directory Structure (for reviewers)

```
main.py
api/
  __init__.py
  routes_chat.py          # /chat
  routes_files.py         # /add_file, /update_file/{id}, /delete_file/{id}
core/
  __init__.py
  config.py               # loads .env (OpenAI, Pinecone)
services/
  __init__.py
  data_ingestion_service.py  # PDF extract + chunk
  embeddings_service.py      # OpenAI embeddings client (or local if toggled)
  vectordb_service.py        # Pinecone upsert/query/delete
  graph_service.py           # LangGraph pipeline
utils/
  __init__.py
  logger.py
requirements.txt
README.md
.env                        # local only (not committed)
```
