from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from services.data_ingestion_service import (
    save_upload_to_disk, extract_text_pages,
    iter_vectors_for_upsert, new_file_id
)
from services.embeddings_service import embed_texts
from services.vectordb_service import upsert_text_vectors, delete_by_file_id
from utils.logger import get_logger
import traceback

UPLOAD_DIR = "data"
router = APIRouter()
log = get_logger(__name__)

BATCH_SIZE = 128            # keep memory small
MAX_CHUNKS = 20000          # hard cap to avoid runaway memory usage
CHUNK_SIZE = 1200
OVERLAP = 200

class AddFileResponse(BaseModel):
    file_id: str
    chunks: int

@router.post("/add_file", response_model=AddFileResponse)
async def add_file(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}. Please upload a PDF.")

    fid = new_file_id()
    try:
        # 1) Save to disk
        try:
            file_path = save_upload_to_disk(UPLOAD_DIR, file, fid)
            log.info(f"Saved upload as {file_path}")
        except Exception as e:
            log.error("Save failed:\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Save failed: {e}")

        # 2) Extract pages
        try:
            pages = extract_text_pages(file_path)
            total_text = sum(len(p or "") for p in pages)
            if total_text == 0:
                raise HTTPException(status_code=400, detail="No extractable text found in PDF (image-only?).")
        except HTTPException:
            raise
        except Exception as e:
            log.error("PDF text extraction failed:\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"PDF text extraction failed: {e}")

        # 3) Stream chunks -> embed in batches -> upsert in batches
        try:
            batch_vecs: List[dict] = []
            chunks_count = 0

            for v in iter_vectors_for_upsert(fid, pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
                batch_vecs.append(v)
                chunks_count += 1

                if chunks_count > MAX_CHUNKS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF produced too many chunks (> {MAX_CHUNKS}). Try a smaller PDF or increase CHUNK_SIZE."
                    )

                if len(batch_vecs) >= BATCH_SIZE:
                    embs = embed_texts([x["text"] for x in batch_vecs])
                    upsert_text_vectors(batch_vecs, embs)
                    batch_vecs.clear()

            # flush final partial batch
            if batch_vecs:
                embs = embed_texts([x["text"] for x in batch_vecs])
                upsert_text_vectors(batch_vecs, embs)
                batch_vecs.clear()

            if chunks_count == 0:
                raise HTTPException(status_code=400, detail="Chunking produced 0 chunks. Try a different PDF.")

        except HTTPException:
            raise
        except Exception as e:
            log.error("Ingestion (chunk/embed/upsert) failed:\n" + traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

        return AddFileResponse(file_id=fid, chunks=chunks_count)

    except HTTPException:
        raise
    except Exception as e:
        log.error("Unexpected error:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@router.delete("/delete_file/{file_id}")
async def delete_file(file_id: str):
    try:
        delete_by_file_id(file_id)
        return {"status": "deleted", "file_id": file_id}
    except Exception as e:
        log.error("Delete failed:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

@router.put("/update_file/{file_id}", response_model=AddFileResponse)
async def update_file(file_id: str, file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}. Please upload a PDF.")
    try:
        delete_by_file_id(file_id)

        file_path = save_upload_to_disk(UPLOAD_DIR, file, file_id)
        pages = extract_text_pages(file_path)
        total_text = sum(len(p or "") for p in pages)
        if total_text == 0:
            raise HTTPException(status_code=400, detail="No extractable text found in new PDF.")

        batch_vecs: List[dict] = []
        chunks_count = 0
        for v in iter_vectors_for_upsert(file_id, pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
            batch_vecs.append(v)
            chunks_count += 1
            if chunks_count > MAX_CHUNKS:
                raise HTTPException(status_code=413, detail=f"Too many chunks (> {MAX_CHUNKS}).")
            if len(batch_vecs) >= BATCH_SIZE:
                embs = embed_texts([x["text"] for x in batch_vecs])
                upsert_text_vectors(batch_vecs, embs)
                batch_vecs.clear()

        if batch_vecs:
            embs = embed_texts([x["text"] for x in batch_vecs])
            upsert_text_vectors(batch_vecs, embs)

        return AddFileResponse(file_id=file_id, chunks=chunks_count)
    except HTTPException:
        raise
    except Exception as e:
        log.error("Update failed:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")
