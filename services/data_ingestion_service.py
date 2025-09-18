import os
import uuid
import re
from typing import List, Dict, Iterable
from pypdf import PdfReader

# ----------- cleaning & utils -----------

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", "")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> Iterable[str]:
    """
    Memory-safe generator: yields chunks without building huge lists.
    Guarantees forward progress and guards overlap.
    """
    text = clean_text(text)
    n = len(text)
    if n == 0:
        return
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    step = max(1, chunk_size - overlap)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= n:
            break
        start += step  # always forward

# ----------- PDF extraction -----------

def extract_text_pages(file_path: str) -> List[str]:
    """
    Return a list of cleaned text strings, one per page.
    This avoids concatenating the entire PDF into a single giant string.
    """
    reader = PdfReader(file_path)
    pages: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(clean_text(t))
    return pages

# ----------- vector prep (as generator) -----------

def iter_vectors_for_upsert(file_id: str, pages: List[str],
                            chunk_size: int = 1200, overlap: int = 200) -> Iterable[Dict]:
    """
    Yield vectors one-by-one (id, text, metadata) to keep memory low.
    """
    chunk_idx = 0
    for page_idx, page_text in enumerate(pages):
        if not page_text:
            continue
        for ch in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            yield {
                "id": f"{file_id}::chunk::{chunk_idx}",
                "text": ch,
                "metadata": {"file_id": file_id, "chunk_index": chunk_idx, "page": page_idx}
            }
            chunk_idx += 1

def save_upload_to_disk(upload_dir: str, upfile, file_id: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(upfile.filename or "")[1].lower() or ".pdf"
    file_path = os.path.join(upload_dir, f"{file_id}{ext}")
    with open(file_path, "wb") as f:
        f.write(upfile.file.read())
    return file_path

def new_file_id() -> str:
    return uuid.uuid4().hex
