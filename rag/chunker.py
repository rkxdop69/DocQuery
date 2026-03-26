"""chunker.py — Split text into overlapping chunks, tagged with source filename."""

def chunk_text(text: str, source: str, size: int = 400, overlap: int = 80) -> list[dict]:
    """
    Returns a list of chunk dicts:
      { "text": "...", "source": "filename.pdf", "chunk_id": 3 }
    """
    words = text.split()
    chunks, i, chunk_id = [], 0, 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": chunk_id,
            })
            chunk_id += 1
        i += size - overlap
    return chunks
