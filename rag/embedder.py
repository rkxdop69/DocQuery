"""embedder.py — Embed chunk dicts and build a FAISS index."""
import numpy as np

def build_index(chunks: list[dict], existing_index=None, existing_model=None):
    """
    Accepts list of chunk dicts: [{"text": ..., "source": ..., "chunk_id": ...}]
    Supports incremental indexing — pass existing index+model to add to them.
    Returns (faiss_index, model).
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    texts = [c["text"] for c in chunks]

    # Reuse model if already loaded (avoids re-downloading on each upload)
    model = existing_model or SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    if existing_index is not None:
        existing_index.add(embeddings)
        return existing_index, model

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, model
