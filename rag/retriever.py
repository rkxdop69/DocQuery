"""retriever.py — Find top-k most relevant chunk dicts for a query."""
import numpy as np

def retrieve(query: str, index, chunks: list[dict], model, top_k: int = 4) -> list[dict]:
    """
    Returns top-k chunk dicts sorted by relevance:
      [{"text": ..., "source": ..., "chunk_id": ..., "score": ...}]
    """
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_emb, min(top_k, len(chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            chunk = dict(chunks[idx])          # copy so we don't mutate original
            chunk["score"] = float(dist)       # L2 distance — lower = more relevant
            results.append(chunk)

    return results
