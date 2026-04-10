from pathlib import Path
import json
import numpy as np

# faiss is optional; static analyzers may not have it available locally
# pylint: disable=import-error
try:
    import faiss
    _FAISS = True
except ImportError:
    faiss = None
    _FAISS = False


def _normed(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def load_index(cache_dir: Path):
    emb_file = cache_dir / "embeddings.npy"
    meta_file = cache_dir / "metadata.json"
    if not emb_file.exists() or not meta_file.exists():
        return None

    embeddings = np.load(emb_file)
    with open(meta_file, "r", encoding="utf-8") as fh:
        metas = json.load(fh)

    index = None
    if _FAISS and (cache_dir / "faiss.index").exists():
        index = faiss.read_index(str(cache_dir / "faiss.index"))

    return {"embeddings": embeddings, "metas": metas, "faiss": index}


def query_index(index_data, query_embedding, top_k=5):
    """Return top_k (score, meta) pairs.

    If FAISS available, use it; otherwise compute cosine similarity with numpy.
    """
    if index_data is None:
        return []

    embeddings = index_data["embeddings"]
    metas = index_data["metas"]
    faiss_index = index_data.get("faiss")

    if faiss_index is not None:
        # ensure normalized vectors for inner-product search
        q = query_embedding.astype(np.float32, copy=True)
        faiss.normalize_L2(q.reshape(1, -1))
        D, I = faiss_index.search(q.reshape(1, -1), top_k)
        return [(float(score), metas[int(idx)]) for score, idx in zip(D[0], I[0])]

    # fallback: cosine similarity brute-force using normalized vectors
    q = _normed(query_embedding)
    emb_norm = np.vstack([_normed(e) for e in embeddings])
    sims = emb_norm.dot(q)
    idxs = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), metas[int(i)]) for i in idxs]
