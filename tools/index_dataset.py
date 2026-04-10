# Build embeddings and an optional FAISS index for the datasets folder.
# pylint: disable=wrong-import-position,duplicate-code,no-value-for-parameter
"""Build embeddings and an optional FAISS index for the datasets folder.

Usage:
    python scripts/index_dataset.py --dataset datasets --out reports/index_cache --max-files 200

This writes `embeddings.npy`, `metadata.json` and `faiss.index` (if faiss available).
"""
# For scripts that must modify `sys.path` in order to import local packages,
# keep the necessary insertion but silence the import-position warning.
# pylint: disable=wrong-import-position
import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Ensure the project root is on sys.path so `modules` can be imported when the
# script is executed directly from the repository root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sentence_transformers import SentenceTransformer

try:
    import faiss
    _FAISS = True
except ImportError:
    faiss = None
    _FAISS = False

from modules.dataset_loader import iter_reference_documents
from modules.text_extractor import extract_text
from modules.text_preprocessor import clean_text, chunk_text


def build_index(dataset_dir: Path, out_dir: Path, model_name: str, max_files: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)

    all_chunks = []
    metas = []
    file_count = 0
    
    for fp, _ in iter_reference_documents(dataset_dir):
        if max_files is not None and file_count >= max_files:
            break
        try:
            raw = extract_text(fp)
        except (OSError, ValueError, AttributeError):
            continue
        text = clean_text(raw)
        if len(text) < 50:
            continue
            
        # Split into sliding window chunks for 90%+ resolution
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metas.append({
                "path": str(fp), 
                "chunk_id": i,
                "total_chunks": len(chunks),
                "size": fp.stat().st_size
            })
        
        file_count += 1

    if not all_chunks:
        raise RuntimeError("No reference texts found to index")

    print(f"Encoding {len(all_chunks)} chunks from {file_count} documents with {model_name}...")
    embeddings = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)

    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "metadata.json"
    np.save(emb_path, embeddings)
    meta_path.write_text(json.dumps(metas, indent=2), encoding="utf-8")

    if _FAISS:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        # normalize for cosine similarity
        # faiss.normalize_L2 has a C extension signature pylint may not infer
        faiss.normalize_L2(embeddings)  # pylint: disable=no-value-for-parameter
        index.add(embeddings)
        faiss.write_index(index, str(out_dir / "faiss.index"))
        print("Saved FAISS index")
    else:
        print("FAISS not available — saved embeddings and metadata only")

    return emb_path, meta_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets"))
    parser.add_argument("--out", type=Path, default=Path("reports/index_cache"))
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--max-files", type=int, default=200)
    args = parser.parse_args()

    build_index(args.dataset, args.out, args.model, args.max_files)
