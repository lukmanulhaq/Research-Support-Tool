import json
import os
import numpy as np

_embedder = None
_index = None
_corpus = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def build_index(corpus_path: str = None) -> dict:
    import faiss

    if corpus_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus_path = os.path.join(base, "data", "corpus.json")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    texts = [paper["chunk_text"] for paper in corpus]

    embedder = _get_embedder()
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return {
        "index": index,
        "corpus": corpus,
        "embeddings": embeddings,
    }


def search(query: str, store: dict, k: int = 5) -> list[dict]:
    embedder = _get_embedder()
    query_vec = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
    query_vec = np.array(query_vec, dtype=np.float32)

    scores, indices = store["index"].search(query_vec, k)
    corpus = store["corpus"]

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        paper = corpus[idx]
        results.append({
            "id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors"][:2] if len(paper["authors"]) > 2 else paper["authors"],
            "year": paper["year"],
            "venue": paper["venue"],
            "keywords": paper.get("keywords", []),
            "score": float(score),
            "chunk_text": paper["chunk_text"],
        })
    return results
