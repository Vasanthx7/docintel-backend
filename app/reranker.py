"""Cross-encoder reranking with bge-reranker-v2-m3.

Reranking a broad candidate set (top-20 from hybrid search) down to a small,
high-precision set (top-5) is the single highest-ROI accuracy lever in this
pipeline. Loaded lazily as a singleton.
"""
from functools import lru_cache

from .config import settings
from .device import use_fp16


@lru_cache(maxsize=1)
def _reranker():
    from FlagEmbedding import FlagReranker

    return FlagReranker(settings.reranker_model, use_fp16=use_fp16())


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """`candidates` are dicts that must contain a "text" key. Returns the
    top_k candidates sorted by cross-encoder relevance, each with a
    "rerank_score" added."""
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = _reranker().compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
