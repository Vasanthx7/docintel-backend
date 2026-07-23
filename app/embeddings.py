"""BGE-M3 embedding wrapper.

BGE-M3 produces a dense vector AND a sparse (lexical) vector for the same text
in a single forward pass, which is exactly what we need for hybrid search.
The model is loaded lazily as a process-wide singleton so the (heavy) weights
load once per worker.
"""
from functools import lru_cache

from .config import settings
from .device import use_fp16


@lru_cache(maxsize=1)
def _model():
    from FlagEmbedding import BGEM3FlagModel

    # fp16 on GPU (faster, less VRAM); fp32 on CPU (correct/fast). Auto-resolved.
    return BGEM3FlagModel(settings.embed_model, use_fp16=use_fp16())


def _to_sparse(lexical_weights: dict) -> dict:
    """Convert BGE-M3 lexical weights {token_id: weight} into the
    {indices, values} shape Qdrant expects for a sparse vector."""
    indices, values = [], []
    for token_id, weight in lexical_weights.items():
        indices.append(int(token_id))
        values.append(float(weight))
    return {"indices": indices, "values": values}


def embed_texts(texts: list[str]) -> list[dict]:
    """Return one {"dense": [...], "sparse": {"indices": [...], "values": [...]}}
    per input text."""
    out = _model().encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]
    sparse = out["lexical_weights"]
    return [
        {"dense": dense[i].tolist(), "sparse": _to_sparse(sparse[i])}
        for i in range(len(texts))
    ]


def embed_query(text: str) -> dict:
    return embed_texts([text])[0]
