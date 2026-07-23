"""Qdrant hybrid vector store.

The collection stores two vectors per chunk under named slots:
  - "dense"  : BGE-M3 dense embedding (cosine)
  - "sparse" : BGE-M3 lexical weights

Search prefetches candidates from both and fuses them with Reciprocal Rank
Fusion (RRF) — true hybrid retrieval, not just cosine top-k.
"""
from functools import lru_cache

from qdrant_client import QdrantClient, models

from .config import settings


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    c = client()
    if c.collection_exists(settings.qdrant_collection):
        return
    c.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.dense_dim, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


def upsert_chunks(points: list[dict]) -> None:
    """Each point: {id, dense, sparse:{indices,values}, payload}."""
    c = client()
    c.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=p["id"],
                vector={
                    "dense": p["dense"],
                    "sparse": models.SparseVector(
                        indices=p["sparse"]["indices"],
                        values=p["sparse"]["values"],
                    ),
                },
                payload=p["payload"],
            )
            for p in points
        ],
    )


def hybrid_search(
    dense: list[float],
    sparse: dict,
    limit: int,
    document_id: str | None = None,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Hybrid search, optionally scoped to one document (``document_id``) or a
    set of documents (``document_ids``, e.g. the docs pinned to a conversation).
    With neither, the whole collection is searched."""
    c = client()
    flt = None
    ids = list(document_ids) if document_ids else ([document_id] if document_id else [])
    if ids:
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=ids),
                )
            ]
        )
    res = c.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=limit, filter=flt),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse["indices"], values=sparse["values"]
                ),
                using="sparse",
                limit=limit,
                filter=flt,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    out = []
    for point in res.points:
        payload = point.payload or {}
        out.append(
            {
                "chunk_id": str(point.id),
                "score": float(point.score),
                "text": payload.get("text", ""),
                "page": payload.get("page", 0),
                "document_id": payload.get("document_id", ""),
            }
        )
    return out


def delete_document(document_id: str) -> None:
    c = client()
    c.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )
