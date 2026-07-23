from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from env vars (see .env.example). Defaults target
    running everything locally without Docker."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — Ollama exposes an OpenAI-compatible API, so the same client
    # works against a cloud provider by swapping these three values.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"

    # Postgres (document/chunk metadata — the source of truth)
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/docintel"

    # Redis (Celery broker + result backend)
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant (vectors live here, not in Postgres)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"

    # Embeddings / retrieval
    embed_model: str = "BAAI/bge-m3"          # dense + sparse in one model
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    dense_dim: int = 1024                       # BGE-M3 dense size
    retrieve_top_k: int = 20                    # hybrid search candidates
    rerank_top_k: int = 5                       # kept after reranking
    chunk_size_words: int = 350                 # ~500 tokens
    chunk_overlap_words: int = 40
    # fp16 for the embed/rerank models. None = auto (fp16 on GPU, fp32 on CPU).
    # Set USE_FP16=true/false to force it.
    use_fp16: bool | None = None

    # Storage
    upload_dir: str = "./data/uploads"


settings = Settings()
