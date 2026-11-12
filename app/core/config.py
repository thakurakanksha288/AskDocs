"""Application configuration loaded from environment variables."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "AskDocs"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://askdocs:askdocs@localhost:5432/askdocs"
    db_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    embedding_cache_ttl: int = 86400

    # Vector store
    faiss_index_path: str = "data/faiss.index"
    faiss_id_map_path: str = "data/faiss_id_map.json"

    # Embeddings
    embedding_model: str = "models/text-embedding-004"
    embedding_batch_size: int = 32
    embedding_dim: int = 768

    # Chunking
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 1024

    # Retrieval
    default_top_k: int = 5

    # Upload limits
    max_file_size_mb: int = 20
    allowed_extensions: list[str] = [".pdf", ".txt", ".md"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
