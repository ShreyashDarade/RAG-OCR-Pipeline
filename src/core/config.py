from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    app_name: str = "OCR-rag"
    environment: str = "production"
    data_dir: Path = Path("data").resolve()

    # === ELASTICSEARCH CLOUD (Required) ===
    es_cloud_id: str = ""  # Elasticsearch Cloud ID (required)
    es_api_key: str = ""   # Elasticsearch API Key (required)
    es_index_text: str = "doc-text"
    es_index_tables: str = "doc-tables"
    es_index_images: str = "doc-images"

    # === OPENAI CONFIGURATION ===
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str | None = None  # For Azure OpenAI

    # Embedding / chunking
    chunk_size: int = 800
    chunk_overlap: int = 200  # Increased for better context
    keyword_top_k: int = 20

    # Retrieval settings
    retriever_top_k: int = 10  # Increased
    retriever_bm25_k1: float = 1.5
    retriever_bm25_b: float = 0.75
    hybrid_alpha: float = 0.5  # Equal weight BM25 and vector
    rerank_enabled: bool = True  # Enable re-ranking
    rerank_top_k: int = 6  # Final results after rerank

    # === DOCUMENT LINKAGE SETTINGS ===
    enable_cross_references: bool = True  # Link text, tables, images from same page
    page_context_window: int = 1  # Include adjacent pages in context
    min_chunk_size: int = 100  # Minimum chunk size to index

    allowed_file_extensions: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    # === PRODUCTION SETTINGS ===
    rate_limit_per_minute: int = 100
    cache_ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379"
    use_redis_cache: bool = False
    max_workers: int = 4
    request_timeout_seconds: int = 120

    # OCR settings
    ocr_gpu_enabled: bool = True
    ocr_batch_size: int = 5
    supported_ocr_languages: List[str] = ["en", "mr", "hi"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

