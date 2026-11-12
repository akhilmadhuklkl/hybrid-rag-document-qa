from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chroma_path: Path = Path("chroma_data")
    collection_name: str = "company_documents"
    chunk_size: int = Field(default=180, ge=20)
    chunk_overlap: int = Field(default=30, ge=0)
    semantic_weight: float = Field(default=0.8, ge=0)
    keyword_weight: float = Field(default=0.2, ge=0)
    rrf_k: int = Field(default=60, ge=1)
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_candidate_multiplier: int = Field(default=4, ge=2, le=20)
    min_evidence_score: float = Field(default=0.25, ge=0, le=1)
    database_path: Path = Path("atlas.db")
    max_upload_mb: int = Field(default=20, ge=1, le=200)
    generator: str = Field(default="extractive", alias="RAG_GENERATOR")
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()

