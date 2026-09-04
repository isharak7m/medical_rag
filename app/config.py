"""
Central configuration for MyoCortex.
Loads from .env file with explicit override of system environment variables.
This prevents stale system-level env vars from shadowing .env values.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import os

from pydantic_settings import BaseSettings

# Absolute path to .env — works regardless of working directory or subprocess
_ENV_FILE = Path(__file__).parent.parent / ".env"


def _force_load_env() -> None:
    """
    Manually parse .env and force-set os.environ BEFORE pydantic reads it.
    This ensures .env values always win over stale system environment variables.
    """
    if not _ENV_FILE.exists():
        return
    with open(_ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# Force load immediately on import
_force_load_env()


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "MyoCortex"
    APP_VERSION: str = "1.0.0"
    APP_DEBUG: bool = False

    # PubMed
    PUBMED_EMAIL: str = "isharak7m@gmail.com"
    PUBMED_API_KEY: str = ""
    PUBMED_MAX_RESULTS: int = 15
    PUBMED_RETMAX: int = 10
    PUBMED_RETRY_ATTEMPTS: int = 3
    PUBMED_RETRY_WAIT: float = 1.0

    # Embedding
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # FAISS
    FAISS_TOP_K: int = 12

    # Multi-query retrieval
    QUERY_EXPANSION_N: int = 6        # number of alternative queries to generate
    MULTI_QUERY_MAX_PAPERS: int = 80  # hard cap on merged papers before rerank

    # Decision Engine thresholds
    CONFIDENCE_HIGH_THRESHOLD: float = 0.70
    CONFIDENCE_MEDIUM_THRESHOLD: float = 0.40
    CONFLICT_THRESHOLD: float = 0.30

    # Cache
    CACHE_BACKEND: str = "memory"
    CACHE_DB_PATH: str = "myocortex_cache.db"

    # LLM
    LLM_BACKEND: str = "groq"
    GROQ_API_KEY: str = ""
    GROQ_MODEL_ID: str = "llama-3.3-70b-versatile"
    HF_API_TOKEN: str = ""
    HF_API_MODEL_ID: str = "katanemo/Arch-Router-1.5B"

    # Knowledge Graph
    NER_MODEL: str = "en_ner_bc5cdr_md"
    KG_DB_PATH: str = "knowledge_graph.db"

    # Research Version Control
    ARTIFACTS_DB_PATH: str = "research_artifacts.db"

    # Collaboration
    COLLAB_DB_PATH: str = "collaboration.db"

    # Auth
    JWT_SECRET: str = "myocortex-secret-key-change-in-production"
    JWT_EXPIRY_HOURS: int = 24
    USERS_DB_PATH: str = "users.db"

    # bioRxiv
    BIORXIV_MAX_RESULTS: int = 15
    BIORXIV_DAYS_BACK: int = 90

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
