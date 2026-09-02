"""
Cache service — application-level cache logic sitting on top of CacheStore.
Handles key normalisation and serialization concerns.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

from db.cache_store import BaseCacheStore
from db.schemas import QueryResponse, RichQueryResponse
from utils.text_cleaning import normalize
from utils.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    def __init__(self, store: BaseCacheStore) -> None:
        self._store = store

    def _make_key(self, query: str) -> str:
        """Normalize query to a stable cache key."""
        return normalize(query)

    def get(self, query: str) -> Optional[QueryResponse | RichQueryResponse]:
        key = self._make_key(query)
        result = self._store.get(key)
        if result:
            logger.info(f"Cache hit for: {query!r}")
        return result

    def set(self, query: str, response: BaseModel) -> None:
        key = self._make_key(query)
        self._store.set(key, response)
        logger.info(f"Cached response for: {query!r}")
