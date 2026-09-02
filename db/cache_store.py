"""
Cache storage abstraction supporting in-memory and SQLite backends.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from db.schemas import QueryResponse, RichQueryResponse
from utils.logger import get_logger

logger = get_logger(__name__)


def _deserialize_response(payload: str) -> QueryResponse | RichQueryResponse:
    data = json.loads(payload)
    if "evidence_cards" in data:
        return RichQueryResponse(**data)
    return QueryResponse(**data)


class BaseCacheStore(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[QueryResponse | RichQueryResponse]:
        ...

    @abstractmethod
    def set(self, key: str, response: BaseModel) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class MemoryCacheStore(BaseCacheStore):
    def __init__(self) -> None:
        self._store: dict[str, QueryResponse | RichQueryResponse] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[QueryResponse | RichQueryResponse]:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, response: BaseModel) -> None:
        with self._lock:
            if isinstance(response, RichQueryResponse):
                self._store[key] = response
            else:
                self._store[key] = QueryResponse(**response.model_dump())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class SQLiteCacheStore(BaseCacheStore):
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get(self, key: str) -> Optional[QueryResponse | RichQueryResponse]:
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT response_json FROM query_cache WHERE query_key = ?",
                    (key,),
                ).fetchone()
                if not row:
                    return None
                return _deserialize_response(row[0])

    def set(self, key: str, response: BaseModel) -> None:
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO query_cache (query_key, response_json)
                    VALUES (?, ?)
                    ON CONFLICT(query_key) DO UPDATE SET response_json = excluded.response_json
                    """,
                    (key, response.model_dump_json()),
                )
                conn.commit()

    def clear(self) -> None:
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM query_cache")
                conn.commit()


def build_cache_store(backend: str, db_path: str) -> BaseCacheStore:
    if backend == "sqlite":
        logger.info(f"Using SQLite cache: {db_path}")
        return SQLiteCacheStore(db_path)
    logger.info("Using in-memory cache")
    return MemoryCacheStore()
