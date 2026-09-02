"""
Embedding service — wraps sentence-transformers.
Singleton pattern: model loaded once at startup.
"""

from __future__ import annotations
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """
    Thin wrapper around SentenceTransformer.
    Exposes embed() for single texts and embed_batch() for efficiency.
    """

    def __init__(self) -> None:
        logger.info(
            f"Loading embedding model: {settings.EMBEDDING_MODEL} "
            f"on device: {settings.EMBEDDING_DEVICE}"
        )
        self._model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready. Dim={self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string → 1D float32 ndarray."""
        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return vector.astype("float32")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple strings → 2D float32 ndarray (N, dim)."""
        if not texts:
            return np.empty((0, self._dim), dtype="float32")
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.astype("float32")
