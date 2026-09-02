"""
FAISS vector store abstraction.
Creates a fresh flat index per query (no persistence needed for per-query retrieval).
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import faiss

from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Lightweight per-query FAISS index.
    Usage:
        store = VectorStore(dim=384)
        store.add(vectors)          # np.ndarray shape (N, dim)
        hits = store.search(q, k=5) # returns (indices, distances)
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._count = 0
        logger.debug(f"FAISS IndexFlatIP created with dim={dim}")

    def add(self, vectors: np.ndarray) -> None:
        """Add a batch of L2-normalised vectors."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected shape (N, {self.dim}), got {vectors.shape}"
            )
        # Normalise to unit length so inner product == cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vectors = (vectors / norms).astype("float32")
        self._index.add(vectors)
        self._count += len(vectors)
        logger.debug(f"Added {len(vectors)} vectors. Total: {self._count}")

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Returns list of (index, cosine_similarity) sorted descending.
        """
        if self._count == 0:
            return []

        q = query_vector.reshape(1, -1).astype("float32")
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        k = min(top_k, self._count)
        distances, indices = self._index.search(q, k)

        results = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0
        ]
        logger.debug(f"FAISS search returned {len(results)} hits")
        return results
