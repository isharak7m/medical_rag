"""
bioRxiv API integration — fetches preprints alongside PubMed papers.
Provides access to the latest unpublished research.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

import httpx

from db.schemas import Paper
from utils.logger import get_logger

logger = get_logger(__name__)

BIORXIV_API = "https://api.biorxiv.org/details/biorxiv"


class BioRxivService:
    """Fetches preprints from the bioRxiv API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info("BioRxiv service ready")

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_preprints(
        self,
        query: str,
        max_results: int = 15,
        days_back: int = 90,
    ) -> List[Paper]:
        """
        Search bioRxiv for preprints matching query.
        Uses the /details endpoint with cursor-based pagination.
        """
        try:
            papers = await self._search_preprints(query, max_results, days_back)
            logger.info(f"bioRxiv returned {len(papers)} preprints for '{query}'")
            return papers
        except Exception as exc:
            logger.warning(f"bioRxiv fetch failed: {exc}")
            return []

    async def _search_preprints(
        self, query: str, max_results: int, days_back: int
    ) -> List[Paper]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        papers: List[Paper] = []
        cursor = 0
        attempts = 0
        max_attempts = 5

        while len(papers) < max_results and attempts < max_attempts:
            attempts += 1
            params = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "cursor": cursor,
                "per_page": min(100, max_results - len(papers) + 20),
            }
            try:
                resp = await self._client.get(BIORXIV_API, params=params)
                if resp.status_code != 200:
                    logger.warning(f"bioRxiv status {resp.status_code}")
                    break
                data = resp.json()
                collection = data.get("collection", [])
                if not collection:
                    break

                for item in collection:
                    title = item.get("title", "").strip()
                    abstract = item.get("abstract", "").strip()
                    if not title:
                        continue

                    # Simple keyword relevance check
                    query_terms = set(query.lower().split())
                    text = f"{title} {abstract}".lower()
                    if not any(t in text for t in query_terms if len(t) > 2):
                        continue

                    doi = item.get("doi", "")
                    biorxiv_id = item.get("biorxiv_id", doi)
                    authors_list = [
                        a.strip() for a in item.get("authors", "").split(";") if a.strip()
                    ]

                    papers.append(
                        Paper(
                            pmid=f"biorxiv:{biorxiv_id}",
                            title=title,
                            abstract=abstract,
                            sample_size=self._extract_sample_size(abstract),
                        )
                    )
                    if len(papers) >= max_results:
                        break

                cursor += len(collection)
                if len(collection) < params["per_page"]:
                    break

            except Exception as exc:
                logger.warning(f"bioRxiv page fetch error: {exc}")
                break

        return papers

    @staticmethod
    def _extract_sample_size(abstract: str) -> Optional[int]:
        import re
        patterns = [
            r"[nN]\s*=\s*(\d[\d,]*)",
            r"(\d[\d,]*)\s+participants",
            r"(\d[\d,]*)\s+subjects",
            r"enrolled\s+(\d[\d,]*)",
        ]
        for pattern in patterns:
            m = re.search(pattern, abstract)
            if m:
                try:
                    return int(m.group(1).replace(",", ""))
                except ValueError:
                    continue
        return None
