"""
PubMed service — all HTTP interactions with NCBI E-utilities.
Implements retry with exponential backoff via tenacity.
"""

from __future__ import annotations
import asyncio
import re
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from db.schemas import Paper
from app.config import get_settings
from utils.logger import get_logger
from utils.text_cleaning import clean_abstract

logger = get_logger(__name__)
settings = get_settings()

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedService:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_papers(self, query: str) -> List[Paper]:
        """
        Search PubMed for `query` and return a list of Paper objects.
        Returns empty list on failure (logged, not raised).
        """
        try:
            pmids = await self._search(query)
            if not pmids:
                logger.warning(f"PubMed: no results for query: {query!r}")
                return []
            papers = await self._fetch_details(pmids)
            logger.info(f"PubMed: fetched {len(papers)} papers for: {query!r}")
            return papers
        except Exception as exc:
            logger.error(f"PubMed fetch failed: {exc}")
            return []

    async def fetch_papers_multi(
        self,
        queries: List[str],
        max_total: int = 35,
    ) -> List[Paper]:
        """
        Fetch papers for multiple queries concurrently, deduplicate by PMID.

        Args:
            queries   : list of PubMed query strings (from query expander)
            max_total : hard cap on total papers before reranking

        Returns:
            Deduplicated list of Paper objects, capped at max_total.
        """
        # Run all fetches concurrently
        results = await asyncio.gather(
            *[self.fetch_papers(q) for q in queries],
            return_exceptions=True,
        )

        # Merge + deduplicate by PMID
        seen: Dict[str, Paper] = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Query {i+1} fetch failed: {result}")
                continue
            for paper in result:
                if paper.pmid not in seen:
                    seen[paper.pmid] = paper

            logger.info(
                f"Query {i+1}/{len(queries)}: '{queries[i]}' -> "
                f"{len(result) if not isinstance(result, Exception) else 0} papers"
            )

        merged = list(seen.values())
        logger.info(
            f"Multi-query fetch: {len(merged)} unique papers from "
            f"{len(queries)} queries (cap={max_total})"
        )

        # Cap total before reranking to prevent overload
        return merged[:max_total]

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal: ESearch
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(settings.PUBMED_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=settings.PUBMED_RETRY_WAIT, min=1, max=10
        ),
        reraise=True,
    )
    async def _search(self, query: str) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": settings.PUBMED_MAX_RESULTS,
            "retmode": "json",
            "email": settings.PUBMED_EMAIL,
        }
        if settings.PUBMED_API_KEY:
            params["api_key"] = settings.PUBMED_API_KEY
        resp = await self._client.get(ESEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    # ------------------------------------------------------------------
    # Internal: EFetch
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(settings.PUBMED_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=settings.PUBMED_RETRY_WAIT, min=1, max=10
        ),
        reraise=True,
    )
    async def _fetch_details(self, pmids: List[str]) -> List[Paper]:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "email": settings.PUBMED_EMAIL,
        }
        if settings.PUBMED_API_KEY:
            params["api_key"] = settings.PUBMED_API_KEY
        resp = await self._client.get(EFETCH_URL, params=params)
        resp.raise_for_status()
        return self._parse_xml(resp.text)

    # ------------------------------------------------------------------
    # Internal: XML parsing
    # ------------------------------------------------------------------

    def _parse_xml(self, xml_text: str) -> List[Paper]:
        papers: List[Paper] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error(f"PubMed XML parse error: {exc}")
            return papers

        for article in root.iter("PubmedArticle"):
            pmid = self._extract_text(article, ".//PMID")
            title = self._extract_text(article, ".//ArticleTitle")
            abstract_parts = [
                el.text or ""
                for el in article.findall(".//AbstractText")
            ]
            abstract = clean_abstract(" ".join(abstract_parts))
            sample_size = self._extract_sample_size(abstract)

            if pmid and title and abstract:
                papers.append(
                    Paper(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        sample_size=sample_size,
                    )
                )
        return papers

    @staticmethod
    def _extract_text(element: ET.Element, path: str) -> str:
        el = element.find(path)
        return (el.text or "").strip() if el is not None else ""

    @staticmethod
    def _extract_sample_size(text: str) -> Optional[int]:
        """Heuristic: look for 'n = 123' or '123 participants/patients/subjects'."""
        patterns = [
            r"\bn\s*=\s*(\d+)",
            r"(\d+)\s+(?:participants|patients|subjects|individuals)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        return None
