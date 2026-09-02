"""
Citation Agent — formats references in standard academic formats.
Handles APA, MLA, Vancouver, and PubMed citation styles.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List

from agents.orchestrator import AgentResult, AgentRole, AgentTask, BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class CitationAgent(BaseAgent):
    role = AgentRole.CITATION

    def __init__(self, llm_generate: Callable = None, **kwargs):
        super().__init__(llm_generate, **kwargs)

    async def execute(self, task: AgentTask) -> AgentResult:
        try:
            papers = task.input_data.get("papers", [])
            style = task.input_data.get("style", "apa")
            text_with_citations = task.input_data.get("text", "")

            if text_with_citations and self._llm:
                return await self._inline_cite(task)

            formatted = []
            for p in papers:
                citation = self._format_citation(p, style)
                formatted.append(citation)

            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                data={
                    "citations": formatted,
                    "style": style,
                    "count": len(formatted),
                },
            )
        except Exception as exc:
            logger.error(f"CitationAgent failed: {exc}")
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=str(exc),
            )

    async def _inline_cite(self, task: AgentTask) -> AgentResult:
        text = task.input_data.get("text", "")
        papers = task.input_data.get("papers", [])
        style = task.input_data.get("style", "apa")

        if not self._llm:
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error="No LLM available for inline citation",
            )

        paper_list = json.dumps(
            [
                {
                    "pmid": p.get("pmid", ""),
                    "title": p.get("title", ""),
                    "authors": p.get("authors", []),
                    "journal": p.get("journal", ""),
                    "year": p.get("year", ""),
                    "doi": p.get("doi", ""),
                }
                for p in papers[:20]
            ],
            indent=2,
        )

        prompt = (
            f"Add inline academic citations to this text using {style.upper()} style.\n\n"
            f"AVAILABLE PAPERS:\n{paper_list}\n\n"
            f"TEXT TO CITE:\n{text}\n\n"
            "Rules:\n"
            "- Add (Author, Year) or [number] citations where claims are made\n"
            "- Create a References section at the end\n"
            "- Only cite papers from the list above\n"
            "- Be precise about what each citation supports\n"
        )

        try:
            result = self._llm(prompt)
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=True,
                data={
                    "cited_text": result,
                    "style": style,
                    "papers_cited": len(papers[:20]),
                },
            )
        except Exception as exc:
            return AgentResult(
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error=str(exc),
            )

    def _format_citation(self, paper: dict, style: str) -> str:
        formatters = {
            "apa": _format_apa,
            "vancouver": _format_vancouver,
            "mla": _format_mla,
            "pubmed": _format_pubmed,
        }
        formatter = formatters.get(style, _format_apa)
        return formatter(paper)


def _format_apa(paper: dict) -> str:
    authors = paper.get("authors", [])
    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) <= 20:
        author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
    else:
        author_str = ", ".join(authors[:19]) + f", ... {authors[-1]}"

    year = paper.get("year", "n.d.")
    title = paper.get("title", "Untitled")
    journal = paper.get("journal", "")
    doi = paper.get("doi", "")
    pmid = paper.get("pmid", "")

    citation = f"{author_str} ({year}). {title}."
    if journal:
        citation += f" {journal}."
    if doi:
        citation += f" https://doi.org/{doi}"
    elif pmid:
        citation += f" PubMed: {pmid}"
    return citation


def _format_vancouver(paper: dict) -> str:
    authors = paper.get("authors", [])
    author_str = ", ".join(authors[:6])
    if len(authors) > 6:
        author_str += ", et al."

    year = paper.get("year", "")
    title = paper.get("title", "")
    journal = paper.get("journal", "")
    pmid = paper.get("pmid", "")

    citation = f"{author_str}. {title}. {journal}."
    if year:
        citation = f"{author_str}. {title}. {journal}. {year}."
    if pmid:
        citation += f" PMID: {pmid}."
    return citation


def _format_mla(paper: dict) -> str:
    authors = paper.get("authors", [])
    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]}, and {authors[1]}"
    else:
        author_str = f"{authors[0]}, et al."

    title = paper.get("title", "")
    journal = paper.get("journal", "")
    year = paper.get("year", "")
    doi = paper.get("doi", "")

    citation = f'{author_str}. "{title}."'
    if journal:
        citation += f" {journal},"
    if year:
        citation += f" {year}."
    if doi:
        citation += f" DOI: {doi}."
    return citation


def _format_pubmed(paper: dict) -> str:
    authors = paper.get("authors", [])
    author_str = ", ".join(authors[:6])
    if len(authors) > 6:
        author_str += ", et al."

    title = paper.get("title", "")
    journal = paper.get("journal", "")
    year = paper.get("year", "")
    pmid = paper.get("pmid", "")

    return f"{author_str}. {title}. {journal}. {year}. PMID: {pmid}."
