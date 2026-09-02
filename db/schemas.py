"""
Shared Pydantic schemas across the system.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Stance(str, Enum):
    SUPPORT = "support"
    OPPOSE = "oppose"
    NEUTRAL = "neutral"


class Verdict(str, Enum):
    STRONG_SUPPORT = "STRONG_SUPPORT"
    MODERATE_SUPPORT = "MODERATE_SUPPORT"
    CONFLICTED = "CONFLICTED"
    WEAK = "WEAK"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Paper(BaseModel):
    pmid: str
    title: str
    abstract: str
    sample_size: Optional[int] = None


class RelevanceLabel(str, Enum):
    DIRECTLY_RELEVANT = "directly_relevant"
    INDIRECTLY_RELEVANT = "indirectly_relevant"
    CONTEXTUAL = "contextual"
    IRRELEVANT = "irrelevant"


class Claim(BaseModel):
    paper_pmid: str
    paper_title: str
    claim_text: str
    stance: Stance
    relevance_label: RelevanceLabel = RelevanceLabel.DIRECTLY_RELEVANT
    intervention_match: bool = False
    outcome_match: bool = False


class RankedEvidence(BaseModel):
    claim: Claim
    score: float


class ContradictionReport(BaseModel):
    support_count: int
    oppose_count: int
    neutral_count: int
    has_conflict: bool
    scope_note: str = ""


class DecisionResult(BaseModel):
    verdict: Verdict
    confidence: Confidence
    support_score: float
    oppose_score: float
    total_score: float
    support_ratio: float
    oppose_ratio: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class SourceItem(BaseModel):
    title: str
    pmid: str


class QueryResponse(BaseModel):
    claim: str
    confidence: Confidence
    evidence_strength: str
    supporting_studies: int
    opposing_studies: int
    summary: str
    sources: List[SourceItem]


class CachedEntry(BaseModel):
    query_key: str
    response: QueryResponse


class EvidenceCard(BaseModel):
    pmid: str
    title: str
    claim_text: str
    stance: Stance
    relevance_score: float
    sample_size: Optional[int]
    relevance_label: RelevanceLabel = RelevanceLabel.DIRECTLY_RELEVANT


class ContradictionSummary(BaseModel):
    has_conflict: bool
    support_count: int
    oppose_count: int
    neutral_count: int
    explanation: str
    scope_note: str = ""


class ClaimLink(BaseModel):
    claim_text: str
    stance: Stance
    paper_title: str
    pmid: str


class ConfidenceDetail(BaseModel):
    label: Confidence
    score: int
    explanation: str


class PipelineDiagnostics(BaseModel):
    search_queries: List[str] = Field(default_factory=list)
    total_unique_papers: int = 0
    papers_after_filter: int = 0
    final_ranked_papers: int = 0
    support_count: int = 0
    oppose_count: int = 0
    neutral_count: int = 0
    direct_evidence_ratio: float = 0.0
    cache_hit: bool = False
    notes: List[str] = Field(default_factory=list)
    directly_relevant_count: int = 0
    indirectly_relevant_count: int = 0
    irrelevant_count: int = 0
    avg_relevance_score: float = 0.0


class RichQueryResponse(BaseModel):
    query: str
    interpreted_query: str = ""
    final_answer: str
    summary: str
    confidence: ConfidenceDetail
    verdict: Verdict
    evidence_strength: str
    evidence_cards: List[EvidenceCard]
    claim_links: List[ClaimLink]
    contradiction: ContradictionSummary
    faithfulness_score: float = 0.0
    coverage_score: float = 0.0
    retrieval_score: float = 0.0
    llm_backend_used: str = "unknown"
    diagnostics: PipelineDiagnostics = Field(default_factory=PipelineDiagnostics)


class EvalLog(BaseModel):
    query: str
    retrieved_pmids: List[str]
    final_answer: str
    faithfulness_score: float
    coverage_score: float
    retrieval_score: float
    verdict: str
    confidence_score: int
    diagnostics: PipelineDiagnostics = Field(default_factory=PipelineDiagnostics)


class LiteratureReviewRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    style: str = "comprehensive"


class LiteratureReviewResponse(BaseModel):
    query: str
    review: str
    papers_reviewed: int
    sections: List[str] = Field(default_factory=list)


class KnowledgeGraphResponse(BaseModel):
    nodes: List[dict] = Field(default_factory=list)
    edges: List[dict] = Field(default_factory=list)
    entity_count: int = 0
    relation_count: int = 0
    entity_types: dict = Field(default_factory=dict)
    top_entities: List[dict] = Field(default_factory=list)


class ArtifactRequest(BaseModel):
    artifact_type: str = Field(..., pattern="^(note|hypothesis|manuscript|review)$")
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    author: str = "system"
    tags: List[str] = Field(default_factory=list)
    message: str = "Created"


class ArtifactUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    message: str = "Updated"
    author: str = "system"
    tags: Optional[List[str]] = None


class ArtifactResponse(BaseModel):
    artifact_id: str
    artifact_type: str
    title: str
    content: str
    current_version: int
    created_at: str
    updated_at: str
    author: str
    tags: List[str] = Field(default_factory=list)


class VersionResponse(BaseModel):
    version_id: str
    version_number: int = 0
    content: str
    message: str
    created_at: str
    author: str
    diff_from_previous: str = ""
    tags: List[str] = Field(default_factory=list)


class WorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    created_by: str = "system"


class CommentRequest(BaseModel):
    author: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    artifact_id: Optional[str] = None
    parent_comment_id: Optional[str] = None


class WorkspaceResponse(BaseModel):
    workspace_id: str
    name: str
    description: str
    created_by: str
    created_at: str
    updated_at: str
    members: List[dict] = Field(default_factory=list)
    shared_artifacts: List[str] = Field(default_factory=list)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    task_type: str = "query"
    style: str = "apa"
    draft_type: str = "literature_review"


class AgentQueryResponse(BaseModel):
    task_type: str
    result: dict = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
