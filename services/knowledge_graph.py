"""
Knowledge Graph service — biomedical NER + relationship extraction.
Uses scispaCy for entity recognition and LLM for relationship extraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# Common biomedical entity type abbreviations
ENTITY_TYPES = {
    "DISEASE": "Disease",
    "CHEMICAL": "Chemical/Drug",
    "GENE": "Gene/Protein",
    "PROCEDURE": "Procedure",
    "ANATOMY": "Anatomy",
    "SYMPTOM": "Symptom",
}


@dataclass
class KGEntity:
    id: str
    name: str
    entity_type: str
    source_pmids: List[str] = field(default_factory=list)


@dataclass
class KGRelation:
    source_id: str
    target_id: str
    relation_type: str
    evidence_text: str
    pmid: str
    confidence: float = 0.8


@dataclass
class KnowledgeGraph:
    entities: Dict[str, KGEntity] = field(default_factory=dict)
    relations: List[KGRelation] = field(default_factory=list)

    def to_dict(self) -> dict:
        nodes = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "sources": e.source_pmids,
            }
            for e in self.entities.values()
        ]
        edges = [
            {
                "source": r.source_id,
                "target": r.target_id,
                "relation": r.relation_type,
                "evidence": r.evidence_text,
                "pmid": r.pmid,
                "confidence": r.confidence,
            }
            for r in self.relations
        ]
        return {"nodes": nodes, "edges": edges}


class KnowledgeGraphService:
    """Builds and queries a biomedical knowledge graph from paper abstracts."""

    def __init__(self, llm_generate=None):
        self._llm_generate = llm_generate
        self._nlp = None
        self._load_spacy_model()

    def _load_spacy_model(self) -> None:
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_ner_bc5cdr_md")
                logger.info("Loaded scispaCy BC5CDR model")
            except OSError:
                logger.warning("scispaCy BC5CDR model not found — using rule-based NER fallback")
                self._nlp = spacy.load("en_core_web_sm") if _model_exists("en_core_web_sm") else None
        except ImportError:
            logger.warning("spaCy not installed — using regex-based NER")

    def extract_entities(
        self, texts: List[Tuple[str, str]]
    ) -> List[KGEntity]:
        """
        Extract biomedical entities from (pmid, text) pairs.
        Returns deduplicated entities with source attribution.
        """
        entity_map: Dict[str, KGEntity] = {}

        for pmid, text in texts:
            if not text or not text.strip():
                continue

            raw_entities = self._extract_from_text(text)
            for name, etype in raw_entities:
                normalized = name.lower().strip()
                eid = f"{etype}:{normalized}"
                if eid in entity_map:
                    if pmid not in entity_map[eid].source_pmids:
                        entity_map[eid].source_pmids.append(pmid)
                else:
                    entity_map[eid] = KGEntity(
                        id=eid, name=name, entity_type=etype, source_pmids=[pmid]
                    )

        logger.info(f"Extracted {len(entity_map)} unique entities from {len(texts)} texts")
        return list(entity_map.values())

    def extract_relations(
        self, entities: List[KGEntity], texts: List[Tuple[str, str, str]]
    ) -> List[KGRelation]:
        """
        Extract relationships between entities.
        texts: list of (pmid, sentence, context) tuples.
        Uses LLM if available, otherwise co-occurrence heuristic.
        """
        if self._llm_generate and len(entities) > 0:
            return self._extract_relations_llm(entities, texts)
        return self._extract_relations_heuristic(entities, texts)

    def build_graph(
        self,
        papers_with_abstracts: List[Tuple[str, str]],
        query: str = "",
    ) -> KnowledgeGraph:
        """Build a complete knowledge graph from papers."""
        entities = self.extract_entities(papers_with_abstracts)
        texts = [(pmid, text, "") for pmid, text in papers_with_abstracts]
        relations = self.extract_relations(entities, texts)

        # Filter to only entities that have relations
        connected_ids: Set[str] = set()
        for r in relations:
            connected_ids.add(r.source_id)
            connected_ids.add(r.target_id)

        kg = KnowledgeGraph()
        for e in entities:
            if e.id in connected_ids or len(connected_ids) == 0:
                kg.entities[e.id] = e
        kg.relations = relations

        logger.info(
            f"Knowledge graph: {len(kg.entities)} entities, {len(kg.relations)} relations"
        )
        return kg

    def _extract_from_text(self, text: str) -> List[Tuple[str, str]]:
        entities = []
        if self._nlp and hasattr(self._nlp, "__call__"):
            try:
                doc = self._nlp(text[:10000])
                for ent in doc.ents:
                    etype = _map_spacy_label(ent.label_)
                    if etype and len(ent.text) > 1:
                        entities.append((ent.text.strip(), etype))
                return entities
            except Exception as exc:
                logger.debug(f"spaCy NER failed, falling back to regex: {exc}")

        return self._regex_ner(text)

    def _regex_ner(self, text: str) -> List[Tuple[str, str]]:
        entities = []
        # Disease patterns
        for m in re.finditer(
            r"\b(diabetes|cancer|obesity|hypertension|alzheimer|parkinson|"
            r"depression|anxiety|asthma|arthritis|stroke|inflammation|"
            r"atherosclerosis|dementia|epilepsy|migraine|hepatitis|"
            r"osteoporosis|fibromyalgia|lupus|crohn|colitis)\b",
            text, re.IGNORECASE
        ):
            entities.append((m.group(1).title(), "Disease"))

        # Drug/Chemical patterns
        for m in re.finditer(
            r"\b(metformin|aspirin|ibuprofen|creatine|omega-3|caffeine|"
            r"vitamin\s*[A-Z]?|omega-3|resveratrol|curcumin|insulin|"
            r"statin|antibiotic|corticosteroid|nsaid|acetaminophen|"
            r"loratadine|omeprazole|metoprolol|lisinopril|amlodipine)\b",
            text, re.IGNORECASE
        ):
            entities.append((m.group(0).strip(), "Chemical"))

        # Gene/protein patterns
        for m in re.finditer(
            r"\b(p53|BRCA[12]?|EGFR|KRAS|TP53|VEGF|mTOR|AMPK|SIRT[1-7]|"
            r"NF-?kB|TNF-?alpha?|IL-?[1-9]|FOXO[13]?|PGC-?1alpha?|"
            r"BDNF|IGF-?1?|HIF-?1alpha?)\b",
            text, re.IGNORECASE
        ):
            entities.append((m.group(0), "Gene"))

        return entities

    def _extract_relations_llm(
        self,
        entities: List[KGEntity],
        texts: List[Tuple[str, str, str]],
    ) -> List[KGRelation]:
        relations = []
        entity_names = {e.name.lower(): e for e in entities}

        # Process in batches of texts
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prompt = (
                "Given these biomedical entity pairs and the text context, "
                "identify relationships between entities.\n\n"
                f"Entities: {', '.join(e.name for e in entities[:30])}\n\n"
            )
            for pmid, text, _ in batch:
                prompt += f"PMID {pmid}: {text[:600]}\n\n"
            prompt += (
                "Return a JSON array of relationships. Each item: "
                '{"source": "entity_name", "target": "entity_name", '
                '"relation": "relates_to/treats/causes/associated_with/inhibits", '
                '"evidence": "brief text", "pmid": "id"}\n'
                "Return ONLY valid JSON array, no other text."
            )

            try:
                result = self._llm_generate(prompt)
                parsed = json.loads(result) if isinstance(result, str) else result
                if isinstance(parsed, list):
                    for item in parsed:
                        src = item.get("source", "").lower()
                        tgt = item.get("target", "").lower()
                        if src in entity_names and tgt in entity_names:
                            relations.append(
                                KGRelation(
                                    source_id=entity_names[src].id,
                                    target_id=entity_names[tgt].id,
                                    relation_type=item.get("relation", "relates_to"),
                                    evidence_text=item.get("evidence", ""),
                                    pmid=item.get("pmid", ""),
                                    confidence=0.7,
                                )
                            )
            except Exception as exc:
                logger.warning(f"LLM relation extraction failed: {exc}")

        return relations

    def _extract_relations_heuristic(
        self,
        entities: List[KGEntity],
        texts: List[Tuple[str, str, str]],
    ) -> List[KGRelation]:
        relations = []
        entity_map = {e.name.lower(): e for e in entities}
        seen = set()

        for pmid, text, _ in texts:
            text_lower = text.lower()
            found_in_text = [
                e for e in entities if e.name.lower() in text_lower
            ]
            for i in range(len(found_in_text)):
                for j in range(i + 1, len(found_in_text)):
                    e1, e2 = found_in_text[i], found_in_text[j]
                    if e1.entity_type == e2.entity_type:
                        continue
                    key = tuple(sorted([e1.id, e2.id]) + [pmid])
                    if key in seen:
                        continue
                    seen.add(key)
                    rel_type = _infer_relation_type(e1.entity_type, e2.entity_type)
                    relations.append(
                        KGRelation(
                            source_id=e1.id,
                            target_id=e2.id,
                            relation_type=rel_type,
                            evidence_text=f"Co-occur in PMID {pmid}",
                            pmid=pmid,
                            confidence=0.5,
                        )
                    )

        return relations


def _map_spacy_label(label: str) -> Optional[str]:
    mapping = {
        "CHEMICAL": "Chemical",
        "DISEASE": "Disease",
        "DISO": "Disease",
        "ORGANISM": None,
        "GENE_OR_GENE_PRODUCT": "Gene",
        "SIMPLE_CHEMICAL": "Chemical",
        "ANATOMICAL_SYSTEM": "Anatomy",
        "PROCEDURE": "Procedure",
    }
    return mapping.get(label)


def _model_exists(model_name: str) -> bool:
    try:
        import spacy
        spacy.load(model_name)
        return True
    except (OSError, ImportError):
        return False


def _infer_relation_type(type1: str, type2: str) -> str:
    if {type1, type2} == {"Chemical", "Disease"}:
        return "treats"
    if {type1, type2} == {"Gene", "Disease"}:
        return "associated_with"
    if {type1, type2} == {"Chemical", "Gene"}:
        return "modulates"
    if "Anatomy" in (type1, type2):
        return "localized_in"
    return "relates_to"
