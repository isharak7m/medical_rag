"""
NLI-based evidence classification.
Uses a fine-tuned DeBERTa model for biomedical natural language inference,
with LLM fallback for zero-shot classification.
"""

from __future__ import annotations

from typing import List, Tuple

from db.schemas import Claim, RankedEvidence, Stance
from utils.logger import get_logger

logger = get_logger(__name__)

_NLI_LABELS = ["entailment", "neutral", "contradiction"]
_LABEL_TO_STANCE = {
    "entailment": Stance.SUPPORT,
    "contradiction": Stance.OPPOSE,
    "neutral": Stance.NEUTRAL,
}


class NLIClassifier:
    """
    Hybrid NLI classifier:
    - Primary: cross-encoder sentence-transformers model (fast, local)
    - Fallback: LLM zero-shot prompt (slower but higher quality)
    """

    def __init__(self, llm_generate=None):
        self._llm_generate = llm_generate
        self._model = None
        self._model_name = "cross-encoder/nli-deberta-v3-base"
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info(f"NLI model loaded: {self._model_name}")
        except Exception as exc:
            logger.warning(f"Failed to load NLI model: {exc} — will use LLM fallback")

    def classify_pair(
        self, premise: str, hypothesis: str
    ) -> Tuple[Stance, float]:
        """Classify a single premise-hypothesis pair."""
        if self._model is not None:
            return self._classify_with_model(premise, hypothesis)
        if self._llm_generate:
            return self._classify_with_llm(premise, hypothesis)
        return Stance.NEUTRAL, 0.5

    def classify_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[Stance, float]]:
        """Classify multiple premise-hypothesis pairs."""
        if self._model is not None and len(pairs) > 0:
            try:
                import numpy as np
                scores = self._model.predict(pairs)
                results = []
                for score_row in scores:
                    if hasattr(score_row, "__len__") and len(score_row) >= 3:
                        probs = _softmax(score_row)
                        best_idx = int(probs.argmax())
                        stance = _LABEL_TO_STANCE.get(_NLI_LABELS[best_idx], Stance.NEUTRAL)
                        confidence = float(probs[best_idx])
                    else:
                        stance, confidence = Stance.NEUTRAL, 0.5
                    results.append((stance, confidence))
                return results
            except Exception as exc:
                logger.warning(f"Batch NLI classification failed: {exc}")

        return [self.classify_pair(p, h) for p, h in pairs]

    def _classify_with_model(
        self, premise: str, hypothesis: str
    ) -> Tuple[Stance, float]:
        try:
            scores = self._model.predict([(premise, hypothesis)])
            probs = _softmax(scores[0])
            best_idx = int(probs.argmax())
            stance = _LABEL_TO_STANCE.get(_NLI_LABELS[best_idx], Stance.NEUTRAL)
            confidence = float(probs[best_idx])
            return stance, confidence
        except Exception as exc:
            logger.warning(f"NLI model classification failed: {exc}")
            return Stance.NEUTRAL, 0.5

    def _classify_with_llm(
        self, premise: str, hypothesis: str
    ) -> Tuple[Stance, float]:
        prompt = (
            "Classify the relationship between these two statements.\n"
            f"STATEMENT 1 (premise): {premise[:500]}\n"
            f"STATEMENT 2 (hypothesis): {hypothesis[:500]}\n\n"
            "Respond with EXACTLY one word: entailment, neutral, or contradiction.\n"
            "Classification:"
        )
        try:
            result = self._llm_generate(prompt).strip().lower()
            for label in _NLI_LABELS:
                if label in result:
                    return _LABEL_TO_STANCE[label], 0.7
            return Stance.NEUTRAL, 0.5
        except Exception as exc:
            logger.warning(f"LLM NLI classification failed: {exc}")
            return Stance.NEUTRAL, 0.5


def _softmax(values) -> list:
    import math
    if hasattr(values, "tolist"):
        values = values.tolist()
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps]
