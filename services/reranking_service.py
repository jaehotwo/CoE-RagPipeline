"""Reusable reranking utilities for structured search results."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


def rrf_fusion(rankings: Dict[str, Dict[str, int]], k0: int = 60) -> Dict[str, float]:
    """Reciprocal rank fusion for combining multiple ranked lists."""
    fused: Dict[str, float] = {}
    for item_id, ranks in rankings.items():
        score = 0.0
        for rank in ranks.values():
            score += 1.0 / (k0 + max(1, int(rank)))
        fused[item_id] = score
    return fused


class CrossEncoderReranker:
    """Optional FlagEmbedding-based cross encoder reranker.

    Falls back gracefully when the library or model is unavailable.
    """

    def __init__(self, model_name: Optional[str] = None, use_fp16: bool = True):
        self.available = False
        self.model = None
        self.model_name = model_name or "BAAI/bge-reranker-base"
        try:
            from FlagEmbedding import FlagReranker  # type: ignore

            self.model = FlagReranker(self.model_name, use_fp16=use_fp16)
            self.available = True
            logger.info("CrossEncoderReranker loaded: %s", self.model_name)
        except Exception as exc:  # pragma: no cover - best effort import
            logger.warning(
                "CrossEncoderReranker unavailable (install FlagEmbedding). Falling back to base ranking. Reason: %s",
                exc,
            )

    def rerank(
        self,
        query: str,
        docs: List[Tuple[str, Dict[str, Any]]],
        top_n: int = 50,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return reranked documents sorted by relevance."""
        if not self.available or not self.model:
            return [(text, 0.0, metadata) for text, metadata in docs[:top_n]]

        try:
            pairs = [[query, text] for text, _metadata in docs]
            scores = self.model.compute_score(pairs)
            if isinstance(scores, float):
                scores = [scores]
            scored: List[Tuple[str, float, Dict[str, Any]]] = []
            for (text, metadata), raw_score in zip(docs, scores):
                try:
                    score = float(raw_score)
                except Exception:
                    score = 0.0
                scored.append((text, score, metadata))
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_n]
        except Exception as exc:  # pragma: no cover - best effort fallback
            logger.error("CrossEncoderReranker failed: %s", exc)
            return [(text, 0.0, metadata) for text, metadata in docs[:top_n]]

