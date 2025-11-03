from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    import cohere  # type: ignore
    from cohere.core.api_error import ApiError  # type: ignore
except ImportError as exc:  # pragma: no cover
    cohere = None
    ApiError = RuntimeError  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class CohereReranker:
    """Minimal Cohere reranker wrapper with fail-loud semantics."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_candidates: Optional[int] = None,
    ) -> None:
        if cohere is None:
            raise ImportError(
                "cohere package is required for CohereReranker"
            ) from _IMPORT_ERROR

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY must be set in the environment or provided explicitly"
            )

        self._client = cohere.Client(api_key=key)
        self._model = model or os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
        cap = max_candidates or os.getenv("COHERE_RERANK_MAX_CANDIDATES")
        self._max_candidates = int(cap) if cap else 300

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []

        limit = len(documents)
        if top_n is not None:
            limit = min(limit, int(top_n))
        limit = min(limit, self._max_candidates)
        if limit <= 0:
            return []

        payload = [{"text": doc or ""} for doc in documents]

        try:
            response = self._client.rerank(
                model=self._model,
                query=query,
                documents=payload,
                top_n=limit,
            )
        except ApiError:
            raise

        scored: List[RerankResult] = []
        for item in response.results:
            score = getattr(item, "score", None)
            if score is None:
                score = getattr(item, "relevance_score", None)
            if score is None:
                raise AttributeError(
                    "Cohere rerank response missing score/relevance_score field; check SDK version"
                )
            scored.append(RerankResult(index=int(item.index), score=float(score)))
        scored.sort(key=lambda r: (-r.score, r.index))
        return scored


__all__ = ["CohereReranker", "RerankResult"]

