from typing import Any, List
import re

from langchain_community.vectorstores.utils import maximal_marginal_relevance


def _get_text(hit: Any) -> str:
    if hasattr(hit, "text"):
        return getattr(hit, "text") or ""
    if isinstance(hit, dict):
        return hit.get("text") or ""
    return ""


def _embed_docs(embedder, docs: List[str]) -> List[List[float]]:
    if hasattr(embedder, "embed_documents"):
        return embedder.embed_documents(docs)
    if hasattr(embedder, "embed"):
        return embedder.embed(docs)
    return [embedder.embed_query(t or "") for t in docs]


# -----------------------------
# Anchor-aware preboost
# -----------------------------
_RX_ANCHORS = [
    re.compile(r"\bRFP[\s/-]?\d+\b", re.IGNORECASE),
    re.compile(r"\b\d{6,7}[\s/-]?RFP\b", re.IGNORECASE),
    re.compile(r"\bA/M/\d+\b", re.IGNORECASE),
    re.compile(r"\b5D[A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"\b\d{4}/\d{4}/\d+\b", re.IGNORECASE),
]

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _norm_digits(s: str) -> str:
    return (s or "").translate(_ARABIC_DIGITS)


def _extract_anchors_from_query(q: str) -> List[str]:
    qn = _norm_digits(q or "")
    out: List[str] = []
    for rx in _RX_ANCHORS:
        for m in rx.findall(qn):
            out.append(str(m))
    # dedupe order-preserving (case-insensitive)
    seen = set()
    uniq: List[str] = []
    for a in out:
        k = a.upper()
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq


def _preboost_results_by_anchors(query_text: str, results: List[Any]) -> List[Any]:
    if not results:
        return results
    anchors = _extract_anchors_from_query(query_text)
    if not anchors:
        return results
    anchors_norm = [ _norm_digits(a).lower() for a in anchors ]
    docs_norm = [ _norm_digits(_get_text(r)).lower() for r in results ]
    def is_match(i: int) -> bool:
        di = docs_norm[i]
        for a in anchors_norm:
            if a and a in di:
                return True
        return False
    idxs = list(range(len(results)))
    idxs.sort(key=lambda i: (not is_match(i), i))
    return [results[i] for i in idxs]


class MMRWrapperRetriever:
    def __init__(self, base, embedder, mmr_k: int, lambda_mult: float, always_keep_top):
        self._base = base
        self._embedder = embedder
        self._mmr_k = int(mmr_k)
        self._lambda = float(lambda_mult)
        self._keep_top_n = int(always_keep_top)

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def get_search_results(self, *args, **kwargs):
        rc = dict(kwargs.pop("retriever_config", {}) or {})
        top_k = int(rc.get("top_k") or self._mmr_k)
        # Forward using base signature (expects top_k rather than retriever_config)
        kwargs["top_k"] = top_k
        results: List[Any] = self._base.get_search_results(*args, **kwargs)
        if not results:
            return results
        try:
            query_text = kwargs.get("query_text") or (args[0] if args else "")
            results = _preboost_results_by_anchors(query_text, results)
            q = self._embedder.embed_query(query_text)
            docs = [_get_text(r) for r in results]
            X = _embed_docs(self._embedder, docs)
            k = min(self._mmr_k, len(results))
            if k <= 0:
                return results
            keep_n = int(self._keep_top_n)
            if keep_n > 0:
                keep_n = min(keep_n, k, len(results))
                if keep_n == k:
                    return results[:k]
                idxs_sub = maximal_marginal_relevance(q, X[keep_n:], lambda_mult=self._lambda, k=k - keep_n)
                mapped = list(range(keep_n)) + [i + keep_n for i in idxs_sub]
                return [results[i] for i in mapped]
            idxs = maximal_marginal_relevance(q, X, lambda_mult=self._lambda, k=k)
            return [results[i] for i in idxs]
        except Exception:
            return results

    def _invoke(self, method, *args, **kwargs):
        rc = dict(kwargs.pop("retriever_config", {}) or {})
        top_k = int(rc.get("top_k") or self._mmr_k)
        kwargs["top_k"] = top_k
        results: List[Any] = method(*args, **kwargs)
        if not results:
            return results
        try:
            k = min(self._mmr_k, len(results))
            if k <= 0:
                return results
            query_text = kwargs.get("query_text") or (args[0] if args else "")
            results = _preboost_results_by_anchors(query_text, results)
            q = self._embedder.embed_query(query_text)
            docs = [_get_text(r) for r in results]
            X = _embed_docs(self._embedder, docs)
            keep_n = int(self._keep_top_n)
            if keep_n > 0:
                keep_n = min(keep_n, k, len(results))
                if keep_n == k:
                    return results[:k]
                idxs_sub = maximal_marginal_relevance(q, X[keep_n:], lambda_mult=self._lambda, k=k - keep_n)
                mapped = list(range(keep_n)) + [i + keep_n for i in idxs_sub]
                return [results[i] for i in mapped]
            idxs = maximal_marginal_relevance(q, X, lambda_mult=self._lambda, k=k)
            return [results[i] for i in idxs]
        except Exception:
            return results

    def search(self, *args, **kwargs):
        if hasattr(self._base, "search"):
            return self._invoke(self._base.search, *args, **kwargs)
        return []

    def retrieve(self, *args, **kwargs):
        if hasattr(self._base, "retrieve"):
            return self._invoke(self._base.retrieve, *args, **kwargs)
        return []


def wrap_with_mmr(base, embedder, mmr_k: int, lambda_mult: float, always_keep_top: int):
    return MMRWrapperRetriever(base, embedder, mmr_k, lambda_mult, always_keep_top)


