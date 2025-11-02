from typing import Any, List

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


class MMRWrapperRetriever:
    def __init__(self, base, embedder, mmr_k: int, lambda_mult: float, always_keep_top):
        self._base = base
        self._embedder = embedder
        self._mmr_k = int(mmr_k)
        self._lambda = float(lambda_mult)
        try:
            self._keep_top = bool(int(always_keep_top))
        except Exception:
            self._keep_top = True

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
            q = self._embedder.embed_query(query_text)
            docs = [_get_text(r) for r in results]
            X = _embed_docs(self._embedder, docs)
            k = min(self._mmr_k, len(results))
            if k <= 0:
                return results
            if self._keep_top and len(results) > 0:
                if k == 1:
                    return [results[0]]
                idxs_sub = maximal_marginal_relevance(q, X[1:], lambda_mult=self._lambda, k=k-1)
                mapped = [0] + [i + 1 for i in idxs_sub]
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
            q = self._embedder.embed_query(query_text)
            docs = [_get_text(r) for r in results]
            X = _embed_docs(self._embedder, docs)
            if self._keep_top and len(results) > 0:
                if k == 1:
                    return [results[0]]
                idxs_sub = maximal_marginal_relevance(q, X[1:], lambda_mult=self._lambda, k=k-1)
                mapped = [0] + [i + 1 for i in idxs_sub]
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


