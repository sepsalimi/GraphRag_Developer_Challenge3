from typing import Any, List, Optional

import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from neo4j_graphrag.types import RawSearchResult, RetrieverResult
from RAG.AnchorUtils import (
    filter_to_anchor_hits,
    enforce_slot_guard,
    preboost_results_by_anchors,
)


def _get_text(hit: Any) -> str:
    if hasattr(hit, "text"):
        return getattr(hit, "text") or ""
    if hasattr(hit, "content"):
        return getattr(hit, "content") or ""
    if isinstance(hit, dict):
        return hit.get("text") or ""
    return ""


# Embed funciton (e.g,)
def _embed_docs(embedder, docs: List[str]) -> List[List[float]]:
    if hasattr(embedder, "embed_documents"):
        return embedder.embed_documents(docs)
    if hasattr(embedder, "embed"):
        return embedder.embed(docs)
    return [embedder.embed_query(t or "") for t in docs]


def apply_mmr_list(
    query_text: str,
    items: List[Any],
    *,
    embedder,
    mmr_k: int,
    lambda_mult: float,
    always_keep_top: int = 0,
    preprocess: bool = True,
    original_results: Optional[List[Any]] = None,
    get_text=_get_text,
):
    hits = list(items)
    if not hits:
        return hits

    working = list(hits)
    if preprocess:
        filtered = filter_to_anchor_hits(query_text, working, get_text)
        if filtered:
            working = filtered
        working = preboost_results_by_anchors(query_text, working, get_text)

    k = min(int(mmr_k), len(working))
    if k <= 0:
        chosen = working
    else:
        q = np.asarray(embedder.embed_query(query_text), dtype=float)
        q = np.atleast_1d(q)
        if q.ndim > 1:
            q = q.reshape(-1)

        docs = [get_text(r) for r in working]
        X = np.asarray(_embed_docs(embedder, docs), dtype=float)
        X = np.atleast_2d(X)

        if q.size == 0 or X.size == 0 or X.shape[0] != len(working):
            chosen = working[:k]
        else:
            keep_n = max(0, min(int(always_keep_top), k))
            if keep_n >= k:
                chosen = working[:k]
            elif keep_n > 0:
                remaining = maximal_marginal_relevance(
                    q,
                    X[keep_n:],
                    lambda_mult=float(lambda_mult),
                    k=k - keep_n,
                )
                mapped = list(range(keep_n)) + [keep_n + i for i in remaining]
                chosen = [working[i] for i in mapped]
            else:
                idxs = maximal_marginal_relevance(
                    q, X, lambda_mult=float(lambda_mult), k=k
                )
                chosen = [working[i] for i in idxs]

    pool = list(original_results) if original_results is not None else hits
    return enforce_slot_guard(query_text, pool, chosen, get_text)

class MMRWrapperRetriever:
    def __init__(self, base, embedder, mmr_k: int, lambda_mult: float, always_keep_top):
        self._base = base
        self._embedder = embedder
        self._mmr_k = int(mmr_k)
        self._lambda = float(lambda_mult)
        self._keep_top_n = int(always_keep_top)

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def _unwrap(self, result_obj: Any):
        if isinstance(result_obj, RetrieverResult):
            metadata = result_obj.metadata
            return list(result_obj.items), lambda items: RetrieverResult(items=items, metadata=metadata)
        if isinstance(result_obj, RawSearchResult):
            metadata = result_obj.metadata
            return list(result_obj.records), lambda items: RawSearchResult(records=items, metadata=metadata)
        if isinstance(result_obj, list):
            return list(result_obj), lambda items: items
        return [], lambda items: result_obj

    def _apply_mmr(self, query_text: str, items: List[Any]) -> List[Any]:
        return apply_mmr_list(
            query_text,
            list(items),
            embedder=self._embedder,
            mmr_k=self._mmr_k,
            lambda_mult=self._lambda,
            always_keep_top=self._keep_top_n,
            preprocess=True,
            original_results=list(items),
            get_text=_get_text,
        )

    def _run(self, method, args, kwargs):
        rc = dict(kwargs.pop("retriever_config", {}) or {})
        top_k = int(rc.get("top_k") or self._mmr_k)
        kwargs["top_k"] = top_k
        result_obj = method(*args, **kwargs)
        items, rebuild = self._unwrap(result_obj)
        if not items:
            return rebuild(items)
        query_text = kwargs.get("query_text") or (args[0] if args else "")
        ranked = self._apply_mmr(query_text, items)
        return rebuild(ranked)

    def get_search_results(self, *args, **kwargs):
        if not hasattr(self._base, "get_search_results"):
            return []
        return self._run(self._base.get_search_results, args, kwargs)

    def search(self, *args, **kwargs):
        if not hasattr(self._base, "search"):
            return []
        return self._run(self._base.search, args, kwargs)

    def retrieve(self, *args, **kwargs):
        if not hasattr(self._base, "retrieve"):
            return []
        return self._run(self._base.retrieve, args, kwargs)


def wrap_with_mmr(base, embedder, mmr_k: int, lambda_mult: float, always_keep_top: int):
    return MMRWrapperRetriever(base, embedder, mmr_k, lambda_mult, always_keep_top)


