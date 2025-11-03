from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import neo4j
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.retrievers.hybrid import HybridSearchRanker, SearchQueryParseError
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

from RAG.AnchorUtils import (
    extract_anchors,
    filter_to_anchor_hits,
    preboost_results_by_anchors,
    has_slot_near_anchor,
    norm_digits,
    wants_slots,
)
from RAG.AnchorUtils import enforce_slot_guard as _enforce_slot_guard
from RAG.MMR import apply_mmr_list
from RAG.Lucene import escape_fulltext_query
from RAG.AnchorPrimer import fetch_anchor_chunks


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _item_text(item: RetrieverResultItem) -> str:
    return (getattr(item, "content", None) or "")


def _matches_scope(item: RetrieverResultItem, allowed: Sequence[str]) -> bool:
    if not allowed:
        return True
    lowered = {str(s).lower() for s in allowed if s}
    if not lowered:
        return True
    metadata = getattr(item, "metadata", {}) or {}
    candidates = [
        metadata.get("document_key"),
        metadata.get("source"),
        metadata.get("file"),
    ]
    for cand in candidates:
        if not cand:
            continue
        c = str(cand).lower()
        if any(c == target or c.endswith(target) or target in c for target in lowered):
            return True
    text = _item_text(item).lower()
    return any(target in text for target in lowered)


def _filter_by_scope(
    items: Sequence[RetrieverResultItem], allowed: Sequence[str]
) -> Tuple[List[RetrieverResultItem], bool]:
    if not allowed:
        return list(items), False
    filtered = [item for item in items if _matches_scope(item, allowed)]
    if filtered:
        return filtered, True
    return list(items), False


def _ensure_metadata(item: RetrieverResultItem) -> RetrieverResultItem:
    metadata = dict(getattr(item, "metadata", {}) or {})
    doc_key = metadata.get("document_key")
    if doc_key and "source" not in metadata:
        metadata["source"] = str(doc_key).split(":")[0]
    if metadata.get("publication_key") and "source" not in metadata:
        metadata["source"] = str(metadata["publication_key"])

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    single_page = metadata.get("page")
    if page_start is None and single_page is not None:
        page_start = page_end = single_page
    metadata["page_start"] = page_start
    metadata["page_end"] = page_end

    additions: List[str] = []
    closing = metadata.get("closing_date")
    if closing:
        additions.append(f"Closing Date: {closing}")
    price = metadata.get("price_kd")
    if price is not None:
        additions.append(f"Tender Document Price: {price} KD")
    guarantee = metadata.get("guarantee_kd")
    if guarantee is not None:
        additions.append(f"Preliminary Bond: {guarantee} KD")
    table_kv = metadata.get("table_kv")
    if table_kv:
        additions.append(str(table_kv))

    if additions and not metadata.get("__facts_augmented"):
        base = (getattr(item, "content", None) or "").rstrip()
        snippet = "\n".join(additions).strip()
        if base:
            item.content = f"{base}\n\n{snippet}"
        else:
            item.content = snippet
        metadata["__facts_augmented"] = True

    item.metadata = metadata
    return item


@dataclass
class HybridRetrievalConfig:
    vector_index: str
    fulltext_index: str
    alpha: float
    pool_multiplier: float = 4.0
    pool_min: int = 80
    pool_max: int = 300
    apply_mmr: bool = True
    mmr_k: int = 5
    mmr_lambda: float = 0.3
    mmr_keep_top: int = 1
    neo4j_database: Optional[str] = None


class SafeHybridRetriever(HybridRetriever):
    """Hybrid retriever that provides clearer errors when the vector index isn't found.

    The upstream implementation raises an AttributeError when no record is returned;
    this wrapper falls back to the generic SHOW INDEXES query and surfaces a helpful
    message instead of crashing during import.
    """

    def _fetch_index_infos(self, vector_index_name: str) -> None:  # type: ignore[override]
        query = (
            "SHOW VECTOR INDEXES "
            "YIELD name, labelsOrTypes, properties, options "
            "WHERE name = $index_name "
            "RETURN labelsOrTypes as labels, properties, "
            "options.indexConfig.`vector.dimensions` as dimensions"
        )
        result = self.driver.execute_query(
            query,
            {"index_name": vector_index_name},
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )
        records = list(result.records)
        if not records:
            fallback = (
                "SHOW INDEXES YIELD name, type, labelsOrTypes, properties "
                "WHERE type = 'VECTOR' AND name = $index_name "
                "RETURN labelsOrTypes AS labels, properties"
            )
            result = self.driver.execute_query(
                fallback,
                {"index_name": vector_index_name},
                database_=self.neo4j_database,
                routing_=neo4j.RoutingControl.READ,
            )
            records = list(result.records)
        if not records:
            raise RuntimeError(
                f"Vector index '{vector_index_name}' not found. "
                "Ensure it exists (e.g., CREATE VECTOR INDEX Chunk ...) or set NEO4J_VECTOR_INDEX."
            )
        record = records[0]
        labels = record.get("labels") or record.get("labelsOrTypes")
        properties = record.get("properties")
        dimensions = record.get("dimensions")
        if not labels or not properties:
            raise RuntimeError(
                f"Unable to read index metadata for '{vector_index_name}'. "
                "Check Neo4j permissions and index definition."
            )
        self._node_label = labels[0]
        self._embedding_node_property = properties[0]
        if dimensions is not None:
            self._embedding_dimension = dimensions


class HybridRetrievalPipeline:
    def __init__(
        self,
        driver,
        embedder,
        reranker,
        *,
        config: HybridRetrievalConfig,
    ) -> None:
        self._config = config
        self._driver = driver
        self._embedder = embedder
        self._reranker = reranker
        self._hybrid = SafeHybridRetriever(
            driver,
            config.vector_index,
            config.fulltext_index,
            embedder,
            neo4j_database=config.neo4j_database,
        )

    def _pool_size(self, top_k: int) -> int:
        cfg = self._config
        est = max(top_k, int(cfg.pool_multiplier * max(top_k, 1)))
        est = max(est, cfg.pool_min)
        est = min(est, cfg.pool_max)
        return est

    def retrieve(
        self,
        query_text: str,
        *,
        top_k: int,
        scope: Optional[Sequence[str]] = None,
    ) -> Tuple[RetrieverResult, Dict[str, object]]:
        pool_k = self._pool_size(top_k)
        sanitize_enabled = _env_bool("LUCENE_ESCAPE_ENABLED", True)
        query_fulltext = escape_fulltext_query(query_text) if sanitize_enabled else query_text
        sanitized = query_fulltext != query_text
        vector_fallback = False

        try:
            result = self._hybrid.search(
                query_text=query_fulltext,
                top_k=pool_k,
                ranker=HybridSearchRanker.LINEAR,
                alpha=self._config.alpha,
            )
        except SearchQueryParseError:
            if not _env_bool("VECTOR_FALLBACK_ON_PARSE_ERROR", True):
                raise
            vector_fallback = True
            query_vector = self._embedder.embed_query(query_text)
            result = self._hybrid.search(
                query_vector=query_vector,
                top_k=pool_k,
                ranker=HybridSearchRanker.NAIVE,
            )

        anchors = extract_anchors(query_text or "")
        anchor_primer_items: List[RetrieverResultItem] = []
        if anchors and _env_bool("ANCHOR_PRIMER_ENABLED", True):
            try:
                primer_limit = int(os.getenv("ANCHOR_PRIMER_LIMIT", "200"))
            except ValueError:
                primer_limit = 200
            primer_limit = max(primer_limit, 0)
            try:
                rows = fetch_anchor_chunks(self._driver, anchors, primer_limit)
            except Exception:
                rows = []
            for row in rows:
                text_value = row.get("text")
                if not text_value:
                    continue
                metadata = {
                    "document_key": row.get("document_key"),
                    "chunk_index": row.get("chunk_index"),
                    "page_start": row.get("page_start"),
                    "page_end": row.get("page_end"),
                    "__source": "anchor_primer",
                }
                item = RetrieverResultItem(content=text_value, metadata=metadata)
                anchor_primer_items.append(_ensure_metadata(item))
        anchor_primer_count = len(anchor_primer_items)
        if anchor_primer_items:
            base_items = [_ensure_metadata(item) for item in result.items]
            combined_items: List[RetrieverResultItem] = []
            seen_keys = set()
            for item in anchor_primer_items + base_items:
                metadata = getattr(item, "metadata", None) or {}
                doc_key = metadata.get("document_key")
                chunk_idx = metadata.get("chunk_index")
                key = (doc_key, chunk_idx) if doc_key is not None and chunk_idx is not None else None
                if key is not None:
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                combined_items.append(item)
            result = RetrieverResult(items=combined_items, metadata=result.metadata or {})
        else:
            anchor_primer_count = 0
            result = RetrieverResult(
                items=[_ensure_metadata(item) for item in result.items],
                metadata=result.metadata or {},
            )

        diagnostics: Dict[str, object] = {
            "pool_top_k": pool_k,
            "hybrid_retrieved": len(result.items),
            "ranker": (HybridSearchRanker.NAIVE.value if vector_fallback else HybridSearchRanker.LINEAR.value),
            "alpha": self._config.alpha,
            "fulltext_sanitized": sanitized,
            "vector_fallback": vector_fallback,
        }
        diagnostics["anchor_primer_candidates"] = anchor_primer_count
        diagnostics["union_total"] = len(result.items)

        scoped_items, scope_applied = _filter_by_scope(result.items, scope or [])
        diagnostics["scope_applied"] = scope_applied
        diagnostics["scope_hits"] = len(scoped_items)

        anchor_hits = filter_to_anchor_hits(query_text, scoped_items, _item_text)
        diagnostics["anchor_hits"] = len(anchor_hits)
        if anchor_hits:
            working = preboost_results_by_anchors(query_text, anchor_hits, _item_text)
        else:
            working = scoped_items

        ordered: List[RetrieverResultItem]

        if self._reranker and working:
            rerank_results = self._reranker.rerank(
                query_text,
                [_item_text(item) for item in working],
                top_n=len(working),
            )
            diagnostics["rerank_model"] = self._reranker.model
            diagnostics["rerank_top"] = len(rerank_results)
            seen = set()
            ordered = []
            for res in rerank_results:
                if 0 <= res.index < len(working) and res.index not in seen:
                    ordered.append(working[res.index])
                    seen.add(res.index)
            for idx, item in enumerate(working):
                if idx not in seen:
                    ordered.append(item)
        else:
            ordered = list(working)

        diagnostics["ordered"] = len(ordered)

        mmr_applied = False
        selected = ordered
        if self._config.apply_mmr and top_k > 1 and ordered:
            mmr_k = min(max(self._config.mmr_k, top_k), len(ordered))
            selected = apply_mmr_list(
                query_text,
                ordered,
                embedder=self._embedder,
                mmr_k=mmr_k,
                lambda_mult=self._config.mmr_lambda,
                always_keep_top=self._config.mmr_keep_top,
                preprocess=False,
                original_results=ordered,
                get_text=_item_text,
            )
            mmr_applied = True

        diagnostics["mmr_applied"] = mmr_applied

        final_items = selected[:top_k] if top_k > 0 else selected
        diagnostics["final"] = len(final_items)

        guarded = _enforce_slot_guard(
            query_text,
            ordered,
            final_items,
            _item_text,
        )
        diagnostics["slot_guarded"] = len(guarded)

        strict_guard = _env_bool("STRICT_SLOT_GUARD", False)
        if strict_guard and anchors:
            wants = wants_slots(query_text)
            if any(wants):
                anchors_norm = [norm_digits(a).lower() for a in anchors]
                evidence_found = any(
                    has_slot_near_anchor(query_text, _item_text(item), anchors_norm, wants)
                    for item in guarded
                )
                if not evidence_found:
                    guarded = []

        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "pool_top_k": pool_k,
                "scope_applied": scope_applied,
                "anchor_hits": diagnostics["anchor_hits"],
                "mmr_applied": mmr_applied,
            }
        )

        return RetrieverResult(items=guarded, metadata=metadata), diagnostics


__all__ = ["HybridRetrievalPipeline", "HybridRetrievalConfig"]

