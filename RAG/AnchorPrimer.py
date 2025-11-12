
"""Anchor-driven chunk retrieval helpers."""

from __future__ import annotations

from typing import Iterable, List, Dict, Any

from neo4j import Driver

from RAG.AnchorUtils import norm_digits


def _normalize_anchor_value(text: str) -> str:
    value = norm_digits(text or "")
    value = value.replace("‏", "").replace("‎", "")
    value = value.strip()
    if not value:
        return ""
    value = " ".join(value.upper().split())
    value = value.replace(" / ", "/").replace(" - ", "-")
    value = value.replace("/ ", "/").replace(" /", "/")
    value = value.replace("- ", "-").replace(" -", "-")
    return value


def fetch_anchor_chunks(
    driver: Driver,
    anchors: Iterable[str],
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return chunk rows for the latest documents tied to the given anchors."""

    normalized = []
    seen = set()
    for anchor in anchors:
        value = _normalize_anchor_value(anchor)
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    if not normalized or limit <= 0:
        return []

    cypher = (
        """
        UNWIND $values AS value
        MATCH (a:Anchor {value: value})-[:LATEST]->(d:Document)
        MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        RETURN d.document_key AS document_key,
               d.publication_key AS publication_key,
               c.chunk_index   AS chunk_index,
               c.text          AS text,
               c.page_start    AS page_start,
               c.page_end      AS page_end,
               c.closing_date  AS closing_date,
               c.price_kd      AS price_kd,
               c.guarantee_kd  AS guarantee_kd,
               c.table_kv      AS table_kv,
               c.source        AS source
        ORDER BY coalesce(c.chunk_index, 0) ASC
        LIMIT $limit
        """
    )

    records, _, _ = driver.execute_query(cypher, values=normalized, limit=int(limit))
    return records


__all__ = ["fetch_anchor_chunks"]
