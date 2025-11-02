"""
Utilities for retrieving supplement snippets tied to anchor terms.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

from .AnchorUtils import extract_anchors


def _dedupe_lower(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _collect_search_terms(query_text: str, base_text: str) -> List[str]:
    terms: Set[str] = set()
    for anchor in extract_anchors(query_text or "") + extract_anchors(base_text or ""):
        if not anchor:
            continue
        lowered = anchor.lower()
        if lowered:
            terms.add(lowered)
        digits = "".join(ch for ch in lowered if ch.isdigit())
        if digits:
            terms.add(digits)
    return _dedupe_lower(terms)


def _fetch_latest_supplement_key(session, publication_key: str) -> Optional[str]:
    row = session.run(
        """
        MATCH (sup:Publication)-[:SUPPLEMENT_OF]->(base:Publication {publication_key: $base})
        RETURN sup.publication_key AS key
        ORDER BY sup.supplement_index DESC
        LIMIT 1
        """,
        {"base": publication_key},
    ).single()
    if not row:
        return None
    return row.get("key")


def _fetch_matching_supplement_texts(session, supplement_key: str, anchors: Sequence[str], window: int) -> List[str]:
    limit = max(3, window * 2 + 1) if window else 3
    rows = session.run(
        """
        MATCH (sup:Publication {publication_key: $pub})-[:CONTAINS]->(:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE size($anchors) = 0 OR any(a IN $anchors WHERE toLower(c.text) CONTAINS a)
        RETURN c.chunk_index AS idx, c.text AS text
        ORDER BY idx ASC
        LIMIT $limit
        """,
        {"pub": supplement_key, "anchors": list(anchors), "limit": limit},
    )
    return [row["text"] for row in rows]


def _fetch_latest_anchor_texts(session, anchors: Sequence[str], window: int) -> List[str]:
    if not anchors:
        return []
    limit = max(3, window * 2 + 1) if window else 3
    rows = session.run(
        """
        MATCH (p:Publication)-[:CONTAINS]->(:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE any(a IN $anchors WHERE toLower(c.text) CONTAINS a)
        RETURN p.publication_date AS date, p.supplement_index AS idx, c.chunk_index AS cidx, c.text AS text
        ORDER BY date DESC, idx DESC, cidx ASC
        LIMIT $limit
        """,
        {"anchors": list(anchors), "limit": limit},
    )
    return [row["text"] for row in rows]


def gather_supplement_snippets(
    *,
    session,
    query_text: str,
    base_text: str,
    window: int,
    publication_key: Optional[str],
) -> List[str]:
    """Return supplemental snippets that should accompany a base chunk.

    Parameters
    ----------
    session : neo4j.Session
        Active Neo4j session.
    query_text : str
        Original user query text.
    base_text : str
        Text of the chunk plus neighbor window.
    window : int
        Neighbor window size (used to bound supplement fetches as well).
    publication_key : Optional[str]
        Publication key for the base chunk; used to prefer supplements tied to
        the same publication family.
    """

    terms = _collect_search_terms(query_text, base_text)
    global_hits = _fetch_latest_anchor_texts(session, terms, int(window or 0))

    if global_hits:
        return global_hits

    if not publication_key:
        return []

    sup_key = _fetch_latest_supplement_key(session, publication_key)
    if not sup_key:
        return []

    local_hits = _fetch_matching_supplement_texts(session, sup_key, terms, int(window or 0))
    if local_hits:
        return local_hits

    # Final minimal fallback: fetch recent correction/supplement chunks in same issue
    limit = max(2, int(window or 0) + 1)
    rows = session.run(
        """
        MATCH (base:Publication {publication_key: $base})
        MATCH (sup:Publication {volume_number: base.volume_number, issue_number: base.issue_number})
        WHERE sup.supplement_index > 0
        WITH sup
        MATCH (sup)-[:CONTAINS]->(d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE toLower(d.doc_type) = 'correction' OR toLower(c.text) CONTAINS 'correction' OR toLower(c.text) CONTAINS 'تصحيح' OR toLower(c.text) CONTAINS 'ملحق'
        RETURN sup.publication_date AS date, sup.supplement_index AS idx, c.chunk_index AS cidx, c.text AS text
        ORDER BY date DESC, idx DESC, cidx ASC
        LIMIT $limit
        """,
        {"base": publication_key, "limit": limit},
    )
    return [row["text"] for row in rows]

