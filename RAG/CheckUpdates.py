"""Utilities for retrieving update snippets tied to anchors or supplements."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .AnchorUtils import extract_anchors, norm_digits

_MAX_ANCHORS = 3


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _normalize_anchor_value(text: str) -> str:
    cleaned = norm_digits(text or "")
    cleaned = re.sub(r"[\u200f\u200e]", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.upper()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*([/-])\s*", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _collect_anchor_payloads(query_text: str, base_text: str) -> Tuple[List[Dict[str, str]], List[str]]:
    raw = extract_anchors(query_text or "") + extract_anchors(base_text or "")
    payloads: List[Dict[str, str]] = []
    terms: List[str] = []
    seen_values = set()
    seen_terms = set()
    for anchor in raw:
        normalized = _normalize_anchor_value(anchor)
        if not normalized or normalized in seen_values:
            continue
        seen_values.add(normalized)
        lower = normalized.lower()
        digits = "".join(ch for ch in normalized if ch.isdigit())
        payloads.append({
            "value": normalized,
            "lower": lower,
            "digits": digits,
        })
        for term in (lower, digits):
            if term and term not in seen_terms:
                seen_terms.add(term)
                terms.append(term)
    return payloads, terms


def _format_anchor_rows(rows: Sequence[Dict[str, object]]) -> List[str]:
    snippets: List[str] = []
    for row in rows:
        texts = row.get("texts") or []
        if not texts:
            continue
        header_parts = [f"[anchor={row.get('anchor_value')}]"]
        doc_key = row.get("document_key")
        if doc_key:
            header_parts.append(f"[doc={doc_key}]")
        status = row.get("status")
        if status:
            header_parts.append(f"[status={status}]")
        pub_date = row.get("publication_date")
        if pub_date:
            header_parts.append(f"[date={pub_date}]")
        header = " ".join(header_parts)
        snippets.append("\n".join([header] + list(texts)))
    return snippets


def _format_publication_rows(rows: Sequence[Dict[str, object]], label: str) -> List[str]:
    snippets: List[str] = []
    for row in rows:
        texts = row.get("texts") or []
        if not texts:
            continue
        header_parts = [f"[{label}={row.get('publication_key')}]"]
        status = row.get("status")
        if status:
            header_parts.append(f"[status={status}]")
        header_parts.append(f"[date={row.get('publication_date')}]")
        header = " ".join(part for part in header_parts if part)
        snippets.append("\n".join([header] + list(texts)))
    return snippets


def _fetch_anchor_updates(session, anchors: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, object]]:
    if not anchors:
        return []
    rows = session.run(
        """
        UNWIND $anchors AS anchor
        MATCH (a:Anchor {value: anchor.value})-[:LATEST]->(d:Document)-[:PUBLISHED_IN]->(p:Publication)
        MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH anchor, d, p, c
        ORDER BY c.chunk_index ASC
        WITH anchor, d, p, collect({
            text: c.text,
            matched: (
                anchor.lower = "" OR toLower(c.text) CONTAINS anchor.lower OR
                (anchor.digits <> "" AND toLower(c.text) CONTAINS anchor.digits)
            )
        }) AS chunk_rows
        WITH anchor, d, p,
             [row IN chunk_rows WHERE row.matched][0..$limit] AS matched_rows,
             chunk_rows[0..$limit] AS fallback_rows
        WITH anchor, d, p,
             CASE WHEN size(matched_rows) > 0 THEN matched_rows ELSE fallback_rows END AS chosen
        RETURN anchor.value AS anchor_value,
               d.document_key AS document_key,
               d.status AS status,
               p.publication_date AS publication_date,
               [row IN chosen | row.text] AS texts
        """,
        {"anchors": anchors, "limit": limit},
    ).data()
    return rows


def _fetch_publication_pointer(session, publication_key: str) -> Optional[str]:
    row = session.run(
        """
        MATCH (base:Publication {publication_key: $base})
        RETURN base.latest_updated_by_publication_key AS latest
        """,
        {"base": publication_key},
    ).single()
    if not row:
        return None
    latest = row.get("latest")
    return latest


def _fetch_publication_updates(
    session,
    publication_key: Optional[str],
    limit: int,
    terms: Sequence[str],
) -> List[Dict[str, object]]:
    if not publication_key:
        return []
    pointer = _fetch_publication_pointer(session, publication_key)
    if not pointer:
        return []
    rows = session.run(
        """
        MATCH (sup:Publication {publication_key: $supp})-[:CONTAINS]->(d:Document)
        MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WHERE size($terms) = 0 OR any(term IN $terms WHERE toLower(c.text) CONTAINS term)
        WITH sup, d, c
        ORDER BY c.chunk_index ASC
        WITH sup, d, collect(c.text)[0..$limit] AS texts
        RETURN sup.publication_key AS publication_key,
               sup.publication_date AS publication_date,
               d.status AS status,
               texts
        """,
        {"supp": pointer, "terms": list(terms), "limit": limit},
    ).data()
    return rows


def _fetch_local_issue_updates(
    session,
    publication_key: Optional[str],
    limit: int,
    terms: Sequence[str],
) -> List[Dict[str, object]]:
    if not publication_key:
        return []
    rows = session.run(
        """
        MATCH (base:Publication {publication_key: $base})
        MATCH (other:Publication {volume_number: base.volume_number, issue_number: base.issue_number})
        MATCH (other)-[:CONTAINS]->(d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE coalesce(d.status, 'announced') <> 'announced'
          AND coalesce(d.superseded, false) = false
          AND (size($terms) = 0 OR any(term IN $terms WHERE toLower(c.text) CONTAINS term))
        WITH other, d, c
        ORDER BY other.publication_date DESC, c.chunk_index ASC
        WITH other, d, collect(c.text)[0..$limit] AS texts
        RETURN other.publication_key AS publication_key,
               other.publication_date AS publication_date,
               d.status AS status,
               texts
        LIMIT $limit
        """,
        {"base": publication_key, "terms": list(terms), "limit": limit},
    ).data()
    return rows


def gather_update_snippets(
    *,
    session,
    query_text: str,
    base_text: str,
    window: int,
    publication_key: Optional[str],
) -> List[str]:
    """Return snippets that reflect the latest updates for the given context."""

    anchor_payloads, search_terms = _collect_anchor_payloads(query_text, base_text)
    anchor_payloads = anchor_payloads[:_MAX_ANCHORS]
    limit = max(3, int(window or 0) * 2 + 1)

    anchor_rows = _fetch_anchor_updates(session, anchor_payloads, limit)
    anchor_snippets = _format_anchor_rows(anchor_rows)
    if anchor_snippets:
        return anchor_snippets

    publication_rows = _fetch_publication_updates(session, publication_key, limit, search_terms)
    publication_snippets = _format_publication_rows(publication_rows, "update")
    if publication_snippets:
        return publication_snippets

    local_rows = _fetch_local_issue_updates(session, publication_key, limit, search_terms)
    local_snippets = _format_publication_rows(local_rows, "issue")
    return local_snippets


__all__ = ["gather_update_snippets"]
