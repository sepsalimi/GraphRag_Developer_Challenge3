"""Chunk normalization utilities used during ingestion.

This module provides helpers to:

* Normalize chunk text (digit normalization, whitespace cleanup).
* Parse simple Markdown tables exported from Gazette Markdown files.
* Extract procurement-related facts (closing date, fee, bond value) from tables.
* Provide a flattened key/value representation of tables for retrieval.

The ingestion notebook calls :func:`normalize_chunk` for each chunk and persists the
returned facts back into Neo4j.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_WS_RE = re.compile(r"[ \t]+")
_LINE_WS_RE = re.compile(r"\s*\n\s*")


@dataclass
class Table:
    headers: List[str]
    rows: List[Dict[str, str]]
    raw: str


def normalize_text(raw: str) -> str:
    """Normalize digits and condense whitespace without removing line breaks."""

    if not raw:
        return ""
    text = raw.translate(_ARABIC_DIGITS)
    # Replace hard tabs and multiple spaces but keep newlines intact.
    text = _WS_RE.sub(" ", text)
    # Trim whitespace around newlines to avoid ragged formatting.
    text = _LINE_WS_RE.sub("\n", text)
    return text.strip()


def _split_row(row: str) -> List[str]:
    parts = [cell.strip() for cell in row.strip().strip("|").split("|")]
    return parts


_ALIGN_RE = re.compile(r"^[:\-\s]+$")


def _normalize_table_row(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("|") and stripped.count("|") >= 2:
        return stripped
    idx = stripped.find("|")
    if idx >= 0:
        candidate = stripped[idx:].strip()
        if candidate.startswith("|") and candidate.count("|") >= 2:
            return candidate
    return None


def parse_markdown_tables(text: str) -> List[Table]:
    """Extract pipe-delimited tables, tolerating missing alignment rows."""

    if not text:
        return []

    lines = text.splitlines()
    tables: List[Table] = []
    i = 0
    while i < len(lines):
        normalized = _normalize_table_row(lines[i])
        if not normalized:
            i += 1
            continue

        table_lines: List[str] = []
        while i < len(lines):
            candidate = _normalize_table_row(lines[i])
            if not candidate:
                break
            table_lines.append(candidate)
            i += 1

        if len(table_lines) < 2:
            continue

        header = _split_row(table_lines[0])
        if not header:
            continue

        data_candidates = table_lines[1:]
        if data_candidates and all(
            all(_ALIGN_RE.match(cell.strip() or "") for cell in _split_row(line))
            for line in data_candidates[:1]
        ):
            # Skip leading alignment row if present.
            data_candidates = data_candidates[1:]

        if not data_candidates:
            continue

        rows: List[Dict[str, str]] = []
        for raw_row in data_candidates:
            values = _split_row(raw_row)
            if len(values) < len(header):
                values.extend([""] * (len(header) - len(values)))
            row_dict = {
                header[idx].strip(): values[idx].strip()
                for idx in range(len(header))
            }
            rows.append(row_dict)

        tables.append(Table(headers=header, rows=rows, raw="\n".join(table_lines)))

    return tables


def _parse_date(value: str) -> Optional[str]:
    if not value:
        return None
    candidate = value.translate(_ARABIC_DIGITS).strip()
    if not candidate:
        return None

    # Accept formats like 2025-07-14, 14/07/2025, 14-07-2025.
    patterns = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]
    for pattern in patterns:
        try:
            dt = datetime.strptime(candidate, pattern)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return candidate


def _parse_money(value: str) -> Optional[int]:
    if not value:
        return None
    normalized = value.translate(_ARABIC_DIGITS)
    match = re.search(r"\d[\d,]*", normalized)
    if not match:
        return None
    digits = match.group(0).replace(",", "")
    try:
        return int(digits)
    except ValueError:
        return None


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_header_key(header: str) -> str:
    key = header.translate(_ARABIC_DIGITS).lower()
    return _NON_ALNUM_RE.sub(" ", key).strip()


def _contains_all(key: str, *terms: str) -> bool:
    return all(term in key for term in terms)


def extract_procurement_facts(tables: Iterable[Table]) -> Dict[str, Optional[Any]]:
    """Extract closing date, price (KD), and guarantee (KD) from tables."""

    facts: Dict[str, Optional[Any]] = {
        "closing_date": None,
        "price_kd": None,
        "guarantee_kd": None,
    }

    for table in tables:
        for row in table.rows:
            for header, value in row.items():
                if not value:
                    continue
                key_norm = _normalize_header_key(header)
                if facts["closing_date"] is None and (
                    _contains_all(key_norm, "closing", "date")
                    or "deadline" in key_norm
                    or _contains_all(key_norm, "bid", "submission")
                ):
                    parsed = _parse_date(value)
                    if parsed:
                        facts["closing_date"] = parsed
                elif facts["price_kd"] is None and (
                    "price" in key_norm
                    or "fee" in key_norm
                    or (
                        "cost" in key_norm
                        and ("document" in key_norm or "tender" in key_norm)
                    )
                ):
                    parsed_price = _parse_money(value)
                    if parsed_price is not None:
                        facts["price_kd"] = parsed_price
                elif facts["guarantee_kd"] is None and (
                    "guarantee" in key_norm
                    or "bond" in key_norm
                    or "security" in key_norm
                ):
                    parsed_bond = _parse_money(value)
                    if parsed_bond is not None:
                        facts["guarantee_kd"] = parsed_bond

    return facts


def flatten_table_to_kv(tables: Iterable[Table]) -> str:
    """Return a flattened key/value representation of tables."""

    lines: List[str] = []
    for table in tables:
        for row in table.rows:
            for header, value in row.items():
                if value:
                    lines.append(f"{header.strip()}: {value.strip()}")
    return "\n".join(lines)


def normalize_chunk(text: str) -> Dict[str, Any]:
    """Compute normalized text, flattened table text, and procurement facts."""

    tables = parse_markdown_tables(text)
    facts = extract_procurement_facts(tables)
    table_kv = flatten_table_to_kv(tables) if tables else ""
    return {
        "text_norm": normalize_text(text),
        "facts": facts,
        "table_kv": table_kv or None,
    }


__all__ = [
    "Table",
    "normalize_text",
    "parse_markdown_tables",
    "extract_procurement_facts",
    "flatten_table_to_kv",
    "normalize_chunk",
]
