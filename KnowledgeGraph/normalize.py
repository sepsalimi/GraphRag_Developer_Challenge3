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


def parse_markdown_tables(text: str) -> List[Table]:
    """Extract GitHub-style Markdown tables from a blob of text."""

    if not text:
        return []

    lines = text.splitlines()
    tables: List[Table] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not (line.startswith("|") and line.count("|") >= 2):
            i += 1
            continue

        if i + 1 >= len(lines):
            i += 1
            continue

        align = lines[i + 1].strip()
        if not (align.startswith("|") and re.fullmatch(r"\|[\s:-]+\|", align.replace(" ", ""))):
            i += 1
            continue

        headers = _split_row(line)
        data_lines: List[str] = []
        j = i + 2
        while j < len(lines):
            row = lines[j].strip()
            if not row.startswith("|"):
                break
            data_lines.append(row)
            j += 1

        if not headers or not data_lines:
            i = j
            continue

        rows: List[Dict[str, str]] = []
        for raw_row in data_lines:
            values = _split_row(raw_row)
            # Ensure we have the same number of columns as headers.
            if len(values) < len(headers):
                values.extend([""] * (len(headers) - len(values)))
            row_dict = {
                headers[idx].strip(): values[idx].strip()
                for idx in range(len(headers))
            }
            rows.append(row_dict)

        tables.append(Table(headers=headers, rows=rows, raw="\n".join([line, align] + data_lines)))
        i = j

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


_CLOSING_HEADERS = {
    "closing date",
    "deadline",
    "scheduled date for bid submission (closing date)",
}
_PRICE_HEADERS = {
    "cost of tender documents",
    "price",
    "document price",
    "tender document price",
}
_GUARANTEE_HEADERS = {
    "value of preliminary bond",
    "guarantee",
    "initial security",
    "value of provisional bond",
}


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
                key = header.lower().strip()
                if key in _CLOSING_HEADERS and facts["closing_date"] is None:
                    parsed = _parse_date(value)
                    if parsed:
                        facts["closing_date"] = parsed
                elif key in _PRICE_HEADERS and facts["price_kd"] is None:
                    parsed_price = _parse_money(value)
                    if parsed_price is not None:
                        facts["price_kd"] = parsed_price
                elif key in _GUARANTEE_HEADERS and facts["guarantee_kd"] is None:
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
