from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Reference:
    source: str
    detail: Optional[str] = None


def _coerce_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _format_page_detail(start: Optional[int], end: Optional[int]) -> Optional[str]:
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None:
        return None
    if end is None or start == end:
        return f"p.{start}"
    return f"pp.{start}-{end}"


def _format_line_detail(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Allow formats like "123-126" or "123"
    match = re.fullmatch(r"(\d+)\s*[-\u2013]\s*(\d+)", text)
    if match:
        start, end = match.groups()
        return f"L{start}-L{end}"
    match = re.fullmatch(r"\d+", text)
    if match:
        return f"L{text}"
    # If already formatted (e.g., L10-L20), return as-is
    return text


def _clean_detail(detail: Optional[str]) -> Optional[str]:
    if not detail:
        return None
    d = str(detail).strip()
    if not d:
        return None
    return d


def _extract_from_metadata(meta: dict) -> Tuple[Optional[str], Optional[str]]:
    text = meta.get("text") or meta.get("chunk")
    src_from_text = None
    if isinstance(text, str):
        src_match = re.search(r"\[SRC=([^\]]+)\]", text)
        if src_match:
            src_from_text = src_match.group(1).strip()

    source_candidates = [
        src_from_text,
        meta.get("file"),
        meta.get("source_basename"),
        meta.get("source"),
        meta.get("document_key"),
    ]
    source = next((str(s).strip() for s in source_candidates if s), None)

    if not source and isinstance(text, str):
        doc_match = re.search(r"\[DOC=([^\]]+)\]", text)
        if doc_match:
            source = doc_match.group(1).strip()

    if not source and meta.get("publication_key"):
        source = str(meta["publication_key"]).strip()

    page_detail = _format_page_detail(
        _coerce_int(meta.get("page_start") or meta.get("page")),
        _coerce_int(meta.get("page_end")),
    )
    if not page_detail and isinstance(text, str):
        starts = [int(m.group(1)) for m in re.finditer(r"<page_start>(\d+)</page_start>", text)]
        ends = [int(m.group(1)) for m in re.finditer(r"<page_end>(\d+)</page_end>", text)]
        if starts or ends:
            start_page = starts[0] if starts else (ends[0] if ends else None)
            end_page = ends[-1] if ends else (starts[-1] if starts else None)
            page_detail = _format_page_detail(start_page, end_page)

    if page_detail:
        detail = page_detail
    else:
        detail = _format_line_detail(meta.get("lines") or meta.get("line_range"))

    return source, _clean_detail(detail)


def _extract_reference(item) -> Tuple[Optional[str], Optional[str]]:
    if item is None:
        return None, None

    if isinstance(item, Reference):
        return item.source, _clean_detail(item.detail)

    metadata = None
    if hasattr(item, "metadata"):
        metadata = getattr(item, "metadata") or {}
    elif isinstance(item, dict):
        metadata = item

    if metadata:
        src, detail = _extract_from_metadata(metadata)
        if src:
            return src, detail

    if hasattr(item, "text") and isinstance(item.text, str):
        return _extract_from_metadata({"text": item.text})

    return None, None


def build_references(items: Optional[Iterable]) -> List[Reference]:
    refs: List[Reference] = []
    seen = set()
    if not items:
        return refs

    for item in items:
        source, detail = _extract_reference(item)
        if not source:
            continue
        key = (source.strip(), detail.strip() if detail else None)
        if key in seen:
            continue
        seen.add(key)
        refs.append(Reference(source=key[0], detail=key[1]))
    return refs


def _format_reference(ref: Reference) -> str:
    if ref.detail:
        return f"{ref.source}. {ref.detail}."
    return f"{ref.source}."


def _append_citation_marker(answer: str, indices: Sequence[int]) -> str:
    if not indices:
        return answer
    marker = "[" + ",".join(str(i) for i in indices) + "]"
    stripped = answer.rstrip()
    if not stripped:
        return marker

    if stripped[-1] in ".!?":
        return f"{stripped} {marker}"
    return f"{stripped} {marker}"


def format_with_citations(
    answer: str,
    references: Sequence[Reference],
    style: str = "vancouver",
    seq_offset: int = 0,
) -> str:
    if not references:
        return answer

    if style.lower() != "vancouver":
        raise ValueError("Only Vancouver style is supported.")

    refs = list(references)
    start_index = max(seq_offset, 0) + 1
    indices = list(range(start_index, start_index + len(refs)))
    answer_with_marker = _append_citation_marker(answer, indices)

    ref_lines = ["References"]
    for offset, ref in enumerate(refs):
        ref_lines.append(f"{start_index + offset}. {_format_reference(ref)}")

    return answer_with_marker + "\n\n" + "\n".join(ref_lines)

