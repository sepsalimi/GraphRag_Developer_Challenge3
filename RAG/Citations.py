from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Reference:
    source: str
    detail: Optional[str] = None


def _clean_detail(detail: Optional[str]) -> Optional[str]:
    if not detail:
        return None
    d = str(detail).strip()
    if not d:
        return None
    if not d.lower().startswith("p.") and d[0].isdigit():
        return f"p.{d}"
    return d


def _extract_from_metadata(meta: dict) -> Tuple[Optional[str], Optional[str]]:
    source = meta.get("source") or meta.get("document_key") or meta.get("file")
    detail = meta.get("lines") or meta.get("line_range")

    text = meta.get("text") or meta.get("chunk")
    if not source and isinstance(text, str):
        src_match = re.search(r"\[SRC=([^\]]+)\]", text)
        if src_match:
            source = src_match.group(1)
        else:
            doc_match = re.search(r"\[DOC=([^\]]+)\]", text)
            if doc_match:
                source = doc_match.group(1)

    if not source and meta.get("publication_key"):
        source = str(meta["publication_key"])

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


def _append_citation_marker(answer: str, count: int) -> str:
    if count <= 0:
        return answer
    marker = "[" + ",".join(str(i) for i in range(1, count + 1)) + "]"
    stripped = answer.rstrip()
    if not stripped:
        return marker

    if stripped[-1] in ".!?":
        return f"{stripped} {marker}"
    return f"{stripped} {marker}"


def format_with_citations(answer: str, references: Sequence[Reference], style: str = "vancouver") -> str:
    if not references:
        return answer

    if style.lower() != "vancouver":
        raise ValueError("Only Vancouver style is supported.")

    refs = list(references)
    answer_with_marker = _append_citation_marker(answer, len(refs))

    ref_lines = ["References"]
    for idx, ref in enumerate(refs, start=1):
        ref_lines.append(f"{idx}. {_format_reference(ref)}")

    return answer_with_marker + "\n\n" + "\n".join(ref_lines)

