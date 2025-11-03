"""Utilities for markdown-aware, table-smart chunking.

This module centralises chunk creation so that both vector and hybrid RAG
pipelines benefit from consistent preprocessing.  It purposely fails fast:
missing metadata or malformed tables raise ``ValueError`` instead of silently
falling back to naïve splitting.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    stripped = stripped.strip("| ")
    return bool(stripped) and all(ch in "-: " for ch in stripped)


def _normalize_for_matching(text: str) -> str:
    """Return a normalised variant used only to help retrieval."""

    collapsed = text.translate(_ARABIC_DIGITS)
    collapsed = re.sub(r"\s*/\s*", "/", collapsed)
    collapsed = re.sub(r"\s*-\s*", "-", collapsed)
    collapsed = re.sub(r"\s+", " ", collapsed)
    return collapsed.strip()


@dataclass
class ChunkingConfig:
    """Configuration controlling token sizes and detection heuristics."""

    token_chunk_size: int = 700
    token_chunk_overlap: int = 100
    slot_cue_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"(?i)(kd|k\.d\.|دينار|percent|%)|\b(day|deadline|closing|time)\b"
        )
    )
    anchor_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"(?i)(?:A\s*/\s*M\s*/\s*\d+|A/M/\d+|5D[A-Z0-9]+|RFP[\s/-]?\d+|\d{6,7}[\s/-]?RFP|20\d{2}\s*/\s*00\d{2,3}|MEETING\s+NO\.?\s*\d{4}/\d+)"
        )
    )
    time_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b", re.IGNORECASE)
    )


def chunk_json_file(path: str | Path, cfg: Optional[ChunkingConfig] = None) -> List[Dict[str, Any]]:
    """Split a JSON file into token-aware chunks.

    The JSON file must contain a single mapping from keys to textual values.
    """

    import json

    cfg = cfg or ChunkingConfig()
    splitter = TokenTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=cfg.token_chunk_size,
        chunk_overlap=cfg.token_chunk_overlap,
    )

    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {p}")

    chunks: List[Dict[str, Any]] = []
    for key, value in data.items():
        text = str(value).strip()
        if not text:
            continue
        for piece in splitter.split_text(text):
            chunks.append(
                {
                    "chunk_id": f"{p.stem}-{key}-chunk{uuid.uuid4().hex[:8]}",
                    "Source": key,
                    "source": key,
                    "text": piece,
                }
            )
    return chunks


def chunk_markdown_file(
    path: str | Path,
    publication_meta: Dict[str, Any],
    document_meta: Dict[str, Any],
    cfg: Optional[ChunkingConfig] = None,
) -> List[Dict[str, Any]]:
    """Read *path* and chunk its markdown contents.

    This convenience wrapper simply loads the markdown file before delegating to
    :func:`chunk_document_text`.
    """

    text = Path(path).read_text(encoding="utf-8")
    return chunk_document_text(text, publication_meta, document_meta, cfg=cfg)


def chunk_document_text(
    text: str,
    publication_meta: Dict[str, Any],
    document_meta: Dict[str, Any],
    cfg: Optional[ChunkingConfig] = None,
) -> List[Dict[str, Any]]:
    """Chunk a single markdown document body.

    Parameters
    ----------
    text:
        Raw markdown content of the document.
    publication_meta:
        Must include ``publication_key``, ``issue_number``, ``supplement_index``,
        ``publication_date`` and ``source_basename``.
    document_meta:
        Must include ``document_key`` and ``doc_type``.  Optional fields such as
        ``doc_number`` or ``section_heading`` are incorporated when present.
    """

    required_pub_keys = [
        "publication_key",
        "issue_number",
        "supplement_index",
        "publication_date",
        "source_basename",
    ]
    for key in required_pub_keys:
        if key not in publication_meta:
            raise ValueError(f"Missing publication metadata: {key}")

    required_doc_keys = ["document_key", "doc_type"]
    for key in required_doc_keys:
        if key not in document_meta:
            raise ValueError(f"Missing document metadata: {key}")

    cfg = cfg or ChunkingConfig()
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
    )
    token_splitter = TokenTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=cfg.token_chunk_size,
        chunk_overlap=cfg.token_chunk_overlap,
    )

    sections = header_splitter.split_text(text)
    chunks: List[Dict[str, Any]] = []

    for section in sections:
        section_text = section.page_content.strip()
        if not section_text:
            continue
        section_title = _compose_section_title(section.metadata)
        header_prefix = _build_header_prefix(publication_meta, document_meta, section_title)

        for segment_type, payload in _segment_markdown(section_text):
            if segment_type == "table":
                table_lines, caption, context_lines = payload
                if _needs_row_chunking(table_lines, cfg):
                    chunks.extend(
                        _row_chunks(
                            header_prefix,
                            table_lines,
                            caption,
                            context_lines,
                            cfg,
                            publication_meta,
                            document_meta,
                        )
                    )
                else:
                    table_text = "\n".join([line for line in table_lines if line.strip()])
                    if caption:
                        table_text = f"{caption}\n{table_text}"
                    for piece in token_splitter.split_text(table_text):
                        chunks.append(
                            _make_chunk(
                                header_prefix,
                                piece,
                                publication_meta,
                                document_meta,
                            )
                        )
            else:  # narrative text
                narrative = payload.strip()
                if not narrative:
                    continue
                for piece in token_splitter.split_text(narrative):
                    chunks.append(
                        _make_chunk(
                            header_prefix,
                            piece,
                            publication_meta,
                            document_meta,
                        )
                    )

    return chunks


def _compose_section_title(metadata: Dict[str, Any]) -> str:
    headers = [metadata.get(key) for key in ("Header 1", "Header 2", "Header 3", "Header 4")]
    headers = [header for header in headers if header]
    return " / ".join(headers)


def _segment_markdown(section_text: str) -> Iterable[Tuple[str, Any]]:
    """Yield (segment_type, payload) tuples for a markdown section."""

    lines = section_text.splitlines()
    buffer: List[str] = []
    recent_context: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.lower() in {"announcement", "notice", "notification"}:
            recent_context = [stripped]

        if _is_table_row(line):
            if buffer:
                text_block = "\n".join(buffer).strip()
                if text_block:
                    yield ("text", text_block)
                buffer.clear()

            caption = None
            context_lines = list(recent_context)
            # capture caption immediately preceding table
            if i > 0 and lines[i - 1].strip().startswith("<table_caption>"):
                caption = lines[i - 1].strip()

            table_block: List[str] = []
            while i < len(lines) and (
                _is_table_row(lines[i])
                or lines[i].strip().startswith("<table_caption>")
                or lines[i].strip() in {"<header>", "</header>", "<footer>", "</footer>"}
                or lines[i].strip().startswith("<page_end>")
            ):
                table_block.append(lines[i])
                i += 1

            yield ("table", (table_block, caption, context_lines))
            continue

        buffer.append(line)
        i += 1

    if buffer:
        text_block = "\n".join(buffer).strip()
        if text_block:
            yield ("text", text_block)


def _needs_row_chunking(table_lines: Sequence[str], cfg: ChunkingConfig) -> bool:
    """Return ``True`` when a table row carries slot cues or anchors."""

    header_seen = False
    has_data_row = False
    cues_detected = False
    for line in table_lines:
        if _is_table_row(line):
            normalized = _normalize_for_matching(line)
            if not header_seen:
                header_seen = True
                if cfg.slot_cue_pattern.search(normalized) or cfg.anchor_pattern.search(normalized) or cfg.time_pattern.search(normalized):
                    cues_detected = True
                continue
            if _is_table_separator(line):
                continue
            has_data_row = True
            if cfg.slot_cue_pattern.search(normalized) or cfg.anchor_pattern.search(normalized) or cfg.time_pattern.search(normalized):
                cues_detected = True
    return cues_detected and has_data_row


def _row_chunks(
    header_prefix: str,
    table_lines: Sequence[str],
    caption: Optional[str],
    context_lines: Sequence[str],
    cfg: ChunkingConfig,
    publication_meta: Dict[str, Any],
    document_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return chunk dictionaries for each data row in ``table_lines``."""

    header_line: Optional[str] = None
    alignment_line: Optional[str] = None
    data_rows: List[str] = []

    for line in table_lines:
        if not _is_table_row(line):
            continue
        if header_line is None:
            header_line = line.strip()
            continue
        if alignment_line is None and _is_table_separator(line):
            alignment_line = line.strip()
            continue
        data_rows.append(line.strip())

    if header_line is None or not data_rows:
        raise ValueError("Malformed markdown table encountered; missing header or data rows")

    header_lines = [header_line]
    if alignment_line:
        header_lines.append(alignment_line)

    chunks: List[Dict[str, Any]] = []
    for row in data_rows:
        normalized = _normalize_for_matching(row)
        lines = [header_prefix]
        if context_lines:
            lines.extend(f"[CONTEXT] {ctx}" for ctx in context_lines if ctx)
        if caption:
            lines.append(f"[CAPTION] {caption}")
        lines.extend(header_lines)
        lines.append(row)
        if normalized and normalized != row:
            lines.append(f"[NORMALIZED] {normalized}")
        body = "\n".join(lines[1:])
        chunks.append(_make_chunk(header_prefix, body, publication_meta, document_meta))
    return chunks


def _make_chunk(
    header_prefix: str,
    body_text: str,
    publication_meta: Dict[str, Any],
    document_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Compose the final chunk dictionary used for ingestion."""

    lines = [header_prefix]
    if body_text.strip():
        lines.append(body_text.strip())
    chunk_body = "\n".join(lines)
    return {
        "chunk_id": str(uuid.uuid4()),
        "publication_key": publication_meta["publication_key"],
        "document_key": document_meta["document_key"],
        "article_number": document_meta.get("article_number"),
        "text": chunk_body,
    }


def _build_header_prefix(
    publication_meta: Dict[str, Any],
    document_meta: Dict[str, Any],
    section_title: str,
) -> str:
    """Format provenance metadata into the stable header prefix."""

    parts = [
        f"[PUB={publication_meta['publication_key']}]",
        f"[ISSUE={publication_meta['issue_number']}]",
        f"[SUPP={publication_meta['supplement_index']}]",
        f"[DATE={publication_meta['publication_date']}]",
        f"[SRC={publication_meta['source_basename']}]",
    ]

    doc_bits = [f"[DOC={document_meta['doc_type']} {document_meta.get('doc_number') or ''}]".strip()]
    if document_meta.get("section_heading"):
        doc_bits.append(f"[SECTION={document_meta['section_heading']}]")
    if section_title:
        doc_bits.append(f"[ARTICLE={section_title}]")
    return " ".join(parts + doc_bits)


