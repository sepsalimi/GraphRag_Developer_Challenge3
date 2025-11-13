from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from KnowledgeGraph.LightRAG.lightrag_runner import (
    DEFAULT_SOURCE_DIR,
    build_lightrag,
    ensure_initialized as ensure_lightrag_initialized,
)

VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "RAG-Anything"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from raganything import RAGAnything, RAGAnythingConfig  # type: ignore  # noqa: E402


DEFAULT_WORKING_DIR = Path(".rag/raganything")


class MarkdownOnlyRAGAnything(RAGAnything):
    """
    Minimal RAG-Anything variant that treats Markdown files as plain text.

    The original library expects MinerU/Docling parsers; this subclass bypasses
    those heavy dependencies for `.md` / `.txt` inputs.
    """

    SUPPORTED_TEXT_EXTS = {".md", ".txt"}

    def __post_init__(self):  # noqa: D401
        super().__post_init__()
        # Skip external parser installation checks when we only handle Markdown.
        self._parser_installation_checked = True
        self.doc_parser = None  # type: ignore[assignment]

    async def parse_document(  # type: ignore[override]
        self,
        file_path: str,
        output_dir: str | None = None,
        parse_method: str | None = None,
        display_stats: bool | None = None,
        **kwargs,
    ):
        path = Path(file_path)
        if path.suffix.lower() in self.SUPPORTED_TEXT_EXTS:
            text = path.read_text(encoding="utf-8").strip()
            content_list = [{"type": "text", "text": text}]
            doc_id = self._generate_content_based_doc_id(content_list)

            if display_stats:
                self.logger.info(
                    "Parsed Markdown file %s (length: %s characters)",
                    path.name,
                    len(text),
                )
            return content_list, doc_id

        return await super().parse_document(
            file_path=file_path,
            output_dir=output_dir,
            parse_method=parse_method,
            display_stats=display_stats,
            **kwargs,
        )


def _load_env_once() -> None:
    load_dotenv(override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required. Please add it to your .env file."
        )


def build_raganything(
    *,
    working_dir: Path | str = DEFAULT_WORKING_DIR,
) -> MarkdownOnlyRAGAnything:
    """
    Construct the Markdown-only RAG-Anything helper.

    LightRAG is reused as the underlying storage so the ingestion path stays
    consistent with the official project while avoiding heavy parser deps.
    """

    _load_env_once()

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    lightrag_dir = working_dir / "lightrag"
    lightrag = build_lightrag(working_dir=lightrag_dir)

    config = RAGAnythingConfig(
        working_dir=str(working_dir),
        parse_method="txt",
        parser="mineru",
        display_content_stats=False,
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
        max_concurrent_files=1,
        supported_file_extensions=[".md", ".txt"],
        recursive_folder_processing=False,
    )

    return MarkdownOnlyRAGAnything(
        lightrag=lightrag,
        config=config,
    )


def _iter_markdown_files(source_dir: Path) -> Iterable[Path]:
    return sorted(p for p in source_dir.glob("*.md") if p.is_file())


async def ingest_markdown_files(
    rag: MarkdownOnlyRAGAnything,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    *,
    split_by_character: str | None = "\n\n",
    split_by_character_only: bool = False,
) -> list[Path]:
    """
    Ingest Markdown documents through the RAG-Anything pipeline.
    """

    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Markdown source directory not found: {source_dir}")

    await ensure_lightrag_initialized(rag.lightrag)  # type: ignore[arg-type]

    ingested: list[Path] = []
    for path in _iter_markdown_files(source_dir):
        await rag.process_document_complete(
            str(path),
            parse_method="txt",
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
        )
        ingested.append(path)

    return ingested


async def aquery(
    rag: MarkdownOnlyRAGAnything,
    question: str,
    *,
    mode: str = "mix",
    **query_kwargs,
) -> str:
    """
    Execute an asynchronous query via RAG-Anything (LightRAG under the hood).
    """

    result = await rag.aquery(question, mode=mode, **query_kwargs)
    if isinstance(result, str):
        return result

    chunks = []
    async for chunk in result:
        chunks.append(chunk)
    return "".join(chunks)


def ingest_markdown_files_sync(
    rag: MarkdownOnlyRAGAnything,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    *,
    split_by_character: str | None = "\n\n",
    split_by_character_only: bool = False,
) -> list[Path]:
    return asyncio.run(
        ingest_markdown_files(
            rag,
            source_dir,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
        )
    )


def query(
    rag: MarkdownOnlyRAGAnything,
    question: str,
    *,
    mode: str = "mix",
    **query_kwargs,
) -> str:
    return asyncio.run(aquery(rag, question, mode=mode, **query_kwargs))


__all__ = [
    "DEFAULT_WORKING_DIR",
    "MarkdownOnlyRAGAnything",
    "build_raganything",
    "ingest_markdown_files",
    "ingest_markdown_files_sync",
    "aquery",
    "query",
]

