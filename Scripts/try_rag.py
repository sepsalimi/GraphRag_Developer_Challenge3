#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Callable

from KnowledgeGraph.LightRAG.lightrag_runner import (
    DEFAULT_SOURCE_DIR as LIGHTRAG_SOURCE_DIR,
    aquery as lightrag_aquery,
    build_lightrag,
    ingest_markdown_files as ingest_lightrag_docs,
)
from KnowledgeGraph.LightRAG.raganything_runner import (
    DEFAULT_WORKING_DIR as RAGANYTHING_WORKING_DIR,
    aquery as raganything_aquery,
    build_raganything,
    ingest_markdown_files as ingest_raganything_docs,
)


async def _maybe_ingest(
    backend: str,
    *,
    source_dir: Path,
    split_char: str | None,
    split_only: bool,
    skip_ingest: bool,
):
    if backend == "lightrag":
        rag = build_lightrag()
        if not skip_ingest:
            await ingest_lightrag_docs(
                rag,
                source_dir=source_dir,
                split_by_character=split_char,
                split_by_character_only=split_only,
            )
        return rag, lightrag_aquery

    rag = build_raganything(working_dir=RAGANYTHING_WORKING_DIR)
    if not skip_ingest:
        await ingest_raganything_docs(
            rag,
            source_dir=source_dir,
            split_by_character=split_char,
            split_by_character_only=split_only,
        )
    return rag, raganything_aquery


async def _run_async(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir or LIGHTRAG_SOURCE_DIR)
    split_char = args.split_character

    rag, query_fn = await _maybe_ingest(
        args.backend,
        source_dir=source_dir,
        split_char=split_char,
        split_only=args.split_only,
        skip_ingest=args.skip_ingest,
    )

    answer = await query_fn(
        rag,
        args.question,
        mode=args.mode,
    )
    print(answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick CLI to compare LightRAG and RAG-Anything backends."
    )
    parser.add_argument(
        "--backend",
        choices=("lightrag", "raganything"),
        default="lightrag",
        help="Backend to use.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask the selected backend.",
    )
    parser.add_argument(
        "--mode",
        default="mix",
        help="Retrieval mode (local, global, hybrid, naive, mix, bypass).",
    )
    parser.add_argument(
        "--source-dir",
        help="Directory containing Markdown files (defaults to KnowledgeGraph/source_data).",
    )
    parser.add_argument(
        "--split-character",
        default="\n\n",
        help="Optional character to split documents on before chunking (default: blank line).",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="If set, disable token-based chunking and rely solely on the split character.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion step (useful if data already processed).",
    )
    return parser


def main(cli_args: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()

