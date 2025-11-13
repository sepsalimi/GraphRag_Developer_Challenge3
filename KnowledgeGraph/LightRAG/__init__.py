"""
Helpers for LightRAG ingestion and querying built on top of the core library.
"""

from .lightrag_runner import (
    DEFAULT_SOURCE_DIR as LIGHTRAG_DEFAULT_SOURCE_DIR,
    DEFAULT_WORKING_DIR as LIGHTRAG_DEFAULT_WORKING_DIR,
    aquery as lightrag_aquery,
    build_lightrag,
    ensure_initialized,
    ingest_markdown_files as ingest_lightrag_markdown,
)
from .raganything_runner import (
    DEFAULT_WORKING_DIR as RAGANYTHING_DEFAULT_WORKING_DIR,
    MarkdownOnlyRAGAnything,
    aquery as raganything_aquery,
    build_raganything,
    ingest_markdown_files as ingest_raganything_markdown,
)

__all__ = [
    "LIGHTRAG_DEFAULT_SOURCE_DIR",
    "LIGHTRAG_DEFAULT_WORKING_DIR",
    "RAGANYTHING_DEFAULT_WORKING_DIR",
    "build_lightrag",
    "ensure_initialized",
    "ingest_lightrag_markdown",
    "lightrag_aquery",
    "build_raganything",
    "ingest_raganything_markdown",
    "raganything_aquery",
    "MarkdownOnlyRAGAnything",
]
