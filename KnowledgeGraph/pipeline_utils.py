"""Helpers for constructing Neo4j SimpleKGPipeline with schema enforcement."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

try:
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("neo4j_graphrag experimental pipeline is required") from exc


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def build_simple_pipeline(**kwargs: Any) -> SimpleKGPipeline:
    """Construct SimpleKGPipeline with enforce_schema toggle.

    Respects the KG_ENFORCE_SCHEMA environment variable (defaults to True).
    """

    project_root = os.getenv("GRAPH_RAG_PROJECT_ROOT")
    if project_root:
        load_dotenv(os.path.join(project_root, ".env"), override=False)

    enforce = _env_bool("KG_ENFORCE_SCHEMA", True)
    pipeline = SimpleKGPipeline(enforce_schema=enforce, **kwargs)
    return pipeline


__all__ = ["build_simple_pipeline"]

