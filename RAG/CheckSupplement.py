"""Compatibility shim. Prefer :mod:`RAG.CheckUpdates`."""

from __future__ import annotations

from .CheckUpdates import gather_update_snippets as gather_supplement_snippets

__all__ = ["gather_supplement_snippets"]

