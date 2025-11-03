"""Small helpers for Lucene-safe fulltext queries."""

from __future__ import annotations

import os

_LUCENE_SPECIAL = set("+ - && || ! ( ) { } [ ] ^ \" ~ * ? : \\ /".split())


def _escape_token(token: str) -> str:
    out = []
    for ch in token:
        if ch in "\\+-!(){}[]^\"~*?:/" or ch == '|':
            out.append('\\' + ch)
        else:
            out.append(ch)
    return ''.join(out)


def escape_fulltext_query(text: str) -> str:
    if not text:
        return ""
    if os.getenv("LUCENE_ESCAPE_ENABLED", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return text
    parts = text.split()
    escaped = [_escape_token(p) for p in parts]
    return " ".join(escaped)

__all__ = ["escape_fulltext_query"]
