"""Utilities for working with Text2Cypher retriever output."""

from __future__ import annotations

from neo4j_graphrag.retrievers.text2cypher import extract_cypher


def extract_cypher_query(response_text: str) -> str:
    """Extract the Cypher statement from a Text2Cypher response.

    Raises
    ------
    ValueError
        If the response does not contain a valid Cypher statement.
    """

    statement = (extract_cypher(response_text) or "").strip()
    if not statement:
        raise ValueError(
            f"Unable to extract Cypher statement from response: {response_text!r}"
        )
    return statement


__all__ = ["extract_cypher_query"]

