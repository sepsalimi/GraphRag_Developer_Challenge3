#!/usr/bin/env python3
"""Inspect the cached NodeRAG graph without touching Neo4j."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KnowledgeGraph.NodeRag import (  # noqa: E402
    get_cache_dir,
    load_cached_outputs,
    print_graph_overview,
    summarize_outputs,
)


def main() -> None:
    outputs = load_cached_outputs()
    stats = summarize_outputs(outputs)
    print("NodeRAG cache summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"Cache directory: {get_cache_dir()}")
    print_graph_overview(outputs.graph)


if __name__ == "__main__":
    main()

