#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KnowledgeGraph.config import load_neo4j_graph
from KnowledgeGraph.NodeRag import ensure_vector_index, preprocess_and_ingest


def _count_label(graph, label: str) -> int:
    result = graph.query(f"MATCH (n:{label}) RETURN count(n) AS c")
    return int(result[0]["c"]) if result else 0


def _print_counts(graph) -> None:
    for label in ("NR_Document", "NR_Passage", "NR_Entity", "NR_Relationship"):
        count = _count_label(graph, label)
        print(f"{label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NodeRAG ingestion pipeline.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing NR_* nodes before ingesting (defaults to merge-only).",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable progress output from the NodeRAG pipeline.",
    )
    parser.set_defaults(show_progress=True)
    args = parser.parse_args()

    graph, *_ = load_neo4j_graph()
    print("Connected to Neo4j.")

    ensure_vector_index(graph=graph)
    stats = preprocess_and_ingest(
        graph=graph,
        show_progress=args.show_progress,
        reset_graph=args.reset,
    )
    print("Ingestion summary:", stats)

    _print_counts(graph)


if __name__ == "__main__":
    main()

