#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KnowledgeGraph.NodeRag import (  # noqa: E402
    get_cache_dir,
    print_graph_overview,
    run_noderag,
    summarize_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NodeRAG pipeline without Neo4j.")
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Delete .noderag/cache and .noderag/info before running.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Reuse the existing .noderag/input directory instead of copying from source_data.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable NodeRAG's rich progress output (useful inside notebooks).",
    )
    parser.add_argument(
        "--graph-stats",
        action="store_true",
        help="Print a quick node/edge breakdown of the cached graph after the run.",
    )
    parser.set_defaults(show_progress=True)
    args = parser.parse_args()

    outputs = run_noderag(
        show_progress=args.show_progress,
        reset_cache=args.reset_cache,
        resync_input=not args.skip_sync,
    )
    stats = summarize_outputs(outputs)
    print("NodeRAG cache summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"Cache directory: {get_cache_dir()}")

    if args.graph_stats:
        print_graph_overview(outputs.graph)


if __name__ == "__main__":
    main()