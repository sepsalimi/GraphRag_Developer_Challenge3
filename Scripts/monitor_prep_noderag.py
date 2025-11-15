#!/usr/bin/env python3
"""
Monitor NodeRAG pipeline progress in real-time.

Shows:
- Recent log entries
- Cache file sizes
- Neo4j node counts (optional)
- Pipeline state
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from KnowledgeGraph.config import load_neo4j_graph
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_cache_files() -> dict:
    """Get cache file sizes."""
    cache_dir = PROJECT_ROOT / ".noderag" / "cache"
    if not cache_dir.exists():
        return {}
    
    files = {}
    for ext in [".parquet", ".jsonl"]:
        for path in cache_dir.glob(f"*{ext}"):
            size = path.stat().st_size
            files[path.name] = size
    return files


def get_log_tail(log_path: Path, lines: int = 10) -> list[str]:
    """Get last N lines from log file."""
    if not log_path.exists():
        return []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            return [line.rstrip() for line in all_lines[-lines:]]
    except Exception:
        return []


def get_pipeline_state() -> Optional[dict]:
    """Get current pipeline state from state.json."""
    state_path = PROJECT_ROOT / ".noderag" / "info" / "state.json"
    if not state_path.exists():
        return None
    
    try:
        import json
        with open(state_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def get_neo4j_counts(graph) -> dict:
    """Get node counts from Neo4j."""
    labels = ["NR_Document", "NR_Passage", "NR_Entity", "NR_Relationship"]
    counts = {}
    for label in labels:
        try:
            result = graph.query(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = int(result[0]["c"]) if result else 0
        except Exception:
            counts[label] = None
    return counts


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """Print monitor header."""
    print("=" * 70)
    print("NodeRAG Pipeline Monitor")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()


def print_section(title: str):
    """Print section header."""
    print(f"\n--- {title} ---")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor NodeRAG pipeline progress in real-time."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Update interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Disable Neo4j count monitoring",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=10,
        help="Number of log lines to show (default: 10)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear screen between updates",
    )
    args = parser.parse_args()

    log_path = PROJECT_ROOT / ".noderag" / "info" / "info.log"
    
    graph = None
    if NEO4J_AVAILABLE and not args.no_neo4j:
        try:
            graph, *_ = load_neo4j_graph()
            print("Connected to Neo4j.")
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            print("Continuing without Neo4j monitoring...")
            graph = None

    try:
        while True:
            if not args.no_clear:
                clear_screen()
            
            print_header()
            
            # Pipeline state
            state = get_pipeline_state()
            if state:
                print_section("Pipeline State")
                for key, value in state.items():
                    print(f"  {key}: {value}")
            
            # Cache files
            cache_files = get_cache_files()
            if cache_files:
                print_section("Cache Files")
                total_size = 0
                for name, size in sorted(cache_files.items()):
                    total_size += size
                    print(f"  {name:40s} {format_size(size):>10s}")
                print(f"  {'Total':40s} {format_size(total_size):>10s}")
            else:
                print_section("Cache Files")
                print("  No cache files found yet")
            
            # Neo4j counts
            if graph:
                print_section("Neo4j Node Counts")
                counts = get_neo4j_counts(graph)
                for label, count in counts.items():
                    if count is not None:
                        print(f"  {label:25s} {count:>10,}")
                    else:
                        print(f"  {label:25s} {'Error':>10s}")
            
            # Log tail
            log_lines = get_log_tail(log_path, args.log_lines)
            if log_lines:
                print_section(f"Recent Log Entries (last {len(log_lines)} lines)")
                for line in log_lines:
                    # Truncate very long lines
                    if len(line) > 100:
                        line = line[:97] + "..."
                    print(f"  {line}")
            else:
                print_section("Log")
                print("  No log entries yet (log file may not exist)")
            
            print()
            print("=" * 70)
            print(f"Next update in {args.interval} seconds... (Ctrl+C to stop)")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()


