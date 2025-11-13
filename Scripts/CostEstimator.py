"""
Quick cost estimator for upcoming LightRAG ingestions.

The script loads the current LightRAG configuration, pre-chunks every
Markdown file in the configured source directory, and estimates the token
volumes and dollar cost for both the LLM passes and OpenAI embedding calls.

Example:
    poetry run python Scripts/CostEstimator.py \
        --expansion-factor 3.0 \
        --avg-output 180
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import tiktoken
from dotenv import load_dotenv

# Ensure we can import the project modules when invoked via `poetry run`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from KnowledgeGraph.LightRAG.lightrag_runner import ChunkingConfig, prechunk_markdown

# --- Pricing -----------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    llm_input_per_token: float
    llm_output_per_token: float
    embedding_per_token: Mapping[str, float]


PRICING = Pricing(
    llm_input_per_token=0.25 / 1_000_000,  # gpt-5-mini input
    llm_output_per_token=2.00 / 1_000_000,  # gpt-5-mini output
    embedding_per_token={
        "text-embedding-3-large": 0.13 / 1_000,
        "text-embedding-3-small": 0.02 / 1_000,
        "text-embedding-3-small-r": 0.02 / 1_000,
        "text-embedding-3-large-r": 0.13 / 1_000,
    },
)


# --- Helpers ------------------------------------------------------------------


def _resolve_source_dir(custom: str | None) -> Path:
    if custom:
        path = Path(custom)
    else:
        path = Path(os.getenv("LIGHTRAG_SOURCE_DIR", "KnowledgeGraph/source_data"))
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Markdown source directory not found: {path}")
    return path


def _load_encoding(llm_model: str | None) -> tiktoken.Encoding:
    model = llm_model or "gpt-5-mini"
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("o200k_base")


def _load_env() -> None:
    load_dotenv(REPO_ROOT / ".env", override=True)
    load_dotenv(override=False)


# --- Main logic ---------------------------------------------------------------


def estimate_tokens(
    source_dir: Path,
    encoding: tiktoken.Encoding,
    chunk_cfg: ChunkingConfig,
) -> tuple[int, int, int]:
    """Return (chunk_tokens, num_chunks, files_seen) for Markdown files."""
    chunk_tokens = 0
    num_chunks = 0
    files_seen = 0

    for path in sorted(source_dir.glob("*.md")):
        files_seen += 1
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        chunks = prechunk_markdown(text, chunk_cfg)
        num_chunks += len(chunks)
        for chunk in chunks:
            chunk_tokens += len(encoding.encode(chunk))

    return chunk_tokens, num_chunks, files_seen


def compute_costs(
    *,
    chunk_tokens: int,
    num_chunks: int,
    pricing: Pricing,
    expansion_factor: float,
    prompt_overhead: float,
    avg_output: float,
    embed_model: str,
    include_all_embeddings: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    llm_input_tokens = chunk_tokens + int(prompt_overhead * num_chunks)
    llm_output_tokens = int(avg_output * num_chunks)

    llm_cost = {
        "input_tokens": llm_input_tokens,
        "output_tokens": llm_output_tokens,
        "input_dollars": llm_input_tokens * pricing.llm_input_per_token,
        "output_dollars": llm_output_tokens * pricing.llm_output_per_token,
    }

    embed_tokens = int(math.ceil(chunk_tokens * expansion_factor))

    embed_cost: dict[str, float] = {"tokens": embed_tokens}
    models_to_report = {embed_model}
    if include_all_embeddings:
        models_to_report.update(pricing.embedding_per_token.keys())

    for model in sorted(models_to_report):
        rate = pricing.embedding_per_token.get(model)
        if rate is None:
            continue
        embed_cost[model] = embed_tokens * rate

    return llm_cost, embed_cost


def format_currency(value: float) -> str:
    return f"USD ${value:,.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Override the Markdown source directory (defaults to LIGHTRAG_SOURCE_DIR or KnowledgeGraph/source_data).",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name used for encoding lookup (defaults to LIGHTRAG_LLM_MODEL or gpt-5-mini after loading .env).",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Embedding model assumed for costing (defaults to LIGHTRAG_EMBED_MODEL after loading .env).",
    )
    parser.add_argument(
        "--expansion-factor",
        type=float,
        default=2.5,
        help="Multiplier to approximate entity/relationship embedding volume (default: 2.5).",
    )
    parser.add_argument(
        "--prompt-overhead",
        type=float,
        default=200,
        help="Additional prompt tokens per chunk for metadata/template text (default: 200).",
    )
    parser.add_argument(
        "--avg-output",
        type=float,
        default=150,
        help="Average number of output tokens per chunk extraction (default: 150).",
    )
    parser.add_argument(
        "--include-all-embeddings",
        action="store_true",
        help="Show costs for every known embedding model, not just the configured one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _load_env()

    llm_model = args.llm_model or os.getenv("LIGHTRAG_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5-mini"
    embed_model = (
        args.embed_model
        or os.getenv("LIGHTRAG_EMBED_MODEL")
        or os.getenv("OPENAI_EMBED_MODEL")
        or "text-embedding-3-large"
    )

    source_dir = _resolve_source_dir(args.source_dir)
    chunk_cfg = ChunkingConfig.from_env()
    encoding = _load_encoding(llm_model)

    chunk_tokens, num_chunks, files_seen = estimate_tokens(source_dir, encoding, chunk_cfg)

    llm_cost, embed_cost = compute_costs(
        chunk_tokens=chunk_tokens,
        num_chunks=num_chunks,
        pricing=PRICING,
        expansion_factor=args.expansion_factor,
        prompt_overhead=args.prompt_overhead,
        avg_output=args.avg_output,
        embed_model=embed_model,
        include_all_embeddings=args.include_all_embeddings,
    )

    print(f"Source directory      : {source_dir}")
    print(f"Markdown files        : {files_seen}")
    print(f"Chunks (estimated)    : {num_chunks:,}")
    print(f"Chunk tokens          : {chunk_tokens:,}")
    print(f"Embedding tokens (~)  : {embed_cost['tokens']:,}")
    print()

    print(f"LLM cost ({llm_model}):")
    print(f"  Prompt tokens       : {llm_cost['input_tokens']:,}")
    print(f"  Completion tokens   : {llm_cost['output_tokens']:,}")
    print(f"  Prompt cost         : {format_currency(llm_cost['input_dollars'])}")
    print(f"  Completion cost     : {format_currency(llm_cost['output_dollars'])}")
    print(f"  Total LLM cost      : {format_currency(llm_cost['input_dollars'] + llm_cost['output_dollars'])}")
    print()

    print(f"Embedding cost (base model: {embed_model}):")
    for model, rate in embed_cost.items():
        if model == "tokens":
            continue
        print(f"  {model:<24} {format_currency(rate)}")

    default_embed_cost = embed_cost.get(embed_model)
    if default_embed_cost is not None:
        total = llm_cost["input_dollars"] + llm_cost["output_dollars"] + default_embed_cost
        print()
        print(f"Total estimated cost (LLM + {embed_model}): {format_currency(total)}")


if __name__ == "__main__":
    main()


