#!/usr/bin/env python3
"""
NodeRAG cost estimator.

Loads the NodeRAG settings from `.env`, estimates token/chunk counts for the
Markdown files in `NODERAG_SOURCE_DIR`, and prints the expected OpenAI spend
for the configured LLM and embedding backend.  Hugging Face embeddings are
assumed to run locally (zero API cost).
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import tiktoken
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

LLM_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.0020, 0.0080),
    "gpt-4.1-mini": (0.00040, 0.0016),
    "gpt-4.1-nano": (0.00010, 0.0004),
    "gpt-5": (0.00125, 0.0100),
    "gpt-5-mini": (0.00025, 0.0020),
    "gpt-5-nano": (0.00005, 0.0004),
}

EMBED_PRICING: Dict[str, float] = {
    "text-embedding-3-large": 0.00013,
    "text-embedding-3-large-r": 0.00013,
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-small-r": 0.00002,
}


@dataclass
class Estimate:
    files: int
    total_tokens: int
    chunk_tokens: int
    chunk_count: int


def _load_env() -> None:
    load_dotenv(REPO_ROOT / ".env", override=True)
    load_dotenv(override=False)


def _resolve_source_dir(custom: str | None) -> Path:
    raw = custom or os.getenv("NODERAG_SOURCE_DIR", "KnowledgeGraph/source_data")
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Markdown source directory not found: {path}")
    return path


def _load_encoding(model_name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_name)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("o200k_base")


def _tokenize(text: str, encoding: tiktoken.Encoding) -> Iterable[int]:
    return encoding.encode(text)


def _estimate_chunks(
    token_count: int, chunk_size: int, chunk_overlap: int
) -> Tuple[int, int]:
    if token_count == 0:
        return 0, 0
    step = max(1, chunk_size - chunk_overlap)
    chunk_tokens = 0
    chunk_count = 0
    for start in range(0, token_count, step):
        length = min(chunk_size, token_count - start)
        chunk_tokens += length
        chunk_count += 1
    return chunk_tokens, chunk_count


def _gather_estimates(
    source_dir: Path,
    encoding: tiktoken.Encoding,
    chunk_size: int,
    chunk_overlap: int,
) -> Estimate:
    total_tokens = 0
    chunk_tokens = 0
    chunk_count = 0
    files_seen = 0

    for path in sorted(source_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        tokens = list(_tokenize(text, encoding))
        total_tokens += len(tokens)
        ct, cc = _estimate_chunks(len(tokens), chunk_size, chunk_overlap)
        chunk_tokens += ct
        chunk_count += cc
        files_seen += 1

    return Estimate(
        files=files_seen,
        total_tokens=total_tokens,
        chunk_tokens=chunk_tokens,
        chunk_count=chunk_count,
    )


def _llm_cost(llm_model: str, input_tokens: int, output_tokens: int) -> Tuple[float, float, float]:
    rate = LLM_PRICING.get(llm_model)
    if rate is None:
        raise ValueError(
            f"Unknown LLM model '{llm_model}'. Provide --llm-input-rate "
            "and --llm-output-rate to override pricing."
        )
    prompt_rate, completion_rate = rate
    prompt_cost = input_tokens * prompt_rate / 1000
    completion_cost = output_tokens * completion_rate / 1000
    return prompt_cost, completion_cost, prompt_cost + completion_cost


def _embed_cost(
    backend: str, embed_model: str, embed_tokens: int
) -> Tuple[str, float]:
    backend = backend.lower()
    if backend != "openai":
        return (backend, 0.0)
    rate = EMBED_PRICING.get(embed_model)
    if rate is None:
        raise ValueError(
            f"Unknown embedding model '{embed_model}'. Override with --embed-rate."
        )
    return (embed_model, embed_tokens * rate / 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Override NODERAG_SOURCE_DIR.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override NODERAG_LLM_MODEL / OPENAI_MODEL.",
    )
    parser.add_argument(
        "--embed-backend",
        default=None,
        help="Override NODERAG_EMBED_BACKEND.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Override NODERAG_OPENAI_EMBED_MODEL.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override NODERAG_CHUNK_TARGET_TOKENS.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override NODERAG_CHUNK_OVERLAP_TOKENS.",
    )
    parser.add_argument(
        "--prompt-overhead",
        type=float,
        default=None,
        help="Override NODERAG prompt overhead tokens per chunk.",
    )
    parser.add_argument(
        "--avg-output",
        type=float,
        default=None,
        help="Override average output tokens per chunk.",
    )
    parser.add_argument(
        "--embedding-expansion",
        type=float,
        default=2.5,
        help="Multiplier applied to chunk tokens to approximate extra embedding volume.",
    )
    parser.add_argument(
        "--llm-input-rate",
        type=float,
        default=None,
        help="Override LLM input cost (USD per 1K tokens).",
    )
    parser.add_argument(
        "--llm-output-rate",
        type=float,
        default=None,
        help="Override LLM output cost (USD per 1K tokens).",
    )
    parser.add_argument(
        "--embed-rate",
        type=float,
        default=None,
        help="Override OpenAI embedding cost (USD per 1K tokens).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_env()

    source_dir = _resolve_source_dir(args.source_dir)
    llm_model = args.llm_model or os.getenv("NODERAG_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    embed_backend = args.embed_backend or os.getenv("NODERAG_EMBED_BACKEND", "hf")
    embed_model = args.embed_model or os.getenv("NODERAG_OPENAI_EMBED_MODEL") or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chunk_size = args.chunk_size or int(os.getenv("NODERAG_CHUNK_TARGET_TOKENS", "700"))
    chunk_overlap = args.chunk_overlap or int(os.getenv("NODERAG_CHUNK_OVERLAP_TOKENS", "50"))
    prompt_overhead = args.prompt_overhead or float(os.getenv("NODERAG_PROMPT_OVERHEAD", "200"))
    avg_output = args.avg_output or float(os.getenv("NODERAG_AVG_OUTPUT_TOKENS", "150"))
    embed_expansion = args.embedding_expansion

    encoding = _load_encoding(llm_model)
    estimate = _gather_estimates(source_dir, encoding, chunk_size, chunk_overlap)

    llm_prompt_tokens = estimate.chunk_tokens + int(prompt_overhead * estimate.chunk_count)
    llm_completion_tokens = int(avg_output * estimate.chunk_count)

    if args.llm_input_rate is not None and args.llm_output_rate is not None:
        prompt_cost = llm_prompt_tokens * args.llm_input_rate / 1000
        completion_cost = llm_completion_tokens * args.llm_output_rate / 1000
        llm_total = prompt_cost + completion_cost
    else:
        prompt_cost, completion_cost, llm_total = _llm_cost(
            llm_model, llm_prompt_tokens, llm_completion_tokens
        )

    embed_tokens = int(math.ceil(estimate.chunk_tokens * embed_expansion))
    if args.embed_rate is not None:
        embed_cost_model = embed_model
        embed_cost_value = embed_tokens * args.embed_rate / 1000
    else:
        embed_cost_model, embed_cost_value = _embed_cost(
            embed_backend, embed_model, embed_tokens
        )

    grand_total = llm_total + embed_cost_value

    print(f"Source directory      : {source_dir}")
    print(f"Markdown files        : {estimate.files}")
    print(f"Chunk size / overlap  : {chunk_size} / {chunk_overlap}")
    print(f"Chunks (estimated)    : {estimate.chunk_count:,}")
    print(f"Chunk tokens          : {estimate.chunk_tokens:,}")
    print(f"Embedding tokens (~)  : {embed_tokens:,}")
    print()
    print(f"LLM cost ({llm_model}):")
    print(f"  Prompt tokens       : {llm_prompt_tokens:,}")
    print(f"  Completion tokens   : {llm_completion_tokens:,}")
    print(f"  Prompt cost         : USD ${prompt_cost:,.2f}")
    print(f"  Completion cost     : USD ${completion_cost:,.2f}")
    print(f"  Total LLM cost      : USD ${llm_total:,.2f}")
    print()
    if embed_cost_value == 0:
        label = "HuggingFace / local" if embed_backend.lower() != "openai" else embed_cost_model
        print(f"Embedding cost ({label}):")
        print("  API spend           : USD $0.00")
    else:
        print(f"Embedding cost ({embed_cost_model}):")
        print(f"  API spend           : USD ${embed_cost_value:,.2f}")
    print()
    print(f"Grand total (LLM + embeddings): USD ${grand_total:,.2f}")


if __name__ == "__main__":
    main()