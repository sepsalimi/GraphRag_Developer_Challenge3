from __future__ import annotations

import asyncio
import os
import contextlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

from dotenv import load_dotenv
from tqdm.auto import tqdm

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, compute_mdhash_id


DEFAULT_WORKING_DIR = Path(".rag/lightrag")
DEFAULT_SOURCE_DIR = Path("KnowledgeGraph/source_data")
DEFAULT_PRECHUNK_SENTINEL = "\n\n<!--LIGHTRAG-CHUNK-->\n\n"


def _infer_embedding_dim(model: str) -> int:
    """
    Return the default embedding dimension for a given OpenAI embedding model.

    The OpenAI text-embedding-3 family currently exposes:
        * text-embedding-3-small  -> 1536 dimensions
        * text-embedding-3-large  -> 3072 dimensions
    """

    normalised = (model or "").lower()
    if "large" in normalised:
        return 3072
    if "small" in normalised:
        return 1536
    # Sensible fallback for future models if dimension env var is not supplied.
    return 1536


def _resolve_embedding_params() -> tuple[str, int]:
    """
    Determine embedding model + dimension from the environment.

    - `LIGHTRAG_EMBED_MODEL` selects the OpenAI embedding model (defaults to 3-large).
    - `LIGHTRAG_EMBED_DIM` explicitly sets the dimension; otherwise we infer from the model.
    """

    model = os.getenv("LIGHTRAG_EMBED_MODEL", "text-embedding-3-large")
    raw_dim = os.getenv("LIGHTRAG_EMBED_DIM")

    if raw_dim:
        try:
            dim = int(raw_dim)
        except ValueError:
            dim = _infer_embedding_dim(model)
    else:
        dim = _infer_embedding_dim(model)

    return model, dim


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class ChunkingConfig:
    enable: bool = True
    target_tokens: int = 450
    overlap_tokens: int = 40
    preserve_headers: bool = True
    table_row_group_size: int | None = None

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        return cls(
            enable=_env_bool("LIGHTRAG_CHUNK_ENABLE", True),
            target_tokens=_env_int("LIGHTRAG_CHUNK_TARGET_TOKENS", 450) or 450,
            overlap_tokens=_env_int("LIGHTRAG_CHUNK_OVERLAP_TOKENS", 40) or 40,
            preserve_headers=_env_bool("LIGHTRAG_CHUNK_PRESERVE_HEADERS", True),
            table_row_group_size=_env_int("LIGHTRAG_CHUNK_TABLE_ROW_GROUP", None),
        )


@dataclass
class FallbackConfig:
    escalate_on_error: bool = True
    error_threshold: int = 10
    high_quality_model: str = "gpt-5"

    @classmethod
    def from_env(cls) -> "FallbackConfig":
        return cls(
            escalate_on_error=_env_bool("LIGHTRAG_ESCALATE_ON_ERROR", True),
            error_threshold=_env_int("LIGHTRAG_ESCALATE_THRESHOLD", 10) or 10,
            high_quality_model=os.getenv("LIGHTRAG_ESCALATE_MODEL", "gpt-5"),
        )


@dataclass
class FallbackState:
    config: FallbackConfig
    rag: LightRAG
    error_count: int = 0
    escalated: bool = False

    def register_error(self) -> None:
        self.error_count += 1
        if (
            self.config.escalate_on_error
            and not self.escalated
            and self.error_count >= self.config.error_threshold
        ):
            api_key, base_url = _load_env_once()
            completion = _build_completion_func(
                api_key, base_url, self.config.high_quality_model
            )
            _apply_llm_model(self.rag, completion, self.config.high_quality_model)
            print(
                f"LightRAG fallback: escalating to {self.config.high_quality_model} "
                f"after {self.error_count} format errors"
            )
            self.escalated = True


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _render_with_headers(body: str, headers: Sequence[str], cfg: ChunkingConfig) -> str:
    if cfg.preserve_headers and headers:
        header_line = " / ".join(h for h in headers if h)
        if header_line:
            return f"{header_line}\n\n{body}"
    return body


def _split_table_block(
    block_lines: list[str],
    headers: Sequence[str],
    cfg: ChunkingConfig,
) -> list[str]:
    body_lines = [line.rstrip() for line in block_lines]
    if not body_lines:
        return []

    if not cfg.table_row_group_size or cfg.table_row_group_size <= 0:
        table_text = "\n".join(body_lines).strip()
        if not table_text:
            return []
        return [_render_with_headers(table_text, headers, cfg)]

    # Attempt to keep the first two lines as header (title + alignment)
    header_lines = body_lines[:2]
    if len(header_lines) < 2 or not re.search(r"-", header_lines[1]):
        header_lines = body_lines[:1]
    data_lines = body_lines[len(header_lines) :]
    if not data_lines:
        table_text = "\n".join(body_lines).strip()
        return [_render_with_headers(table_text, headers, cfg)]

    chunks: list[str] = []
    step = max(1, cfg.table_row_group_size)
    for start in range(0, len(data_lines), step):
        rows = data_lines[start : start + step]
        table_lines = header_lines + rows
        table_text = "\n".join(table_lines).strip()
        if table_text:
            chunks.append(_render_with_headers(table_text, headers, cfg))
    return chunks


def _split_prose_block(
    block_lines: list[str],
    headers: Sequence[str],
    cfg: ChunkingConfig,
) -> list[str]:
    text = "\n".join(block_lines).strip()
    if not text:
        return []
    target = max(1, cfg.target_tokens)
    overlap_tokens = max(0, cfg.overlap_tokens)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    overlap_stub = False

    def flush_chunk() -> None:
        nonlocal current, current_tokens, overlap_stub
        if not current:
            return
        chunk_body = "\n\n".join(current).strip()
        if not chunk_body:
            current = []
            current_tokens = 0
            overlap_stub = False
            return
        chunks.append(_render_with_headers(chunk_body, headers, cfg))
        if overlap_tokens > 0:
            words = chunk_body.split()
            overlap_count = min(overlap_tokens, len(words))
            overlap_text = " ".join(words[-overlap_count:])
            current = [overlap_text]
            current_tokens = _count_tokens(overlap_text)
            overlap_stub = True
        else:
            current = []
            current_tokens = 0
            overlap_stub = False

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if current and current_tokens + para_tokens > target:
            flush_chunk()
        current.append(para)
        current_tokens += para_tokens
        overlap_stub = False

    if overlap_stub and len(current) == 1:
        # Only overlap text remaining – drop it.
        current = []
        current_tokens = 0
        overlap_stub = False
    flush_chunk()

    if not chunks:
        chunks.append(_render_with_headers(text, headers, cfg))
    return chunks


def prechunk_markdown(text: str, cfg: ChunkingConfig) -> list[str]:
    """
    Split Markdown text into pre-chunks according to the provided configuration.
    """

    if not cfg.enable:
        return [text]

    lines = text.splitlines()
    section_path: list[str] = []
    blocks: list[tuple[str, list[str], list[str]]] = []
    current: list[str] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            blocks.append(("prose", current.copy(), list(section_path)))
            current = []

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("#"):
            flush_current()
            level = len(stripped) - len(stripped.lstrip("#"))
            level = max(1, min(level, 3))
            title = stripped[level:].strip()
            while len(section_path) >= level:
                section_path.pop()
            section_path.append(title)
            current.append(line)
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped:
            flush_current()
            table_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i].rstrip("\n")
                if next_line.strip().startswith("|") and "|" in next_line.strip():
                    table_lines.append(next_line)
                    i += 1
                else:
                    break
            blocks.append(("table", table_lines, list(section_path)))
            continue

        list_marker = re.match(r"\s*([-*+]\s+|\d+[.)]\s+)", stripped)
        if list_marker:
            flush_current()
            list_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i].rstrip("\n")
                if not next_line.strip():
                    list_lines.append(next_line)
                    i += 1
                    break
                if re.match(r"\s*([-*+]\s+|\d+[.)]\s+)", next_line.strip()):
                    list_lines.append(next_line)
                    i += 1
                elif next_line.startswith(" "):
                    list_lines.append(next_line)
                    i += 1
                else:
                    break
            blocks.append(("prose", list_lines, list(section_path)))
            continue

        if not stripped:
            current.append(line)
            flush_current()
            i += 1
            continue

        current.append(line)
        i += 1

    flush_current()

    chunks: list[str] = []
    for block_type, block_lines, headers in blocks:
        if block_type == "table":
            chunks.extend(_split_table_block(block_lines, headers, cfg))
        else:
            chunks.extend(_split_prose_block(block_lines, headers, cfg))

    if not chunks:
        chunks.append(text.strip())

    return chunks


class _ChunkProgressWriter(io.TextIOBase):
    def __init__(
        self,
        original: io.TextIOBase,
        progress_bar,
        fallback_state: FallbackState | None,
    ) -> None:
        self._original = original
        self._progress = progress_bar
        self._fallback_state = fallback_state
        self._buffer = ""

    def writable(self) -> bool:  # type: ignore[override]
        return True

    def write(self, s: str) -> int:  # type: ignore[override]
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        self._original.flush()

    def _handle_line(self, line: str) -> None:
        stripped = line.rstrip("\r")
        if not stripped:
            return

        match = re.search(r"Chunk\s+(\d+)\s+of\s+(\d+)", stripped)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if self._progress.total != total:
                self._progress.reset(total=total)
            self._progress.n = min(current, total)
            self._progress.refresh()
            return

        if "LLM output format error" in stripped and self._fallback_state:
            self._fallback_state.register_error()

        self._original.write(stripped + "\n")


@contextlib.contextmanager
def _chunk_progress_manager(
    show_progress: bool,
    fallback_state: FallbackState | None,
):
    if not show_progress:
        yield None
        return

    progress_bar = tqdm(total=0, desc="LightRAG chunks", unit="chunk", leave=False)
    writer_stdout = _ChunkProgressWriter(sys.stdout, progress_bar, fallback_state)
    writer_stderr = _ChunkProgressWriter(sys.stderr, progress_bar, fallback_state)

    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(writer_stdout))
        stack.enter_context(contextlib.redirect_stderr(writer_stderr))
        try:
            yield progress_bar
        finally:
            writer_stdout.flush()
            writer_stderr.flush()
            progress_bar.close()


def _load_env_once() -> tuple[str, Optional[str]]:
    """
    Load environment variables (idempotent) and return OpenAI credentials.

    Raises:
        RuntimeError: if OPENAI_API_KEY is missing.
    """

    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Please add it to your .env file."
        )
    return api_key, os.getenv("OPENAI_BASE_URL")


def _build_embedding_func(
    api_key: str,
    base_url: Optional[str],
    model: str,
    embedding_dim: int,
) -> EmbeddingFunc:
    """
    Create an EmbeddingFunc wrapper for LightRAG using OpenAI embeddings.
    """

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        return openai_embed(
            texts,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    return EmbeddingFunc(embedding_dim=embedding_dim, func=_embed)


def _build_completion_func(
    api_key: str,
    base_url: Optional[str],
    model: str,
):
    """
    Wrap OpenAI completion with caching support for LightRAG.
    """

    def _completion(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Optional[Sequence[dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        return openai_complete_if_cache(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    return _completion


def _apply_llm_model(rag: LightRAG, completion_func, model_name: str) -> None:
    rag.llm_model_func = completion_func
    if hasattr(rag, "llm_model"):
        rag.llm_model = model_name
    setattr(rag, "_active_llm_model", model_name)


def build_lightrag(
    *,
    working_dir: Path | str = DEFAULT_WORKING_DIR,
) -> LightRAG:
    """
    Construct a LightRAG instance backed by OpenAI.

    The function reads configuration from environment variables (via `.env`):
        - OPENAI_API_KEY (required)
        - OPENAI_BASE_URL (optional)
        - LIGHTRAG_LLM_MODEL (default: gpt-5-mini)
        - LIGHTRAG_EMBED_MODEL (default: text-embedding-3-large)
        - LIGHTRAG_EMBED_DIM (optional; inferred from the model when omitted)
    """

    api_key, base_url = _load_env_once()
    llm_model = os.getenv("LIGHTRAG_LLM_MODEL", "gpt-5-mini")
    embed_model, embed_dim = _resolve_embedding_params()

    completion_func = _build_completion_func(api_key, base_url, llm_model)
    embedding_func = _build_embedding_func(
        api_key,
        base_url,
        embed_model,
        embed_dim,
    )

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=completion_func,
        embedding_func=embedding_func,
    )

    setattr(rag, "_openai_api_key", api_key)
    setattr(rag, "_openai_base_url", base_url)
    setattr(rag, "_active_llm_model", llm_model)
    setattr(rag, "_active_embed_model", embed_model)
    setattr(rag, "_active_embed_dim", embed_dim)

    return rag


async def ensure_initialized(rag: LightRAG) -> None:
    """
    Ensure LightRAG storages are initialised. Safe to call multiple times.
    """

    await rag.initialize_storages()
    from lightrag.kg.shared_storage import initialize_pipeline_status

    await initialize_pipeline_status()


def _iter_markdown_files(source_dir: Path) -> Iterable[Path]:
    return sorted(p for p in source_dir.glob("*.md") if p.is_file())


async def ingest_markdown_files(
    rag: LightRAG,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    *,
    split_by_character: str | None = "\n\n",
    split_by_character_only: bool = False,
    max_workers: int | None = None,
    show_progress: bool = True,
    show_chunk_progress: bool = False,
    chunking: ChunkingConfig | None = None,
    fallback: FallbackConfig | None = None,
) -> list[Path]:
    """
    Ingest Markdown documents into LightRAG.

    Args:
        rag: Initialised LightRAG instance.
        source_dir: Directory containing `.md` files.
        split_by_character: Optional hard split character for ingestion.
        split_by_character_only: If True, disables token-based chunking.
        max_workers: Optional override for LightRAG worker concurrency.
        show_progress: Display tqdm progress bar.
        show_chunk_progress: Display a single-line chunk progress bar.
        chunking: Optional Markdown pre-chunk configuration.
        fallback: Optional selective fallback configuration.

    Returns:
        List of ingested file paths.
    """

    await ensure_initialized(rag)

    if max_workers is not None and max_workers > 0:
        rag.max_parallel_insert = max_workers
        rag.embedding_func_max_async = max_workers
        rag.llm_model_max_async = max_workers

    chunking_cfg = chunking or ChunkingConfig.from_env()
    fallback_cfg = fallback or FallbackConfig.from_env()
    fallback_state: FallbackState | None = None
    if fallback_cfg.escalate_on_error:
        fallback_state = FallbackState(fallback_cfg, rag)

    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Markdown source directory not found: {source_dir}")

    paths = list(_iter_markdown_files(source_dir))
    iterator: Iterable[Path]
    if show_progress:
        iterator = tqdm(paths, desc="LightRAG ingest", unit="file")
    else:
        iterator = paths

    ingested: list[Path] = []
    for path in iterator:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_id = compute_mdhash_id(text, prefix="doc-")
        input_text = text
        actual_split = split_by_character
        actual_split_only = split_by_character_only

        if chunking_cfg.enable:
            pre_chunks = prechunk_markdown(text, chunking_cfg)
            input_text = DEFAULT_PRECHUNK_SENTINEL.join(pre_chunks)
            actual_split = DEFAULT_PRECHUNK_SENTINEL
            actual_split_only = True

        with _chunk_progress_manager(show_chunk_progress, fallback_state):
            await rag.ainsert(
                input=input_text,
                ids=doc_id,
                file_paths=str(path.name),
                split_by_character=actual_split,
                split_by_character_only=actual_split_only,
            )
        ingested.append(path)

    if fallback_state:
        summary = (
            f"LightRAG format errors: {fallback_state.error_count} "
            + (
                f"(escalated to {fallback_state.config.high_quality_model})"
                if fallback_state.escalated
                else "(no escalation)"
            )
        )
        print(summary)

    return ingested


async def aquery(
    rag: LightRAG,
    question: str,
    *,
    mode: str = "mix",
    **query_kwargs,
) -> str:
    """
    Run an asynchronous LightRAG query.

    Args:
        rag: LightRAG instance (should be initialised & ingested).
        question: Natural language question.
        mode: Retrieval mode (`local`, `global`, `hybrid`, `naive`, `mix`, `bypass`).
        **query_kwargs: Extra parameters forwarded to QueryParam.

    Returns:
        Response string generated by the LLM.
    """

    await ensure_initialized(rag)
    param = QueryParam(mode=mode, **query_kwargs)
    response = await rag.aquery(question, param=param)

    if isinstance(response, str):
        return response

    # Streaming iterator fallback
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    return "".join(chunks)


def ingest_markdown_files_sync(
    rag: LightRAG,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    *,
    split_by_character: str | None = "\n\n",
    split_by_character_only: bool = False,
    max_workers: int | None = None,
    show_progress: bool = True,
    show_chunk_progress: bool = False,
    chunking: ChunkingConfig | None = None,
    fallback: FallbackConfig | None = None,
) -> list[Path]:
    """
    Synchronous wrapper around `ingest_markdown_files`.
    """

    return asyncio.run(
        ingest_markdown_files(
            rag,
            source_dir,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            max_workers=max_workers,
            show_progress=show_progress,
            show_chunk_progress=show_chunk_progress,
            chunking=chunking,
            fallback=fallback,
        )
    )


def query(
    rag: LightRAG,
    question: str,
    *,
    mode: str = "mix",
    **query_kwargs,
) -> str:
    """
    Synchronous wrapper around `aquery`.
    """

    return asyncio.run(aquery(rag, question, mode=mode, **query_kwargs))


__all__ = [
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_WORKING_DIR",
    "ChunkingConfig",
    "FallbackConfig",
    "build_lightrag",
    "ensure_initialized",
    "ingest_markdown_files",
    "ingest_markdown_files_sync",
    "aquery",
    "query",
]

