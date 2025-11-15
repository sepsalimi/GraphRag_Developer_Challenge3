from __future__ import annotations

import asyncio
import os
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
VENDOR_NODE_RAG = REPO_ROOT / "vendor" / "NodeRAG"
if VENDOR_NODE_RAG.exists():
    vendor_path = str(VENDOR_NODE_RAG)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

if "hnswlib_noderag" not in sys.modules:
    try:
        import hnswlib_noderag  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("hnswlib_noderag")

        class _MissingIndex:
            def __init__(self, *_, **__):
                raise RuntimeError(
                    "hnswlib_noderag is required for HNSW graph generation. "
                    "Set NODERAG_DISABLE_HNSW=true (default) to skip this stage."
                )

        stub.Index = _MissingIndex
        sys.modules["hnswlib_noderag"] = stub

from NodeRAG.build.Node import NodeRag, State
from NodeRAG.config import NodeConfig
from NodeRAG.storage.storage import storage


@dataclass
class NodeRagSettings:
    source_dir: Path
    work_dir: Path
    doc_type: str
    language: str
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    embed_dim: int
    embed_backend: str
    hf_model: str
    openai_embed_model: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_rate_limit: int
    disable_hnsw: bool
    disable_summary: bool
    max_workers: int


@dataclass
class PipelineOutputs:
    documents: pd.DataFrame
    text_units: pd.DataFrame
    semantic_units: pd.DataFrame
    entities: pd.DataFrame
    relationships: pd.DataFrame
    embeddings: Dict[str, List[float]]
    graph: Any


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_settings() -> NodeRagSettings:
    load_dotenv(ENV_PATH, override=False)
    load_dotenv(override=False)
    source_dir = _resolve_path(_env_str("NODERAG_SOURCE_DIR", "KnowledgeGraph/source_data"))
    work_dir = _resolve_path(_env_str("NODERAG_WORK_DIR", ".noderag"))
    doc_type = _env_str("NODERAG_DOC_TYPE", "mixed").lower()
    language = _env_str("NODERAG_LANGUAGE", "English")
    chunk_size = _env_int("NODERAG_CHUNK_TARGET_TOKENS", 700)
    chunk_overlap = _env_int("NODERAG_CHUNK_OVERLAP_TOKENS", 50)
    embedding_batch_size = _env_int("NODERAG_EMBED_BATCH_SIZE", 32)
    embed_dim = _env_int("NODERAG_EMBED_DIM", 1024)
    embed_backend = _env_str("NODERAG_EMBED_BACKEND", "hf").lower()
    hf_model = _env_str("NODERAG_HF_MODEL", "BAAI/bge-m3")
    openai_embed_model = _env_str("NODERAG_OPENAI_EMBED_MODEL", _env_str("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    llm_model = _env_str("NODERAG_LLM_MODEL", _env_str("OPENAI_MODEL", "")).strip()
    if not llm_model:
        raise RuntimeError("Set NODERAG_LLM_MODEL (or OPENAI_MODEL) before running NodeRAG.")
    llm_temperature = _env_float("NODERAG_LLM_TEMPERATURE", 0.0)
    llm_max_tokens = _env_int("NODERAG_LLM_MAX_TOKENS", 10000)
    llm_rate_limit = _env_int("NODERAG_LLM_RATE_LIMIT", 50)
    disable_hnsw = _env_bool("NODERAG_DISABLE_HNSW", True)
    disable_summary = _env_bool("NODERAG_DISABLE_SUMMARY", True)
    max_workers = _env_int("MAX_WORKERS", max(2, (os.cpu_count() or 4)))
    return NodeRagSettings(
        source_dir=source_dir,
        work_dir=work_dir,
        doc_type=doc_type,
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_batch_size=embedding_batch_size,
        embed_dim=embed_dim,
        embed_backend=embed_backend,
        hf_model=hf_model,
        openai_embed_model=openai_embed_model,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_rate_limit=llm_rate_limit,
        disable_hnsw=disable_hnsw,
        disable_summary=disable_summary,
        max_workers=max_workers,
    )


def _ensure_subfolders(work_dir: Path, reset_cache: bool) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "cache"
    info_dir = work_dir / "info"
    input_dir = work_dir / "input"
    if reset_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(info_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)


def _sync_source_files(settings: NodeRagSettings, *, resync: bool) -> None:
    input_dir = settings.work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    if resync:
        for existing in input_dir.glob("*"):
            if existing.is_file():
                existing.unlink()
    if settings.doc_type == "mixed":
        patterns: List[str] = ["*.md", "*.txt"]
    else:
        suffix = settings.doc_type.lstrip(".")
        patterns = [f"*.{suffix}"]
    copied = 0
    for pattern in patterns:
        for file in settings.source_dir.glob(pattern):
            if file.is_file():
                dest = input_dir / file.name
                if not dest.exists() or resync:
                    shutil.copy2(file, dest)
                    copied += 1
    if copied:
        print(f"Copied {copied} files into {input_dir}")


def _ensure_node_config_yaml(settings: NodeRagSettings) -> Path:
    config_path = settings.work_dir / "Node_config.yaml"
    NodeConfig.create_config_file(str(settings.work_dir))
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = {}
    config_section = data.setdefault("config", {})
    config_section.update(
        {
            "main_folder": str(settings.work_dir),
            "language": settings.language,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_batch_size": settings.embedding_batch_size,
            "dim": settings.embed_dim,
            "docu_type": settings.doc_type,
            "max_workers": settings.max_workers,
        }
    )
    model_config = data.setdefault("model_config", {})
    model_config.update(
        {
            "service_provider": "openai",
            "model_name": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "rate_limit": settings.llm_rate_limit,
        }
    )
    # Keep secrets in .env rather than persisting them to disk.
    model_config.pop("api_keys", None)
    embedding_config = data.setdefault("embedding_config", {})
    if settings.embed_backend == "hf":
        embedding_config.update(
            {
                "service_provider": "hf_embedding",
                "model_name": settings.hf_model,
                "batch_size": settings.embedding_batch_size,
            }
        )
    else:
        embedding_config.update(
            {
                "service_provider": "openai_embedding",
                "embedding_model_name": settings.openai_embed_model,
                "rate_limit": settings.llm_rate_limit,
            }
        )
        embedding_config.pop("api_keys", None)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return config_path


def _build_node_config(settings: NodeRagSettings) -> NodeConfig:
    _ensure_node_config_yaml(settings)
    NodeConfig._instance = None
    return NodeConfig.from_main_folder(str(settings.work_dir))


def _run_pipeline(settings: NodeRagSettings, node_config: NodeConfig, *, show_progress: bool) -> None:
    runner = NodeRag(node_config, web_ui=show_progress)
    skip_states = set()
    if settings.disable_hnsw:
        skip_states.add(State.HNSW_PIPELINE)
    if settings.disable_summary:
        skip_states.add(State.SUMMARY_PIPELINE)
    for state in skip_states:
        if state in runner.state_sequence:
            runner.state_sequence = [s for s in runner.state_sequence if s != state]
            runner.state_pipeline_map.pop(state, None)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        loop.run_until_complete(runner._run_async())
    else:
        runner.run()


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _convert_embeddings(df: pd.DataFrame) -> Dict[str, List[float]]:
    if df.empty:
        return {}
    result: Dict[str, List[float]] = {}
    for record in df.to_dict("records"):
        hash_id = record.get("hash_id")
        vector = record.get("embedding")
        if not hash_id or vector is None:
            continue
        if isinstance(vector, str) and vector == "done":
            continue
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        result[str(hash_id)] = [float(x) for x in vector]
    return result


def _load_outputs(node_config: NodeConfig) -> PipelineOutputs:
    documents = _safe_read_parquet(Path(node_config.documents_path))
    text_units = _safe_read_parquet(Path(node_config.text_path))
    semantic_units = _safe_read_parquet(Path(node_config.semantic_units_path))
    entities = _safe_read_parquet(Path(node_config.entities_path))
    relationships = _safe_read_parquet(Path(node_config.relationship_path))
    embeddings = _convert_embeddings(_safe_read_parquet(Path(node_config.embedding)))
    graph = None
    graph_path = Path(node_config.graph_path)
    base_graph_path = Path(getattr(node_config, "base_graph_path", graph_path))
    if graph_path.exists():
        graph = storage.load_pickle(graph_path)
    elif base_graph_path.exists():
        graph = storage.load_pickle(base_graph_path)
    return PipelineOutputs(
        documents=documents,
        text_units=text_units,
        semantic_units=semantic_units,
        entities=entities,
        relationships=relationships,
        embeddings=embeddings,
        graph=graph,
    )


def run_noderag(
    *,
    show_progress: bool = True,
    reset_cache: bool = False,
    resync_input: bool = True,
) -> PipelineOutputs:
    settings = _load_settings()
    if not settings.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {settings.source_dir}")
    _ensure_subfolders(settings.work_dir, reset_cache=reset_cache)
    _sync_source_files(settings, resync=resync_input)
    node_config = _build_node_config(settings)
    _run_pipeline(settings, node_config, show_progress=show_progress)
    return _load_outputs(node_config)


def load_cached_outputs() -> PipelineOutputs:
    settings = _load_settings()
    _ensure_subfolders(settings.work_dir, reset_cache=False)
    node_config = _build_node_config(settings)
    return _load_outputs(node_config)


def summarize_outputs(outputs: PipelineOutputs) -> Dict[str, int]:
    return {
        "documents": int(outputs.documents.shape[0]),
        "text_units": int(outputs.text_units.shape[0]),
        "passages": int(outputs.semantic_units.shape[0]),
        "entities": int(outputs.entities.shape[0]),
        "relationships": int(outputs.relationships.shape[0]),
        "embeddings": len(outputs.embeddings),
    }


def print_graph_overview(graph, *, top_k: int = 5) -> None:
    if graph is None:
        print("No graph available in the NodeRAG cache.")
        return
    node_types: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        node_type = data.get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    edge_types: Dict[str, int] = {}
    for u, v, data in graph.edges(data=True):
        label = f"{graph.nodes[u].get('type','?')} -> {graph.nodes[v].get('type','?')}"
        edge_types[label] = edge_types.get(label, 0) + 1
    print("Node counts by type:")
    for node_type, count in sorted(node_types.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {node_type}: {count}")
    print("Edge counts by type:")
    for edge_type, count in sorted(edge_types.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {edge_type}: {count}")
    if top_k > 0:
        degrees = sorted(graph.degree, key=lambda kv: kv[1], reverse=True)[:top_k]
        print(f"Top {top_k} nodes by degree:")
        for node_id, degree in degrees:
            node_type = graph.nodes[node_id].get("type", "unknown")
            print(f"  {node_id} ({node_type}): degree {degree}")


def get_cache_dir() -> Path:
    settings = _load_settings()
    return settings.work_dir / "cache"


__all__ = [
    "get_cache_dir",
    "load_cached_outputs",
    "print_graph_overview",
    "run_noderag",
    "summarize_outputs",
]

