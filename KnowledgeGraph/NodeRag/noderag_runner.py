from __future__ import annotations

import asyncio
import os
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
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

from KnowledgeGraph.config import load_neo4j_graph

ENV_PATH = REPO_ROOT / ".env"


@dataclass
class NodeRAGSettings:
    source_dir: Path
    work_dir: Path
    embed_backend: str
    hf_model: str
    openai_embed_model: str
    embed_dim: int
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    doc_type: str
    language: str
    llm_model: Optional[str]
    llm_temperature: float
    llm_max_tokens: int
    llm_rate_limit: int
    disable_hnsw: bool
    disable_summary: bool
    similarity_weight: float
    accuracy_weight: float
    reset_graph: bool


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


def _ensure_folder_structure(work_dir: Path) -> None:
    for name in ("input", "cache", "info"):
        (work_dir / name).mkdir(parents=True, exist_ok=True)


def _load_settings() -> NodeRAGSettings:
    load_dotenv(ENV_PATH, override=False)
    load_dotenv(override=False)

    source_dir = _resolve_path(os.getenv("NODERAG_SOURCE_DIR", "KnowledgeGraph/source_data"))
    work_dir = _resolve_path(os.getenv("NODERAG_WORK_DIR", ".noderag"))
    embed_backend = os.getenv("NODERAG_EMBED_BACKEND", "hf").strip().lower()
    hf_model = os.getenv("NODERAG_HF_MODEL", "BAAI/bge-m3").strip()
    openai_embed_model = os.getenv(
        "NODERAG_OPENAI_EMBED_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    ).strip()
    embed_dim = _env_int("NODERAG_EMBED_DIM", 1024)
    chunk_size = _env_int("NODERAG_CHUNK_TARGET_TOKENS", 700)
    chunk_overlap = _env_int("NODERAG_CHUNK_OVERLAP_TOKENS", 50)
    embedding_batch_size = _env_int("NODERAG_EMBED_BATCH_SIZE", 32)
    doc_type = os.getenv("NODERAG_DOC_TYPE", "mixed").strip().lower()
    language = os.getenv("NODERAG_LANGUAGE", "English").strip()
    llm_model = os.getenv("NODERAG_LLM_MODEL") or os.getenv("OPENAI_MODEL")
    llm_temperature = _env_float("NODERAG_LLM_TEMPERATURE", 0.0)
    llm_max_tokens = _env_int("NODERAG_LLM_MAX_TOKENS", 10000)
    llm_rate_limit = _env_int("NODERAG_LLM_RATE_LIMIT", 50)
    disable_hnsw = _env_bool("NODERAG_DISABLE_HNSW", True)
    disable_summary = _env_bool("NODERAG_DISABLE_SUMMARY", True)
    similarity_weight = _env_float("NODERAG_SIMILARITY_WEIGHT", 1.0)
    accuracy_weight = _env_float("NODERAG_ACCURACY_WEIGHT", 10.0)
    reset_graph = _env_bool("NODERAG_RESET_GRAPH", True)

    settings = NodeRAGSettings(
        source_dir=source_dir,
        work_dir=work_dir,
        embed_backend=embed_backend,
        hf_model=hf_model,
        openai_embed_model=openai_embed_model,
        embed_dim=embed_dim,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_batch_size=embedding_batch_size,
        doc_type=doc_type,
        language=language,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_rate_limit=llm_rate_limit,
        disable_hnsw=disable_hnsw,
        disable_summary=disable_summary,
        similarity_weight=similarity_weight,
        accuracy_weight=accuracy_weight,
        reset_graph=reset_graph,
    )
    _ensure_folder_structure(settings.work_dir)
    return settings


def _prepare_workdir(settings: NodeRAGSettings) -> None:
    _ensure_folder_structure(settings.work_dir)
    input_dir = settings.work_dir / "input"
    for existing in input_dir.glob("*"):
        if existing.is_file():
            existing.unlink()
    patterns: List[str]
    if settings.doc_type == "mixed":
        patterns = ["*.md", "*.txt"]
    else:
        suffix = settings.doc_type.lstrip(".")
        patterns = [f"*.{suffix}"]
    for pattern in patterns:
        for file in settings.source_dir.glob(pattern):
            if file.is_file():
                shutil.copy2(file, input_dir / file.name)


def _build_config_dict(settings: NodeRAGSettings) -> Dict[str, Any]:
    config_section: Dict[str, Any] = {
        "main_folder": str(settings.work_dir),
        "language": settings.language,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_batch_size": settings.embedding_batch_size,
        "dim": settings.embed_dim,
        "docu_type": settings.doc_type,
        "similarity_weight": settings.similarity_weight,
        "accuracy_weight": settings.accuracy_weight,
        "max_workers": _env_int("MAX_WORKERS", max(2, (os.cpu_count() or 4)))
    }
    model_config: Dict[str, Any] = {
        "service_provider": "openai",
        "model_name": settings.llm_model,
        "api_keys": os.getenv("OPENAI_API_KEY"),
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "rate_limit": settings.llm_rate_limit,
    }
    if not model_config["model_name"]:
        raise RuntimeError("NODERAG_LLM_MODEL or OPENAI_MODEL must be configured.")

    if settings.embed_backend == "openai":
        embedding_config = {
            "service_provider": "openai_embedding",
            "embedding_model_name": settings.openai_embed_model,
            "api_keys": os.getenv("OPENAI_API_KEY"),
            "rate_limit": settings.llm_rate_limit,
        }
    elif settings.embed_backend == "hf":
        embedding_config = {
            "service_provider": "hf_embedding",
            "model_name": settings.hf_model,
            "batch_size": settings.embedding_batch_size,
        }
    else:
        raise ValueError(f"Unsupported embedding backend: {settings.embed_backend}")

    return {
        "config": config_section,
        "model_config": model_config,
        "embedding_config": embedding_config,
    }


def _run_pipeline(settings: NodeRAGSettings, config_dict: Dict[str, Any]) -> NodeConfig:
    NodeConfig._instance = None
    node_config = NodeConfig(config_dict)
    runner = NodeRag(node_config, web_ui=True)
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
    return node_config


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _convert_embeddings(df: pd.DataFrame) -> Dict[str, List[float]]:
    if df.empty:
        return {}
    embeddings: Dict[str, List[float]] = {}
    for record in df.to_dict("records"):
        hash_id = record.get("hash_id")
        vector = record.get("embedding")
        if not hash_id or vector is None:
            continue
        if isinstance(vector, str) and vector == "done":
            continue
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        embeddings[str(hash_id)] = [float(x) for x in vector]
    return embeddings


def _load_outputs(node_config: NodeConfig) -> PipelineOutputs:
    documents = _safe_read_parquet(Path(node_config.documents_path))
    text_units = _safe_read_parquet(Path(node_config.text_path))
    semantic_units = _safe_read_parquet(Path(node_config.semantic_units_path))
    entities = _safe_read_parquet(Path(node_config.entities_path))
    relationships = _safe_read_parquet(Path(node_config.relationship_path))
    embeddings = _convert_embeddings(_safe_read_parquet(Path(node_config.embedding)))
    graph = None
    graph_path = Path(node_config.graph_path)
    if graph_path.exists():
        graph = storage.load_pickle(graph_path)
    return PipelineOutputs(
        documents=documents,
        text_units=text_units,
        semantic_units=semantic_units,
        entities=entities,
        relationships=relationships,
        embeddings=embeddings,
        graph=graph,
    )


def _chunked(items: Iterable[Dict[str, Any]], size: int = 200) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _clean_props(props: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        if isinstance(v, dict) and len(v) == 0:
            continue
        cleaned[k] = v
    return cleaned


def _ingest_outputs(
    outputs: PipelineOutputs,
    graph_db,
    settings: NodeRAGSettings,
    *,
    reset_graph: bool,
) -> Dict[str, int]:
    if reset_graph:
        graph_db.query(
            """
            MATCH (n)
            WHERE n:NR_Document OR n:NR_Passage OR n:NR_Entity OR n:NR_Relationship
            DETACH DELETE n
            """
        )
        try:
            graph_db.query("DROP INDEX `NR_Passage` IF EXISTS")
        except Exception:
            pass

    doc_rows: List[Dict[str, Any]] = []
    for record in outputs.documents.to_dict("records"):
        doc_hash = record.get("doc_hash_id")
        if not doc_hash:
            continue
        props = _clean_props(
            {
                "document_id": record.get("doc_id") or record.get("doc_hash_id"),
                "source_path": record.get("path"),
                "text_id": record.get("text_id"),
                "text_hash_id": record.get("text_hash_id"),
            }
        )
        doc_rows.append({"doc_hash_id": str(doc_hash), "props": props})
    for batch in _chunked(doc_rows):
        graph_db.query(
            """
            UNWIND $rows AS row
            MERGE (d:NR_Document {doc_hash_id: row.doc_hash_id})
            SET d += row.props
            """,
            params={"rows": batch},
        )

    text_lookup: Dict[str, Dict[str, Any]] = {
        str(rec.get("hash_id")): rec for rec in outputs.text_units.to_dict("records") if rec.get("hash_id")
    }

    passage_nodes: List[Dict[str, Any]] = []
    passage_doc_edges: List[Dict[str, Any]] = []
    for record in outputs.semantic_units.to_dict("records"):
        hash_id = record.get("hash_id")
        if not hash_id:
            continue
        text_info = text_lookup.get(str(record.get("text_hash_id")), {})
        doc_hash_id = text_info.get("doc_hash_id")
        embedding = outputs.embeddings.get(str(hash_id))
        props = _clean_props(
            {
                "human_readable_id": record.get("human_readable_id"),
                "context": record.get("context"),
                "text_hash_id": record.get("text_hash_id"),
                "document_hash_id": doc_hash_id,
                "document_id": text_info.get("doc_id"),
                "weight": record.get("weight"),
                "embedding": embedding,
            }
        )
        passage_nodes.append({"hash_id": str(hash_id), "props": props})
        if doc_hash_id:
            passage_doc_edges.append(
                {"hash_id": str(hash_id), "doc_hash_id": str(doc_hash_id)}
            )
    for batch in _chunked(passage_nodes):
        graph_db.query(
            """
            UNWIND $rows AS row
            MERGE (p:NR_Passage {hash_id: row.hash_id})
            SET p += row.props
            """,
            params={"rows": batch},
        )
    for batch in _chunked(passage_doc_edges):
        graph_db.query(
            """
            UNWIND $rows AS row
            MATCH (p:NR_Passage {hash_id: row.hash_id})
            MATCH (d:NR_Document {doc_hash_id: row.doc_hash_id})
            MERGE (p)-[:FROM_DOCUMENT]->(d)
            """,
            params={"rows": batch},
        )

    entity_nodes: List[Dict[str, Any]] = []
    for record in outputs.entities.to_dict("records"):
        hash_id = record.get("hash_id")
        if not hash_id:
            continue
        props = _clean_props(
            {
                "name": record.get("context"),
                "text_hash_id": record.get("text_hash_id"),
                "weight": record.get("weight"),
            }
        )
        entity_nodes.append({"hash_id": str(hash_id), "props": props})
    for batch in _chunked(entity_nodes):
        graph_db.query(
            """
            UNWIND $rows AS row
            MERGE (e:NR_Entity {hash_id: row.hash_id})
            SET e += row.props
            """,
            params={"rows": batch},
        )

    relationship_lookup: Dict[str, Dict[str, Any]] = {
        str(rec.get("hash_id")): rec for rec in outputs.relationships.to_dict("records") if rec.get("hash_id")
    }
    relationship_nodes: List[Dict[str, Any]] = []
    for rel_id, record in relationship_lookup.items():
        props = _clean_props(
            {
                "context": record.get("context"),
                "weight": record.get("weight"),
            }
        )
        relationship_nodes.append({"hash_id": rel_id, "props": props})
    for batch in _chunked(relationship_nodes):
        graph_db.query(
            """
            UNWIND $rows AS row
            MERGE (r:NR_Relationship {hash_id: row.hash_id})
            SET r += row.props
            """,
            params={"rows": batch},
        )

    mentions_edges: List[Dict[str, Any]] = []
    if outputs.graph is not None:
        seen_mentions = set()
        for u, v, data in outputs.graph.edges(data=True):
            u_type = outputs.graph.nodes[u].get("type")
            v_type = outputs.graph.nodes[v].get("type")
            weight = data.get("weight", 1)
            if u_type == "semantic_unit" and v_type == "entity":
                key = (str(u), str(v))
                if key in seen_mentions:
                    continue
                seen_mentions.add(key)
                mentions_edges.append({"hash_id": str(u), "entity_id": str(v), "weight": weight})
            elif u_type == "entity" and v_type == "semantic_unit":
                key = (str(v), str(u))
                if key in seen_mentions:
                    continue
                seen_mentions.add(key)
                mentions_edges.append({"hash_id": str(v), "entity_id": str(u), "weight": weight})
    for batch in _chunked(mentions_edges):
        graph_db.query(
            """
            UNWIND $rows AS row
            MATCH (p:NR_Passage {hash_id: row.hash_id})
            MATCH (e:NR_Entity {hash_id: row.entity_id})
            MERGE (p)-[rel:MENTIONS]->(e)
            SET rel.weight = coalesce(row.weight, 1)
            """,
            params={"rows": batch},
        )

    relationship_edges: List[Dict[str, Any]] = []
    if outputs.graph is not None:
        seen_rel_edges = set()
        for node, data in outputs.graph.nodes(data=True):
            if data.get("type") != "relationship":
                continue
            neighbors = list(outputs.graph.neighbors(node))
            for neighbor in neighbors:
                if outputs.graph.nodes[neighbor].get("type") != "entity":
                    continue
                key = (str(node), str(neighbor))
                if key in seen_rel_edges:
                    continue
                seen_rel_edges.add(key)
                weight = outputs.graph[node][neighbor].get("weight", 1)
                relationship_edges.append(
                    {
                        "relationship_id": str(node),
                        "entity_id": str(neighbor),
                        "weight": weight,
                    }
                )
    for batch in _chunked(relationship_edges):
        graph_db.query(
            """
            UNWIND $rows AS row
            MATCH (r:NR_Relationship {hash_id: row.relationship_id})
            MATCH (e:NR_Entity {hash_id: row.entity_id})
            MERGE (e)-[rel:INVOLVES]->(r)
            SET rel.weight = coalesce(row.weight, 1)
            """,
            params={"rows": batch},
        )

    return {
        "documents": int(outputs.documents.shape[0]),
        "passages": int(outputs.semantic_units.shape[0]),
        "entities": int(outputs.entities.shape[0]),
        "relationships": int(outputs.relationships.shape[0]),
    }


def preprocess_and_ingest(
    *,
    graph=None,
    show_progress: bool = True,
    reset_graph: Optional[bool] = None,
) -> Dict[str, int]:
    settings = _load_settings()
    if not settings.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {settings.source_dir}")
    _prepare_workdir(settings)
    config_dict = _build_config_dict(settings)
    node_config = _run_pipeline(settings, config_dict)
    outputs = _load_outputs(node_config)
    graph = graph or load_neo4j_graph()[0]
    should_reset = settings.reset_graph if reset_graph is None else bool(reset_graph)
    stats = _ingest_outputs(outputs, graph, settings, reset_graph=should_reset)
    ensure_vector_index(graph=graph, config=settings)
    return stats


def ensure_vector_index(*, graph=None, config: Optional[NodeRAGSettings] = None) -> None:
    settings = config or _load_settings()
    graph = graph or load_neo4j_graph()[0]
    graph.query(
        """
        CREATE VECTOR INDEX `NR_Passage` IF NOT EXISTS
        FOR (n:NR_Passage) ON (n.embedding)
        OPTIONS { indexConfig: {
            `vector.dimensions`: $dim,
            `vector.similarity_function`: 'cosine'
        }}
        """,
        params={"dim": int(settings.embed_dim)},
    )


def _embed_queries(queries: Sequence[str], settings: NodeRAGSettings) -> List[List[float]]:
    if not queries:
        return []
    config_dict = _build_config_dict(settings)
    NodeConfig._instance = None
    node_config = NodeConfig(config_dict)
    client = node_config.embedding_client
    if client is None:
        raise RuntimeError("Embedding client is not configured.")
    vectors = client.request(list(queries))
    return [[float(x) for x in vector] for vector in vectors]


def _vector_search(graph, embedding: Sequence[float], *, top_k: int, settings: NodeRAGSettings) -> List[Dict[str, Any]]:
    results = graph.query(
        """
        CALL db.index.vector.queryNodes('NR_Passage', $top_k, $embedding)
        YIELD node, score
        OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(doc:NR_Document)
        OPTIONAL MATCH (node)-[:MENTIONS]->(ent:NR_Entity)
        RETURN
            node.hash_id AS passage_id,
            node.context AS context,
            node.weight AS weight,
            doc.document_id AS document_id,
            doc.doc_hash_id AS document_hash_id,
            collect(DISTINCT ent.name) AS entities,
            score
        ORDER BY score DESC
        """,
        params={"top_k": int(top_k), "embedding": [float(x) for x in embedding]},
    )
    formatted: List[Dict[str, Any]] = []
    for row in results:
        formatted.append(
            {
                "passage_id": row.get("passage_id"),
                "context": row.get("context"),
                "weight": row.get("weight"),
                "document_id": row.get("document_id"),
                "document_hash_id": row.get("document_hash_id"),
                "entities": row.get("entities") or [],
                "score": row.get("score"),
            }
        )
    return formatted


def query_single(question: str, *, top_k: int = 5, graph=None) -> List[Dict[str, Any]]:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    settings = _load_settings()
    vectors = _embed_queries([question], settings)
    if not vectors:
        return []
    graph = graph or load_neo4j_graph()[0]
    return _vector_search(graph, vectors[0], top_k=top_k, settings=settings)


def query_batch(input_path: str | Path, *, top_k: int = 5, graph=None) -> List[Dict[str, Any]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "question" not in df.columns:
            raise ValueError("CSV input must contain a 'question' column.")
        questions = df["question"].dropna().astype(str).tolist()
    else:
        questions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not questions:
        return []
    settings = _load_settings()
    vectors = _embed_queries(questions, settings)
    graph = graph or load_neo4j_graph()[0]
    results: List[Dict[str, Any]] = []
    for question, vector in zip(questions, vectors):
        hits = _vector_search(graph, vector, top_k=top_k, settings=settings)
        results.append({"question": question, "results": hits})
    return results


__all__ = [
    "preprocess_and_ingest",
    "ensure_vector_index",
    "query_single",
    "query_batch",
]

