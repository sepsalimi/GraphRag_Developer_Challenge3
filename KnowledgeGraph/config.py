"""
 
Connect to nio4j knowledge graph databae

"""

import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph


def _infer_embedding_dim(model: str) -> int:
    model = (model or "").lower()
    if "large" in model:
        return 3072
    if "small" in model:
        return 1536
    return 1536


def _backfill_chunk_index(graph: Neo4jGraph) -> None:
    """Ensure c.chunk_index exists on all Chunk nodes.

    Two-step, idempotent process:
      1) Copy numeric article_number into chunk_index when available.
      2) For remaining NULLs, assign deterministic per-document order by id(c).
    """
    try:
        counts = list(graph.query(
            "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN count(c) AS total, "
            "       sum(CASE WHEN c.chunk_index IS NULL THEN 1 ELSE 0 END) AS missing"
        ))
        missing = int((counts[0] or {}).get("missing", 0)) if counts else 0
    except Exception:
        missing = 0
    if missing <= 0:
        return
    try:
        graph.query(
            "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) "
            "WHERE c.chunk_index IS NULL AND c.article_number IS NOT NULL "
            "SET c.chunk_index = toInteger(c.article_number)"
        )
    except Exception:
        pass
    try:
        graph.query(
            "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) "
            "WHERE c.chunk_index IS NULL "
            "WITH d, c ORDER BY id(c) "
            "WITH d, collect(c) AS chunks "
            "UNWIND range(0, size(chunks)-1) AS idx "
            "WITH chunks[idx] AS c, idx "
            "SET c.chunk_index = idx"
        )
    except Exception:
        pass

def load_neo4j_graph(env_path: str = None):
    # Load .env from project root (parent directory)
    if env_path is None:
        # Get the directory where this file is located, then go up one level
        config_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(config_dir)
        env_path = os.path.join(project_root, '.env')
    
    load_dotenv(env_path, override=True)
    
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
    
    # Optional: OpenAI config if needed elsewhere
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings' if os.getenv('OPENAI_BASE_URL') else None
    OPENAI_EMBED_MODEL = os.getenv('OPENAI_EMBED_MODEL', 'text-embedding-3-small')
    raw_embed_dim = os.getenv('OPENAI_EMBED_DIM')
    if raw_embed_dim:
        try:
            embed_dim = int(raw_embed_dim)
        except ValueError:
            embed_dim = _infer_embedding_dim(OPENAI_EMBED_MODEL)
    else:
        embed_dim = _infer_embedding_dim(OPENAI_EMBED_MODEL)
        # Expose the computed dimension to downstream helpers (vector index creation, etc.)
        os.environ['OPENAI_EMBED_DIM'] = str(embed_dim)
    
    # Initialize Neo4j graph object (GraphStore-compatible)
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    # Idempotent backfill to guarantee Chunk.chunk_index exists for neighbor retrieval
    try:
        _backfill_chunk_index(graph)
    except Exception:
        # Avoid failing graph load due to maintenance backfill
        pass

    return graph, OPENAI_API_KEY, OPENAI_ENDPOINT, OPENAI_EMBED_MODEL
