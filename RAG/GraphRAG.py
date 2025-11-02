import os
from pathlib import Path
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from dotenv import load_dotenv
from RAG.GetNeighbor import wrap_with_neighbors
from RAG.MMR import wrap_with_mmr

# Load environment variables from .env at repo root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
INDEX_NAME = "Chunk"

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI)

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
embedder = OpenAIEmbeddings(model=EMBED_MODEL)


# Instantiate the LLM with temperature: 1 for GPT-5, else 0.2
LLM_MODEL = os.getenv("OPENAI_MODEL")
temperature = 1 if (LLM_MODEL or "").lower().startswith("gpt-5") else 0.1
llm = OpenAILLM(model_name=LLM_MODEL, model_params={"temperature": temperature})

_MMR_K = int(os.getenv("MMR_k"))
_MMR_LAMBDA = float(os.getenv("MMR_LAMBDA"))
_ALWAYS_KEEP_TOP = int(os.getenv("ALWAYS_KEEP_TOP"))

def _make_rag(neighbor_window: int):
    base = VectorRetriever(driver, INDEX_NAME, embedder)
    base = wrap_with_mmr(
        base,
        embedder=embedder,
        mmr_k=_MMR_K,
        lambda_mult=_MMR_LAMBDA,
        always_keep_top=_ALWAYS_KEEP_TOP,
    )
    wrapped = wrap_with_neighbors(base, driver, window=neighbor_window)
    return GraphRAG(retriever=wrapped, llm=llm)

# Resolve default top_k from environment
_env_top_k = os.getenv("top_k")
_DEFAULT_top_k = int(_env_top_k) if _env_top_k is not None else 5

def query_graph_rag(query_text: str, top_k: int = _DEFAULT_top_k, neighbor_window: int = 1):
    """Query the GraphRAG system with a question."""
    rag = _make_rag(neighbor_window)
    response = rag.search(query_text=query_text, retriever_config={"top_k": top_k})
    return response.answer