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
from RAG.UseCatalog import load_catalog, card_first_gate

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

# Load catalog once at startup (no-op if already loaded)
try:
    load_catalog()
except Exception:
    pass


def _get_field(obj, name):
    try:
        if hasattr(obj, name):
            return getattr(obj, name)
    except Exception:
        pass
    if isinstance(obj, dict):
        if name in obj:
            return obj.get(name)
        meta = obj.get("metadata")
        if isinstance(meta, dict) and name in meta:
            return meta.get(name)
    if hasattr(obj, "metadata") and isinstance(getattr(obj, "metadata"), dict):
        md = getattr(obj, "metadata")
        if name in md:
            return md.get(name)
    return None


class _SourceScopedRetriever:
    def __init__(self, base_retriever, allowed_sources):
        self._base = base_retriever
        # normalize to lowercase strings
        self._allowed = {str(s).lower() for s in (allowed_sources or [])}

    def __getattr__(self, item):
        return getattr(self._base, item)

    def _keep(self, hit) -> bool:
        if not self._allowed:
            return True
        doc = (
            _get_field(hit, "source")
            or _get_field(hit, "document_key")
            or _get_field(hit, "file")
            or ""
        )
        ds = str(doc).lower()
        if not ds:
            return False
        # allow exact, suffix, or substring match
        for s in self._allowed:
            if ds == s or ds.endswith(s) or s in ds:
                return True
        return False

    def _filter(self, results):
        try:
            return [r for r in (results or []) if self._keep(r)]
        except Exception:
            return results

    def search(self, *args, **kwargs):
        if hasattr(self._base, "search"):
            results = self._base.search(*args, **kwargs)
        elif hasattr(self._base, "retrieve"):
            results = self._base.retrieve(*args, **kwargs)
        else:
            return []
        return self._filter(results)

    def retrieve(self, *args, **kwargs):
        if hasattr(self._base, "retrieve"):
            results = self._base.retrieve(*args, **kwargs)
        elif hasattr(self._base, "search"):
            results = self._base.search(*args, **kwargs)
        else:
            return []
        return self._filter(results)

    def get_search_results(self, *args, **kwargs):
        if hasattr(self._base, "get_search_results"):
            results = self._base.get_search_results(*args, **kwargs)
        elif hasattr(self._base, "search"):
            results = self._base.search(*args, **kwargs)
        elif hasattr(self._base, "retrieve"):
            results = self._base.retrieve(*args, **kwargs)
        else:
            return []
        return self._filter(results)

def _make_rag(neighbor_window: int, scope_files=None):
    base = VectorRetriever(driver, INDEX_NAME, embedder)
    if scope_files:
        base = _SourceScopedRetriever(base, scope_files)
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

def _format_card_answer(answers, provenance) -> str:
    parts = []
    try:
        for k, v in (answers or {}).items():
            parts.append(f"{k}: {v}")
        if provenance:
            prov_bits = []
            for p in provenance:
                src = p.get("source")
                lines = p.get("lines")
                if src is None:
                    continue
                if lines is None or lines == "":
                    prov_bits.append(str(src))
                else:
                    prov_bits.append(f"{src}:{lines}")
            if prov_bits:
                parts.append("Provenance: " + "; ".join(prov_bits))
    except Exception:
        pass
    return "\n".join(parts) if parts else ""


def query_graph_rag(query_text: str, top_k: int = _DEFAULT_top_k, neighbor_window: int = 1):
    """Query the GraphRAG system with a question."""
    # Card-first gate
    scope_files = None
    gate = card_first_gate(query_text, allow_direct_answer=True)
    mode = gate.get("mode")
    if mode == "answer":
        return _format_card_answer(gate.get("answers"), gate.get("provenance"))
    if mode == "restrict":
        scope_files = gate.get("scope_files") or None

    rag = _make_rag(neighbor_window, scope_files=scope_files)
    response = rag.search(query_text=query_text, retriever_config={"top_k": top_k})
    return response.answer