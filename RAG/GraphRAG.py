import os
import re
from pathlib import Path
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResult
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
    if hasattr(obj, name):
        return getattr(obj, name)

    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict) and name in metadata:
        return metadata[name]

    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        metadata = obj.get("metadata")
        if isinstance(metadata, dict) and name in metadata:
            return metadata[name]

    if name in {"source", "document_key", "file"}:
        text = None
        if hasattr(obj, "content"):
            text = getattr(obj, "content")
        if isinstance(text, dict):
            text = text.get("text")
        if not isinstance(text, str) and hasattr(obj, "text"):
            text = getattr(obj, "text")
        if not isinstance(text, str) and isinstance(obj, dict):
            text = obj.get("text")
        if isinstance(text, str):
            if name in {"source", "document_key"}:
                match = re.search(r"\[PUB=([^\]]+)\]", text)
                if match:
                    return match.group(1)
            if name == "file":
                match = re.search(r"\[DOC=([^\]]+)\]", text)
                if match:
                    return match.group(1)

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
        items, rebuild = self._unwrap(results)
        if not items:
            return rebuild(items)
        if not self._allowed:
            return rebuild(items)
        filtered = [hit for hit in items if self._keep(hit)]
        if filtered:
            return rebuild(filtered)
        return rebuild(items)

    def _unwrap(self, result_obj):
        if isinstance(result_obj, RetrieverResult):
            metadata = result_obj.metadata
            return list(result_obj.items), lambda items: RetrieverResult(items=items, metadata=metadata)
        if isinstance(result_obj, RawSearchResult):
            metadata = result_obj.metadata
            return list(result_obj.records), lambda items: RawSearchResult(records=items, metadata=metadata)
        if isinstance(result_obj, list):
            return list(result_obj), lambda items: items
        return [], lambda items: result_obj

    def search(self, *args, **kwargs):
        base = getattr(self._base, "search", None)
        if base is None and hasattr(self._base, "retrieve"):
            base = getattr(self._base, "retrieve")
        if base is None:
            return []
        return self._filter(base(*args, **kwargs))

    def retrieve(self, *args, **kwargs):
        base = getattr(self._base, "retrieve", None)
        if base is None and hasattr(self._base, "search"):
            base = getattr(self._base, "search")
        if base is None:
            return []
        return self._filter(base(*args, **kwargs))

    def get_search_results(self, *args, **kwargs):
        base = getattr(self._base, "get_search_results", None)
        if base is None and hasattr(self._base, "search"):
            base = getattr(self._base, "search")
        if base is None and hasattr(self._base, "retrieve"):
            base = getattr(self._base, "retrieve")
        if base is None:
            return []
        return self._filter(base(*args, **kwargs))

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
    try:
        a = answers or {}
        sentences = []

        def has(k):
            return k in a and a[k] not in (None, "", [])

        def kv(k, default=None):
            return a.get(k, default)

        # Dates and times
        if has("closing_date") and has("closing_time"):
            sentences.append(f"Closing date is {kv('closing_date')} at {kv('closing_time')}.")
        elif has("closing_date"):
            sentences.append(f"Closing date is {kv('closing_date')}.")
        elif has("closing_time"):
            sentences.append(f"Closing time is {kv('closing_time')}.")

        # Bid validity
        if has("bid_validity_days"):
            sentences.append(f"Bids must remain valid for {kv('bid_validity_days')} days.")

        # Fees and bonds
        if has("tender_doc_fee_kd"):
            sentences.append(f"Tender documents cost K.D. {kv('tender_doc_fee_kd')}.")
        if has("initial_bond_value_kd"):
            sentences.append(f"The initial bond is K.D. {kv('initial_bond_value_kd')}.")
        if has("initial_bond_percent"):
            sentences.append(f"The initial bond is {kv('initial_bond_percent')}%.")

        # Alternatives and indivisibility
        if "alternative_offers_allowed" in a:
            allowed = bool(a.get("alternative_offers_allowed"))
            sentences.append("Alternative offers are permitted." if allowed else "Alternative offers are not permitted.")
        if "indivisible" in a:
            indiv = bool(a.get("indivisible"))
            sentences.append("The tender is indivisible." if indiv else "The tender may be divided.")

        # Registration / duration / law / meeting (when applicable)
        if has("registration_no"):
            sentences.append(f"Registration number {kv('registration_no')}.")
        if has("duration_from") and has("duration_to"):
            sentences.append(f"Duration from {kv('duration_from')} to {kv('duration_to')}.")
        if has("law_no"):
            sentences.append(f"Law No. {kv('law_no')}.")
        if has("article_no"):
            sentences.append(f"Article {kv('article_no')}.")
        if has("meeting_no"):
            sentences.append(f"Meeting No. {kv('meeting_no')}.")

        # If nothing recognized, fall back to key:value lines
        if not sentences:
            parts = []
            for k, v in (a or {}).items():
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
            return "\n".join(parts) if parts else ""

        # Build paragraph + provenance
        paragraph = " ".join(sentences).strip()
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
                return paragraph + "\n" + ("Provenance: " + "; ".join(prov_bits))
        return paragraph
    except Exception:
        # On any formatting error, degrade gracefully to old behaviour
        parts = []
        for k, v in (answers or {}).items():
            parts.append(f"{k}: {v}")
        return "\n".join(parts) if parts else ""


def query_graph_rag(query_text: str, top_k: int = _DEFAULT_top_k, neighbor_window: int = 1):
    """Query the GraphRAG system with a question."""
    # Card-first gate
    scope_files = None
    gate = card_first_gate(query_text, allow_direct_answer=True)
    mode = gate.get("mode")
    # TODO What is Mode
    if mode == "answer":
        return _format_card_answer(gate.get("answers"), gate.get("provenance"))
    if mode == "restrict":
        scope_files = gate.get("scope_files") or None
        # Incorporate boost terms (anchors, ids, authority) directly into the query to steer retrieval
        boost_terms = gate.get("boost_terms") or []
        if boost_terms:
            query_text = f"{query_text} \n\n" + " ".join(str(t) for t in boost_terms)

    # Light heuristic: widen neighbors for table-like fee/bond rows
    qt = (query_text or "").lower()
    
    # TO DO: Evaluate this
    table_cues = [
        "table",
        "fee",
        "document",
        "price",
        "cost",
        "bond",
        "%",
        "5dx",
        "agency",
        "registration",
        "practice",
    ]
    table_like = any(k in qt for k in table_cues) or "جدول" in qt # جدول  = table
    if table_like and neighbor_window < 2:
        neighbor_window = 2

    rag = _make_rag(neighbor_window, scope_files=scope_files)
    response = rag.search(query_text=query_text, retriever_config={"top_k": top_k})
    return response.answer