import os
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from dotenv import load_dotenv

from .AnchorUtils import extract_anchors, wants_slots, norm_digits
from .Citations import build_references, format_with_citations
from .GetNeighbor import wrap_with_neighbors
from .HybridRAG import HybridRetrievalPipeline, HybridRetrievalConfig  # type: ignore[import]
from .Reranker import CohereReranker  # type: ignore[import]
from .UseCatalog import load_catalog, card_first_gate
from .normalize import normalize_answer_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
if not NEO4J_URI:
    raise RuntimeError("NEO4J_URI must be set for GraphRAG pipeline")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or None
VECTOR_INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX", "Chunk")
FULLTEXT_INDEX_NAME = os.getenv("NEO4J_FULLTEXT_INDEX")
if not FULLTEXT_INDEX_NAME:
    raise RuntimeError("NEO4J_FULLTEXT_INDEX must be set for hybrid retrieval")


auth_tuple = None
if NEO4J_USERNAME:
    auth_tuple = (NEO4J_USERNAME, NEO4J_PASSWORD or "")
driver = GraphDatabase.driver(NEO4J_URI, auth=auth_tuple)

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
embedder = OpenAIEmbeddings(model=EMBED_MODEL)

LLM_MODEL = os.getenv("OPENAI_MODEL")
temperature = 1 if (LLM_MODEL or "").lower().startswith("gpt-5") else 0.1
llm = OpenAILLM(model_name=LLM_MODEL, model_params={"temperature": temperature})

_MMR_K = int(os.getenv("MMR_k"))
_MMR_LAMBDA = float(os.getenv("MMR_LAMBDA"))
_ALWAYS_KEEP_TOP = int(os.getenv("ALWAYS_KEEP_TOP"))


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


COHERE_ENABLED = _env_bool("COHERE_RERANK_ENABLED", True)
if COHERE_ENABLED:
    try:
        _reranker = CohereReranker()
    except Exception as exc:  # pragma: no cover - fail loud
        raise RuntimeError(
            "Failed to initialise Cohere reranker. Set COHERE_RERANK_ENABLED=0 to disable."
        ) from exc
else:
    _reranker = None

HYBRID_CONFIG = HybridRetrievalConfig(
    vector_index=VECTOR_INDEX_NAME,
    fulltext_index=FULLTEXT_INDEX_NAME,
    alpha=float(os.getenv("HYBRID_ALPHA", "0.6")),
    pool_multiplier=float(os.getenv("HYBRID_POOL_MULTIPLIER", "4")),
    pool_min=int(os.getenv("HYBRID_POOL_MIN", "80")),
    pool_max=int(os.getenv("HYBRID_POOL_MAX", "300")),
    apply_mmr=_env_bool("HYBRID_APPLY_MMR", True),
    mmr_k=int(os.getenv("HYBRID_MMR_K", str(_MMR_K))),
    mmr_lambda=float(os.getenv("HYBRID_MMR_LAMBDA", str(_MMR_LAMBDA))),
    mmr_keep_top=int(os.getenv("HYBRID_MMR_KEEP_TOP", str(_ALWAYS_KEEP_TOP))),
    neo4j_database=NEO4J_DATABASE,
)

HYBRID_PIPELINE = HybridRetrievalPipeline(
    driver,
    embedder,
    _reranker,
    config=HYBRID_CONFIG,
)

# -----------------------------
# Anchor helpers
# -----------------------------


def _normalize_anchor(a: str) -> str:
    a = norm_digits(a or "").strip()
    a = re.sub(r"\s*([/-])\s*", r"\1", a)
    a = re.sub(r"\s+", " ", a)
    return a.upper()


def _docs_for_anchors(values):
    vals = [_normalize_anchor(v) for v in values if v]
    if not vals:
        return []
    with driver.session(database=NEO4J_DATABASE) as s:
        rows = s.run(
            """
            UNWIND $vals AS v
            MATCH (a:Anchor {value: v})<-[:HAS_ANCHOR]-(d:Document)
            RETURN DISTINCT d.document_key AS dk
            """,
            {"vals": vals},
        ).data()
    return [r["dk"] for r in rows]


# Load catalog once at startup (no-op if already loaded)
try:
    load_catalog()
except Exception:
    pass

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

        if has("closing_date") and has("closing_time"):
            sentences.append(f"Closing time is {kv('closing_time')} on {kv('closing_date')}.")
        elif has("closing_date"):
            sentences.append(f"Closing date is {kv('closing_date')}.")
        elif has("closing_time"):
            sentences.append(f"Closing time is {kv('closing_time')}.")

        if has("bid_validity_days"):
            sentences.append(f"Bids must remain valid for {kv('bid_validity_days')} days.")

        if has("tender_doc_fee_kd"):
            sentences.append(f"Tender documents cost {kv('tender_doc_fee_kd')} KD.")
        if has("initial_bond_value_kd"):
            sentences.append(f"The initial bond is {kv('initial_bond_value_kd')} KD.")
        if has("initial_bond_percent"):
            sentences.append(f"The initial bond is {kv('initial_bond_percent')}%.")

        if "alternative_offers_allowed" in a:
            allowed = bool(a.get("alternative_offers_allowed"))
            sentences.append(
                "Alternative offers are permitted." if allowed else "Alternative offers are not permitted."
            )
        if "indivisible" in a:
            indiv = bool(a.get("indivisible"))
            sentences.append("The tender is indivisible." if indiv else "The tender may be divided.")

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
        parts = []
        for k, v in (answers or {}).items():
            parts.append(f"{k}: {v}")
        return "\n".join(parts) if parts else ""


def query_graph_rag(
    query_text: str,
    top_k: int = _DEFAULT_top_k,
    neighbor_window: int = 1,
    reference_seq_offset: int = 0,
    *,
    run_id: Optional[str] = None,
    callback: Optional[Callable[[Dict[str, object]], None]] = None,
):
    """Query the GraphRAG system with a question."""

    original_question = query_text

    def emit(event_type: str, **payload) -> None:
        if not callback:
            return
        event: Dict[str, object] = {"event": event_type}
        if run_id is not None:
            event["run_id"] = run_id
        event.update(payload)
        callback(event)

    gate = card_first_gate(query_text, allow_direct_answer=True)
    emit("gate", **(gate.get("diagnostics") or {}))
    provenance_refs = build_references(gate.get("provenance"))

    mode = gate.get("mode")
    if mode == "answer":
        emit("card_answer", record_id=gate.get("record_id"), decision="answer")
        base_answer = _format_card_answer(gate.get("answers"), gate.get("provenance"))
        normalized = normalize_answer_text(base_answer, original_question)
        final_answer = format_with_citations(normalized, provenance_refs, seq_offset=reference_seq_offset)
        emit("answer", source="card", fallback=False, retrieved=0)
        return final_answer

    boost_terms = gate.get("boost_terms") or []
    scope_files = None
    if mode == "restrict":
        scope_files = gate.get("scope_files") or None
        if boost_terms:
            query_text = f"{query_text}\n\n" + " ".join(str(t) for t in boost_terms)
    elif boost_terms and _env_bool("ALWAYS_BOOST_ANCHORS", True):
        query_text = f"{query_text}\n\n" + " ".join(str(t) for t in boost_terms)

    anchors_in_q = extract_anchors(query_text)
    if anchors_in_q:
        anchor_scope = _docs_for_anchors(anchors_in_q)
        if anchor_scope:
            scope_files = sorted(set((scope_files or []) + anchor_scope))

    qt = (query_text or "").lower()
    anchors_present = bool(anchors_in_q)
    wants_money, wants_percent, wants_date = wants_slots(query_text)
    wants_time = any(k in qt for k in ["time", "1:", "pm", "am"])

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
    table_like = any(k in qt for k in table_cues) or "جدول" in qt
    if (table_like or (anchors_present and (wants_money or wants_percent or wants_date or wants_time))) and neighbor_window < 2:
        neighbor_window = 2

    class _PipelineAdapter:
        def __init__(self, scope: Optional[Sequence[str]]):
            self.scope = list(scope or [])
            self.last_diagnostics: Dict[str, object] = {}

        def _call(self, query: str, retriever_config: Dict[str, object]):
            cfg_top = retriever_config.get("top_k")
            effective_top = int(cfg_top) if cfg_top is not None else top_k
            if effective_top <= 0:
                effective_top = top_k if top_k > 0 else 1
            result, diag = HYBRID_PIPELINE.retrieve(query, top_k=effective_top, scope=self.scope)
            diag = dict(diag)
            diag["scope"] = list(self.scope)
            diag["top_k"] = effective_top
            self.last_diagnostics = diag
            return result

        def search(self, *args, **kwargs):
            rc = dict(kwargs.pop("retriever_config", {}) or {})
            q = kwargs.get("query_text") or (args[0] if args else "")
            return self._call(q, rc)

        def retrieve(self, *args, **kwargs):
            return self.search(*args, **kwargs)

        def get_search_results(self, *args, **kwargs):
            return self.search(*args, **kwargs)

    def _run(query: str, scope: Optional[Sequence[str]], label: str):
        adapter = _PipelineAdapter(scope)
        retriever = wrap_with_neighbors(adapter, driver, window=neighbor_window)
        rag = GraphRAG(retriever=retriever, llm=llm)
        response_local = rag.search(
            query_text=query,
            retriever_config={"top_k": top_k},
            return_context=True,
        )
        diag = dict(adapter.last_diagnostics)
        diag["label"] = label
        diag["neighbor_window"] = neighbor_window
        emit("retrieval", **diag)
        return response_local

    response = _run(query_text, scope_files, "primary")
    retriever_result = getattr(response, "retriever_result", None)
    fallback_used = False
    if scope_files and (retriever_result is None or len(retriever_result.items) == 0):
        emit("gate_fail_open", reason="empty_scope", scope=scope_files)
        response = _run(query_text, None, "fallback")
        retriever_result = getattr(response, "retriever_result", None)
        fallback_used = True

    retrieval_refs = build_references(getattr(retriever_result, "items", None))
    if not retrieval_refs:
        retrieval_refs = build_references(getattr(response, "context", None))

    combined_refs = []
    for ref in provenance_refs + retrieval_refs:
        if ref not in combined_refs:
            combined_refs.append(ref)

    normalized_answer = normalize_answer_text(getattr(response, "answer", ""), original_question)
    final_answer = format_with_citations(normalized_answer, combined_refs, seq_offset=reference_seq_offset)
    emit(
        "answer",
        source="rag",
        fallback=fallback_used,
        retrieved=len(getattr(retriever_result, "items", []) or []),
    )
    return final_answer
