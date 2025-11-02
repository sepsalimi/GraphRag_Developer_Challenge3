from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------------
# Minimal metrics (A/B logging)
# -----------------------------
card_direct_hits: int = 0
card_prefilter_uses: int = 0
card_misses: int = 0


# -----------------------------
# In-memory catalog handle
# -----------------------------
class CatalogHandle:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.by_id: Dict[str, Dict[str, Any]] = {}
        self.aliases: Dict[str, Set[str]] = {}
        self.id_to_sources: Dict[str, Set[str]] = {}


_CATALOG: Optional[CatalogHandle] = None
_DEFAULT_CATALOG_PATH = "RAG/card_catalog.jsonl"


# -----------------------------
# Helpers
# -----------------------------
_WS = re.compile(r"\s+")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _norm_digits(s: str) -> str:
    return s.translate(_ARABIC_DIGITS)


def _norm_key(s: str) -> str:
    return _WS.sub(" ", _norm_digits(str(s).strip())).upper()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    v = str(val).strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def _get_sources(rec: Dict[str, Any]) -> Set[str]:
    sources: Set[str] = set()
    for pv in rec.get("provenance", []) or []:
        src = pv.get("source")
        if src:
            sources.add(str(src))
    return sources


def _add_alias(c: CatalogHandle, alias: Optional[str], record_id: str) -> None:
    if not alias:
        return
    key = _norm_key(alias)
    c.aliases.setdefault(key, set()).add(record_id)


# -----------------------------
# Regex extractors (anchored IDs)
# -----------------------------
_RX_TENDER = [
    re.compile(r"\bRFP-\d+\b", re.IGNORECASE),
    re.compile(r"\b\d{6,7}-RFP\b", re.IGNORECASE),
    re.compile(r"\b\d{4}/\d{4}/\d+\b", re.IGNORECASE),  # CAPT tender style
]

_RX_PRACTICE = [
    re.compile(r"\bA/M/\d+\b", re.IGNORECASE),
    re.compile(r"\b\d-\d{4}/\d{4}\b", re.IGNORECASE),
    re.compile(r"\b5D[A-Z0-9]+\b", re.IGNORECASE),
]

_RX_REGISTRATION = [
    re.compile(r"\b\d{4}/\d{5}\b", re.IGNORECASE),
]

_RX_LAW_WITH_WORDING = [
    re.compile(r"\b(?:DECREE-?LAW|LAW)\s*(?:NO\.?\s*)?\d+/?\d{4}\b", re.IGNORECASE),
]

_RX_LAW_COMPACT = [
    re.compile(r"\b\d+/\d{4}\b", re.IGNORECASE),
]

_RX_MEETING = [
    re.compile(r"\bMEETING (?:MINUTES )?NO\.?\s*\d{4}/\d+\b", re.IGNORECASE),
]


def _extract_anchors(question: str) -> List[str]:
    q = _norm_digits(question or "")
    anchors: List[str] = []

    def collect(rx_list: List[re.Pattern[str]]) -> None:
        for rx in rx_list:
            for m in rx.findall(q):
                anchors.append(str(m))

    collect(_RX_TENDER)
    collect(_RX_PRACTICE)
    collect(_RX_REGISTRATION)
    collect(_RX_MEETING)

    # Laws with wording (direct)
    collect(_RX_LAW_WITH_WORDING)

    # Compact law like 61/2025, only if law/decree wording is present somewhere
    if re.search(r"\b(?:law|decree)\b", q, re.IGNORECASE):
        collect(_RX_LAW_COMPACT)

    # Normalize and dedupe while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for a in anchors:
        k = _norm_key(a)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq


def _question_type_hint(question: str) -> Optional[str]:
    q = _norm_digits(question or "").lower()
    if any(w in q for w in ["tender", "rfp", "مناقصة", "طرح", "عطاء"]):
        return "tender"
    if any(w in q for w in ["practice", "ممارسة"]):
        return "practice"
    if any(w in q for w in ["registration", "agency"]):
        return "registration"
    if any(w in q for w in ["decree", "law", "article", "قانون", "مرسوم", "مادة"]):
        return "law"
    if any(w in q for w in ["capt", "meeting", "minutes", "اجتماع", "محضر"]):
        return "meeting"
    if "correction" in q:
        return "correction"
    return None


def _build_aliases_from_record_id(record_id: str) -> List[str]:
    # Extract tender/practice-like codes from the record_id itself
    aliases: List[str] = []
    rid = record_id or ""
    for rx in (_RX_TENDER + _RX_PRACTICE + _RX_REGISTRATION + _RX_MEETING + _RX_LAW_WITH_WORDING + _RX_LAW_COMPACT):
        for m in rx.findall(rid):
            aliases.append(str(m))
    # Dedupe (case-insensitive)
    seen: Set[str] = set()
    out: List[str] = []
    for a in aliases:
        k = _norm_key(a)
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


# -----------------------------
# Field intent detection
# -----------------------------
_INTENT_MAP: List[Tuple[List[str], List[str]]] = [
    (["closing date", "deadline", "due date"], ["closing_date"]),
    (["closing time", "submission time"], ["closing_time"]),
    (["bid validity", "valid for"], ["bid_validity_days"]),
    (["initial bond", "preliminary bond", "guarantee"], ["initial_bond_value_kd", "initial_bond_percent"]),
    (["document fee", "tender fee", "purchase the documents"], ["tender_doc_fee_kd"]),
    (["alternative offers"], ["alternative_offers_allowed"]),
    (["indivisible", "not divisible"], ["indivisible"]),
    (["objection", "objections", "submit an objection"], ["objection_window_days", "objection_destination"]),
    (["registration number"], ["registration_no"]),
    (["duration", "from", "to"], ["duration_from", "duration_to"]),
    (["meeting number", "minutes no"], ["meeting_no"]),
    (["deadline days"], ["deadline_days"]),
    (["law", "decree", "article", "extend by"], ["law_no", "article_no", "extended_by_years"]),
]


def _detect_intents(question: str) -> List[str]:
    q = (question or "").lower()
    requested: List[str] = []
    for keywords, fields in _INTENT_MAP:
        if all(k in q for k in keywords) or any(k in q for k in keywords):
            for f in fields:
                if f not in requested:
                    requested.append(f)
    return requested


def _is_summarize_like(question: str) -> bool:
    q = (question or "").lower()
    return any(w in q for w in ["summarize", "explain", "details", "table", "overview", "describe", "full text"])


# -----------------------------
# Public API
# -----------------------------
def load_catalog(path: str = _DEFAULT_CATALOG_PATH) -> CatalogHandle:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    handle = CatalogHandle()

    p = Path(path)
    if not p.exists():
        # try relative to project root (two levels up from this file)
        root = Path(__file__).resolve().parents[1]
        p = (root / path)

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            handle.records.append(rec)

    for rec in handle.records:
        record_id = str(rec.get("record_id") or "").strip()
        if not record_id:
            continue
        key = _norm_key(record_id)
        handle.by_id[key] = rec

        # provenance sources
        handle.id_to_sources[record_id] = _get_sources(rec)

        # aliases from core secondary ids
        core = rec.get("core", {}) or {}
        _add_alias(handle, core.get("practice_no"), record_id)
        _add_alias(handle, core.get("registration_no"), record_id)
        _add_alias(handle, core.get("law_no"), record_id)
        _add_alias(handle, core.get("meeting_no"), record_id)

        # aliases derived from record_id itself
        for a in _build_aliases_from_record_id(record_id):
            _add_alias(handle, a, record_id)

    _CATALOG = handle
    return handle


def _match_records(question: str, catalog: CatalogHandle) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Return (matched_record_ids, record_map). If multiple, may be >1."""
    anchors = _extract_anchors(question)
    if not anchors:
        return [], {}

    matched_ids: List[str] = []
    record_map: Dict[str, Dict[str, Any]] = {}

    # Try exact record_id first, then aliases
    for a in anchors:
        a_key = _norm_key(a)
        rec = catalog.by_id.get(a_key)
        if rec is not None:
            rid = rec.get("record_id")
            if rid and rid not in record_map:
                record_map[rid] = rec
                matched_ids.append(rid)
            continue

        rid_set = catalog.aliases.get(a_key)
        if rid_set:
            for rid2 in rid_set:
                rec2 = catalog.by_id.get(_norm_key(rid2))
                if rec2 is not None and rid2 not in record_map:
                    record_map[rid2] = rec2
                    matched_ids.append(rid2)

    # If multiples, try to prefer by type hint from question; otherwise keep all
    if len(matched_ids) > 1:
        hint = _question_type_hint(question)
        if hint:
            for rid in matched_ids:
                rt = str(record_map[rid].get("record_type") or "").lower()
                if hint in rt:
                    return [rid], {rid: record_map[rid]}

    return matched_ids, record_map


def _collect_answers(rec: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    core = rec.get("core", {}) or {}
    out: Dict[str, Any] = {}
    for f in fields:
        if f in core and core[f] not in (None, "", []):
            out[f] = core[f]
    return out


def _has_conflict_or_uncertain(rec: Dict[str, Any]) -> bool:
    status = str(rec.get("status") or "").lower()
    return status in {"conflict", "uncertain", "absent"}


def _anchor_matches_record(question: str, rec: Dict[str, Any]) -> bool:
    anchors = _extract_anchors(question)
    if not anchors:
        return False
    
    core = rec.get("core", {}) or {}
    record_ids = [
        rec.get("record_id"),
        core.get("practice_no"),
        core.get("registration_no"),
        core.get("law_no"),
        core.get("meeting_no"),
    ]
    
    anchor_keys = {_norm_key(a) for a in anchors}
    for rid in record_ids:
        if rid and _norm_key(str(rid)) in anchor_keys:
            return True
    return False


def card_first_gate(question: str, *, allow_direct_answer: bool = True) -> Dict[str, Any]:
    """Return GateDecision: {mode, record_id?, answers?, provenance?, scope_files?, boost_terms?}
    """
    global card_misses, card_direct_hits, card_prefilter_uses

    # Allow runtime override from environment (supports .env loaded by caller)
    gate_on = _env_bool("CARD_FIRST_ENABLED", False)
    if not gate_on:
        return {"mode": "pass", "record_id": None}

    catalog = _CATALOG or load_catalog(_DEFAULT_CATALOG_PATH)

    matched_ids, record_map = _match_records(question, catalog)
    if not matched_ids:
        card_misses += 1
        return {"mode": "pass", "record_id": None}

    # If multiple records match, do not merge; prefer restrict using union of sources
    if len(matched_ids) > 1:
        all_sources: Set[str] = set()
        prov: List[Dict[str, Any]] = []
        for rid in matched_ids:
            rec = record_map[rid]
            for pv in rec.get("provenance", []) or []:
                prov.append({"source": pv.get("source"), "lines": pv.get("lines")})
            all_sources.update(catalog.id_to_sources.get(rid, set()))
        card_prefilter_uses += 1
        return {
            "mode": "restrict",
            "record_id": None,
            "provenance": prov,
            "scope_files": sorted(all_sources),
            "boost_terms": [],
        }

    # Single match
    record_id = matched_ids[0]
    rec = record_map[record_id]
    intents = _detect_intents(question)
    answers = _collect_answers(rec, intents) if intents else {}
    all_fields_available = bool(intents) and len(answers) == len(set(intents))

    # Prepare provenance
    provenance = [{"source": pv.get("source"), "lines": pv.get("lines")} for pv in (rec.get("provenance", []) or [])]

    # Decide answer vs restrict vs pass
    allow_da = allow_direct_answer and _env_bool("ALLOW_DIRECT_ANSWER", False)
    if allow_da and intents and not _has_conflict_or_uncertain(rec):
        # Only direct answer if all requested fields are available and anchor matches record
        if all_fields_available and answers and _anchor_matches_record(question, rec):
            card_direct_hits += 1
            return {
                "mode": "answer",
                "record_id": record_id,
                "answers": answers,
                "provenance": provenance,
            }

    # If summarization-like or missing fields → restrict
    need_restrict = _is_summarize_like(question) or all_fields_available
    if need_restrict:
        sources = sorted(catalog.id_to_sources.get(record_id, set()))
        core = rec.get("core", {}) or {}
        boost_terms: List[str] = []
        for k in ["record_id", "practice_no", "registration_no", "law_no", "title"]:
            v = rec.get(k) if k == "record_id" else core.get(k)
            if v:
                boost_terms.append(str(v))
        card_prefilter_uses += 1
        return {
            "mode": "restrict",
            "record_id": record_id,
            "provenance": provenance,
            "scope_files": sources,
            "boost_terms": boost_terms,
        }

    # Otherwise pass-through
    card_misses += 1
    return {"mode": "pass", "record_id": record_id}


def get_catalog_metrics() -> Dict[str, int]:
    return {
        "card_direct_hits": card_direct_hits,
        "card_prefilter_uses": card_prefilter_uses,
        "card_misses": card_misses,
    }


