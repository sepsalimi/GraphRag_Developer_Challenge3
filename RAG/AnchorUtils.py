import re
from typing import Any, Callable, List, Tuple

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm_digits(s: str) -> str:
    return (s or "").translate(_ARABIC_DIGITS)


RX_ANCHORS = [
    re.compile(r"\bRFP[\s/-]?\d+\b", re.IGNORECASE),
    re.compile(r"\b\d{6,7}[\s/-]?RFP\b", re.IGNORECASE),
    re.compile(r"\bA/M/\d+\b", re.IGNORECASE),
    re.compile(r"\b5D[A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"\b\d{4}/\d{5}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}/\d{4}/\d+\b", re.IGNORECASE),
]


def extract_anchors(text: str) -> List[str]:
    t = norm_digits(text or "")
    out: List[str] = []
    for rx in RX_ANCHORS:
        out.extend(rx.findall(t))
    seen = set(); uniq: List[str] = []
    for a in out:
        k = a.upper()
        if k not in seen:
            seen.add(k); uniq.append(a)
    return uniq


RX_MONEY = re.compile(r"(?i)\b(?:kd|k\.?d\.?|دينار)\b.{0,150}?\b\d[\d,]*\b")
RX_PERCENT = re.compile(r"(?<!\d)\b\d{1,3}%\b")
RX_DATE = re.compile(r"\b\d{4}[-/.]\d{2}[-/.]\d{2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")


def wants_slots(query_text: str) -> Tuple[bool, bool, bool]:
    ql = (norm_digits(query_text or "").lower())
    wants_money = any(k in ql for k in ["fee", "document", "price", "cost", "bond", "kd", "k.d", "دينار"]) or bool(
        re.search(r"\b\d[\d,]*\s*(?:kd|k\.?d\.?)\b", ql)
    )
    wants_percent = ("%" in ql) or ("percent" in ql) or ("percentage" in ql) or ("٪" in ql)
    wants_date = any(k in ql for k in ["date", "closing", "deadline", "submit", "آخر موعد"]) or bool(
        re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", ql)
    )
    return wants_money, wants_percent, wants_date


def has_slot_near_anchor(text: str, anchors_norm: List[str], wants: Tuple[bool, bool, bool], window_chars: int = 180) -> bool:
    di = norm_digits(text or "").lower()
    for a in anchors_norm:
        p = di.find(a)
        if p == -1:
            continue
        lo = max(0, p - window_chars)
        hi = min(len(di), p + window_chars)
        window = di[lo:hi]
        wm, wp, wd = wants
        if (wm and RX_MONEY.search(window)) or (wp and RX_PERCENT.search(window)) or (wd and RX_DATE.search(window)):
            return True
    return False


def filter_to_anchor_hits(query_text: str, results: List[Any], get_text: Callable[[Any], str]) -> List[Any]:
    anchors = extract_anchors(query_text)
    if not anchors or not results:
        return results
    anchors_norm = [norm_digits(a).lower() for a in anchors]
    out = []
    for r in results:
        di = norm_digits(get_text(r)).lower()
        if any(a in di for a in anchors_norm):
            out.append(r)
    return out or results


def enforce_slot_guard(query_text: str, original_results: List[Any], selected_results: List[Any], get_text: Callable[[Any], str]) -> List[Any]:
    anchors = extract_anchors(query_text)
    if not anchors:
        return selected_results
    wants = wants_slots(query_text)
    if not any(wants):
        return selected_results
    anchors_norm = [norm_digits(a).lower() for a in anchors]
    for r in selected_results:
        if has_slot_near_anchor(get_text(r), anchors_norm, wants):
            return selected_results
    for r in original_results:
        if has_slot_near_anchor(get_text(r), anchors_norm, wants):
            out = list(selected_results)
            if r in out:
                return out
            if out:
                out[-1] = r
            else:
                out = [r]
            return out
    return selected_results


def preboost_results_by_anchors(query_text: str, results: List[Any], get_text: Callable[[Any], str]) -> List[Any]:
    if not results:
        return results
    anchors = extract_anchors(query_text)
    if not anchors:
        return results
    anchors_norm = [norm_digits(a).lower() for a in anchors]
    docs_norm = [norm_digits(get_text(r)).lower() for r in results]

    ql = (norm_digits(query_text or "").lower())
    wants_money, wants_percent, wants_date = wants_slots(ql)

    def score(i: int) -> int:
        di = docs_norm[i]
        s = 0
        hit_pos = -1
        for a in anchors_norm:
            if not a:
                continue
            p = di.find(a)
            if p != -1:
                s += 2
                if hit_pos == -1 or p < hit_pos:
                    hit_pos = p
        if hit_pos != -1:
            lo = max(0, hit_pos - 180)
            hi = min(len(di), hit_pos + 180)
            window = di[lo:hi]
            if wants_money and RX_MONEY.search(window):
                s += 3
            if wants_percent and RX_PERCENT.search(window):
                s += 3
            if wants_date and RX_DATE.search(window):
                s += 3
        return s

    idxs = list(range(len(results)))
    idxs.sort(key=lambda i: (-score(i), i))
    return [results[i] for i in idxs]


