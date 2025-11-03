import re
from typing import Any, Callable, List, Tuple

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm_digits(s: str) -> str:
    return (s or "").translate(_ARABIC_DIGITS)


def _canon(s: str) -> str:
    """Lowercase, normalize digits, and drop non-alphanumerics.

    Helps match anchors that appear with variable spacing or punctuation,
    such as "A/M/804" vs "A / M / 804".
    """
    import re
    return re.sub(r"[^a-z0-9]", "", norm_digits(s or "").lower())


def _anchor_present(doc_text_norm: str, anchor_text: str) -> bool:
    """Return True if the anchor appears in the document, allowing flexible separators."""
    import re
    # Fast path: exact substring after digit normalization
    a_norm = norm_digits(anchor_text or "").lower()
    if a_norm and a_norm in doc_text_norm:
        return True
    # Canonical path: strip punctuation/whitespace and compare in canonical space
    return _canon(anchor_text) in _canon(doc_text_norm)


def _first_anchor_pos(doc_text_norm: str, anchor_text: str) -> int:
    """Find first index of an anchor in the original normalized doc string.

    Tries exact substring first; if not found, falls back to a regex that allows
    optional spaces around separators like '/' and '-'. If still not found,
    uses canonical containment without a reliable position.
    """
    import re
    a_norm = norm_digits(anchor_text or "").lower()
    if not a_norm:
        return -1
    p = doc_text_norm.find(a_norm)
    if p != -1:
        return p
    # Build a tolerant regex for common separators
    pattern = re.escape(a_norm)
    pattern = pattern.replace("/", r"\s*/\s*").replace("-", r"\s*-\s*")
    m = re.search(pattern, doc_text_norm, flags=re.IGNORECASE)
    if m:
        return m.start()
    # Last resort: canonical containment (position unknown)
    return -1


RX_ANCHORS = [
    re.compile(r"\bRFP[\s/-]?\d+\b", re.IGNORECASE),
    re.compile(r"\b\d{6,7}[\s/-]?RFP\b", re.IGNORECASE),
    re.compile(r"\bA\s*/\s*M\s*/\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bA/M/\d+\b", re.IGNORECASE),
    re.compile(r"\b5D\s*[A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"\b5D[A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"\b20\d{2}\s*/\s*00\d{2,3}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}/\d{5}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}/\d{4}/\d+\b", re.IGNORECASE),
]


def extract_anchors(text: str) -> List[str]:
    t = norm_digits(text or "")
    out: List[str] = []
    for rx in RX_ANCHORS:
        for raw in rx.findall(t):
            cleaned = re.sub(r"\s*/\s*", "/", raw)
            cleaned = re.sub(r"\s*-\s*", "-", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                out.append(cleaned)
    seen = set(); uniq: List[str] = []
    for a in out:
        k = a.upper()
        if k not in seen:
            seen.add(k); uniq.append(a)
    return uniq


RX_MONEY = re.compile(r"(?i)\b(?:kd|k\.?d\.?|دينار)\b.{0,150}?\b\d[\d,]*\b")
RX_BOND = re.compile(r"(?i)(bond|security\s+deposit|initial\s+security|preliminary\s+bond|التأمين|كفالة|ضمان)")
RX_PERCENT = re.compile(r"(?<!\d)\b\d{1,3}%\b")
RX_DATE = re.compile(r"\b\d{4}[-/.]\d{2}[-/.]\d{2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
RX_TIME = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b", re.IGNORECASE)


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


def has_slot_near_anchor(query_text: str, text: str, anchors_norm: List[str], wants: Tuple[bool, bool, bool], window_chars: int = 150) -> bool:
    di = norm_digits(text or "").lower()
    ql = norm_digits(query_text or "").lower()
    bond_in_query = ("bond" in ql) or ("security" in ql) or ("التأمين" in ql) or ("كفالة" in ql) or ("ضمان" in ql)
    wants_time = ("time" in ql) or bool(re.search(r"\b\d{1,2}:\d{2}\b", ql))
    for a in anchors_norm:
        p = _first_anchor_pos(di, a)
        if p == -1:
            continue
        lo = max(0, p - window_chars)
        hi = min(len(di), p + window_chars)
        window = di[lo:hi]
        wm, wp, wd = wants
        money_ok = RX_MONEY.search(window) is not None
        if bond_in_query and wm:
            money_ok = money_ok and (RX_BOND.search(window) is not None)
        time_ok = RX_TIME.search(window) is not None if wants_time else False
        if (wm and money_ok) or (wp and RX_PERCENT.search(window)) or (wd and RX_DATE.search(window)) or time_ok:
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
        if any(_anchor_present(di, a) for a in anchors_norm):
            out.append(r)
    # Soft gating: if none of the current results contain anchors, keep originals
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
        if has_slot_near_anchor(query_text, get_text(r), anchors_norm, wants):
            return selected_results
    for r in original_results:
        if has_slot_near_anchor(query_text, get_text(r), anchors_norm, wants):
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
    wants_time = ("time" in ql) or bool(re.search(r"\b\d{1,2}:\d{2}\b", ql))

    date_pattern = re.compile(r"\[date=([0-9]{4})-([0-9]{2})-([0-9]{2})\]", re.IGNORECASE)
    supp_pattern = re.compile(r"\[supp=([0-9]+)\]", re.IGNORECASE)

    def _header_info(idx: int):
        raw = get_text(results[idx]) or ""
        first_line = raw.splitlines()[0] if raw else ""
        date_match = date_pattern.search(first_line)
        if date_match:
            year, month, day = map(int, date_match.groups())
            date_tuple = (year, month, day)
        else:
            date_tuple = (0, 0, 0)
        supp_match = supp_pattern.search(first_line)
        supp_idx = int(supp_match.group(1)) if supp_match else 0
        anchor_key = None
        di = docs_norm[idx]
        for a in anchors_norm:
            if a and _anchor_present(di, a):
                anchor_key = a
                break
        return date_tuple, supp_idx, anchor_key

    header_meta = [_header_info(i) for i in range(len(results))]

    base_scores = []

    def score(i: int) -> int:
        di = docs_norm[i]
        s = 0
        hit_pos = -1
        for a in anchors_norm:
            if not a:
                continue
            p = _first_anchor_pos(di, a)
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
            if wants_time and RX_TIME.search(window):
                s += 3
        return s

    for idx in range(len(results)):
        base_scores.append(score(idx))

    boosts = [0] * len(results)

    clusters = {}
    for idx, (_, _, anchor_key) in enumerate(header_meta):
        if anchor_key:
            clusters.setdefault(anchor_key, []).append(idx)

    for idx, (_, supp_idx, _) in enumerate(header_meta):
        if supp_idx > 0:
            boosts[idx] += 1

    for idx_list in clusters.values():
        if len(idx_list) < 2:
            continue
        latest_date = max(header_meta[i][0] for i in idx_list)
        for i in idx_list:
            if header_meta[i][0] == latest_date:
                boosts[i] += 2

    idxs = list(range(len(results)))
    idxs.sort(key=lambda i: (-(base_scores[i] + boosts[i]), i))
    return [results[i] for i in idxs]


