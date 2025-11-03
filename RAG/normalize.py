"""Answer text normalization helpers shared across GraphRAG entry points."""

from __future__ import annotations

import re
from typing import Optional


_AMOUNT_PATTERN = r"\d[\d,]*(?:\.\d+)?"
_CURRENCY_TOKEN_PATTERNS = [
    r"k\.?\s*d\.?",
    r"kwd",
    r"usd",
    r"us\$",
    r"\$",
    r"eur",
    r"\u20ac",
    r"gbp",
    r"\u00a3",
    r"aed",
    r"sar",
    r"qar",
    r"jod",
    r"\u062f\u064a\u0646\u0627\u0631(?:\s+\u0643\u0648\u064a\u062a\u064a)?",
    r"\u0643\.?\s*\u062f",
]
_CURRENCY_TOKEN_PATTERN = "(" + "|".join(_CURRENCY_TOKEN_PATTERNS) + ")"
_CURRENCY_PREFIX_RE = re.compile(
    rf"(?i)(?<!\w)(?P<currency>{_CURRENCY_TOKEN_PATTERN})(?:\s*[:=]?)\s*(?P<amount>{_AMOUNT_PATTERN})"
)
_CURRENCY_SUFFIX_RE = re.compile(
    rf"(?i)(?P<amount>{_AMOUNT_PATTERN})\s*(?:[:=]?\s*)?(?P<currency>{_CURRENCY_TOKEN_PATTERN})(?!\w)"
)

_PRICE_WORDS = [
    "fee",
    "fees",
    "cost",
    "price",
    "document",
    "bond",
    "guarantee",
    "charge",
    "tariff",
]

_QUESTION_HINTS = {
    "KD": [
        "kd",
        "kwd",
        "kuwaiti dinar",
        "dinar",
        "\u062f\u064a\u0646\u0627\u0631",
        "\u0643\u0648\u064a\u062a",
        "\u0643.\u062f",
        "\u0643 \u062f",
    ],
    "USD": ["usd", "us$", "dollar", "dollars", "$"],
    "EUR": ["eur", "euro", "euros", "\u20ac"],
    "GBP": ["gbp", "pound", "pounds", "\u00a3"],
    "AED": ["aed", "dirham"],
    "SAR": ["sar", "saudi riyal"],
    "QAR": ["qar", "qatari riyal"],
    "JOD": ["jod", "jordanian dinar"],
}


def _currency_code(token: str) -> Optional[str]:
    raw = (token or "").strip()
    if not raw:
        return None
    if "\u062f\u064a\u0646\u0627\u0631" in raw:
        return "KD"
    cleaned = re.sub(r"[.\s]", "", raw.lower())
    if cleaned in {"kd", "kwd", "كد"}:
        return "KD"
    if raw == "$" or cleaned in {"$", "usd", "us$"}:
        return "USD"
    if raw == "€" or cleaned == "eur":
        return "EUR"
    if raw == "£" or cleaned == "gbp":
        return "GBP"
    if cleaned == "aed":
        return "AED"
    if cleaned == "sar":
        return "SAR"
    if cleaned == "qar":
        return "QAR"
    if cleaned == "jod":
        return "JOD"
    return None


def _normalize_currency_tokens(text: str) -> str:
    def _replace_prefix(match: re.Match) -> str:
        code = _currency_code(match.group("currency"))
        amount = match.group("amount").strip()
        return f"{amount} {code}" if code else match.group(0)

    def _replace_suffix(match: re.Match) -> str:
        code = _currency_code(match.group("currency"))
        amount = match.group("amount").strip()
        return f"{amount} {code}" if code else match.group(0)

    updated = _CURRENCY_PREFIX_RE.sub(_replace_prefix, text)
    updated = _CURRENCY_SUFFIX_RE.sub(_replace_suffix, updated)

    for code in ("KD", "USD", "EUR", "GBP", "AED", "SAR", "QAR", "JOD"):
        updated = re.sub(rf"\b{code}\.(?=\s|\d|$)", code, updated)
    return updated


def _question_mentions_money(question_lower: str) -> bool:
    return any(word in question_lower for word in _PRICE_WORDS)


def _infer_currency_code(question_lower: str) -> Optional[str]:
    for code, hints in _QUESTION_HINTS.items():
        for hint in hints:
            if hint in question_lower:
                return code
    return None


def _normalize_days(text: str, question_lower: str) -> str:
    if re.fullmatch(r"\d+", text) and any(
        token in question_lower for token in ["valid", "validity", "period", "days", "how long"]
    ):
        return f"{text} days"
    return re.sub(r"\b(\d+)\s*day\b", r"\1 days", text, flags=re.IGNORECASE)


def _to_iso_date(txt: str) -> str:
    t = txt.strip()
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)
    m = re.fullmatch(r"(\d{4})[-/.](\d{2})[-/.](\d{2})", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.fullmatch(r"(\d{2})[-/](\d{2})[-/](\d{4})", t)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    month_lookup = {name: f"{idx:02d}" for idx, name in enumerate(months, start=1)}
    m = re.fullmatch(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),\s*(\d{4})", t, flags=re.IGNORECASE)
    if m:
        return f"{m.group(3)}-{month_lookup[m.group(1).lower()]}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", t, flags=re.IGNORECASE)
    if m:
        return f"{m.group(3)}-{month_lookup[m.group(2).lower()]}-{int(m.group(1)):02d}"
    return txt


def _normalize_dates(text: str, question_lower: str) -> str:
    if any(token in question_lower for token in ["date", "closing", "deadline"]):
        return _to_iso_date(text)
    return text


def _ensure_general_court_arabic(text: str) -> str:
    if "المحكمة الكلية" in text:
        return text
    return re.sub(
        r"General Court(?:\s*\(Court of First Instance\))?",
        "General Court (المحكمة الكلية)",
        text,
        flags=re.IGNORECASE,
    )


def normalize_answer_text(answer: str, question: str) -> str:
    """Normalize answer text for consistent currency, dates, and terminology."""

    if not isinstance(answer, str):
        return answer

    text = answer.strip()
    if not text:
        return text

    question_lower = (question or "").lower()
    normalized = _normalize_currency_tokens(text)

    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", normalized) and _question_mentions_money(question_lower):
        inferred = _infer_currency_code(question_lower) or "KD"
        normalized = f"{normalized} {inferred}"

    normalized = _normalize_days(normalized, question_lower)
    normalized = _normalize_dates(normalized, question_lower)
    normalized = _ensure_general_court_arabic(normalized)
    return normalized

