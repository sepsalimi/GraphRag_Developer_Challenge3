import re
from typing import List, Optional

from neo4j import Driver

from RAG.AnchorUtils import norm_digits, extract_anchors, RX_MONEY, RX_PERCENT, RX_DATE


def extractive_answer(driver: Driver, query_text: str, scope_files: Optional[List[str]] = None) -> Optional[str]:
    anchors = extract_anchors(query_text)
    if not anchors:
        return None
    ql = (norm_digits(query_text or "").lower())
    wants_money = any(k in ql for k in ["fee", "document", "price", "cost", "bond", "kd", "k.d", "دينار"])
    wants_percent = ("%" in ql) or ("percent" in ql) or ("percentage" in ql) or ("٪" in ql)
    wants_date = any(k in ql for k in ["date", "closing", "deadline", "submit", "آخر موعد"])
    if not (wants_money or wants_percent or wants_date):
        return None

    an = [norm_digits(a).lower() for a in anchors]
    params = {"anchors": an, "scope": scope_files or []}
    cypher = (
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE any(a IN $anchors WHERE toLower(c.text) CONTAINS a)
          AND (size($scope)=0 OR d.document_key IN $scope)
        RETURN c.text AS t
        LIMIT 12
        """
    )
    with driver.session() as s:
        rows = s.run(cypher, **params).data()
    for row in rows:
        di = norm_digits(row.get("t") or "")
        dil = di.lower()
        for a in an:
            p = dil.find(a)
            if p == -1:
                continue
            lo = max(0, p - 180)
            hi = min(len(di), p + 180)
            window = di[lo:hi]
            if wants_money:
                m = RX_MONEY.search(window)
                if m:
                    return re.sub(r"(?i)\b(?:KWD|KD|K\.?D\.?)\b", "K.D.", m.group(0))
            if wants_percent:
                m = RX_PERCENT.search(window)
                if m:
                    return m.group(0)
            if wants_date:
                m = RX_DATE.search(window)
                if m:
                    txt = m.group(0)
                    mm = re.fullmatch(r"(\d{4})[-/.](\d{2})[-/.](\d{2})", txt)
                    if mm:
                        return f"{mm.group(1)}-{mm.group(2)}-{mm.group(3)}"
                    mm = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", txt)
                    if mm:
                        y = mm.group(3)
                        y = ("20" + y) if len(y) == 2 else y
                        return f"{y}-{int(mm.group(2)):02d}-{int(mm.group(1)):02d}"
                    return txt
    return None


