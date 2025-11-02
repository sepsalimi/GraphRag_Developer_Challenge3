import os
from neo4j_graphrag.retrievers import VectorRetriever as _VectorRetriever


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


def _set_text(obj, new_text):
    try:
        if hasattr(obj, "text"):
            setattr(obj, "text", new_text)
            return True
    except Exception:
        pass
    if isinstance(obj, dict):
        obj["text"] = new_text
        return True
    return False


def _fetch_neighbor_texts(session, document_key, center_index, window):
    result = session.run(
        """
        MATCH (n:Chunk {document_key: $doc})
        WHERE n.chunk_index >= $start AND n.chunk_index <= $end
        RETURN n.chunk_index AS idx, n.text AS text
        ORDER BY idx ASC
        """,
        doc=document_key,
        start=max(0, int(center_index) - int(window)),
        end=int(center_index) + int(window),
    )
    rows = list(result)
    return [r["text"] for r in rows]


class NeighborWindowRetriever(_VectorRetriever):
    def __init__(self, base_retriever, neo4j_driver, window):
        self._base = base_retriever
        self._driver = neo4j_driver
        self._window = int(window or 0)

    def __getattr__(self, item):
        return getattr(self._base, item)

    def _expand_results(self, results):
        if not results or self._window <= 0:
            return results
        try:
            with self._driver.session() as session:
                for hit in results:
                    doc_key = _get_field(hit, "document_key")
                    idx = _get_field(hit, "chunk_index")
                    if doc_key is None or idx is None:
                        continue
                    texts = _fetch_neighbor_texts(session, doc_key, idx, self._window)
                    if not texts:
                        continue
                    combined = "\n".join(texts)
                    _set_text(hit, combined)
        except Exception:
            return results
        return results

    def search(self, *args, **kwargs):
        if hasattr(self._base, "search"):
            results = self._base.search(*args, **kwargs)
        elif hasattr(self._base, "retrieve"):
            results = self._base.retrieve(*args, **kwargs)
        else:
            return []
        return self._expand_results(results)

    def retrieve(self, *args, **kwargs):
        if hasattr(self._base, "retrieve"):
            results = self._base.retrieve(*args, **kwargs)
        elif hasattr(self._base, "search"):
            results = self._base.search(*args, **kwargs)
        else:
            return []
        return self._expand_results(results)


def wrap_with_neighbors(base_retriever, neo4j_driver, window=1):
    try:
        win = int(window)
    except Exception:
        win = 1
    if win <= 0:
        return base_retriever
    return NeighborWindowRetriever(base_retriever, neo4j_driver, win)


