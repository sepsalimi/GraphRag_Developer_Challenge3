from typing import Any

from neo4j_graphrag.retrievers import VectorRetriever as _VectorRetriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResult

from .CheckUpdates import gather_update_snippets


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
    return None


def _set_text(obj, new_text):
    if hasattr(obj, "text"):
        setattr(obj, "text", new_text)
        return True
    if hasattr(obj, "content"):
        setattr(obj, "content", new_text)
        return True
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
    def _unwrap(self, result_obj: Any):
        if isinstance(result_obj, RetrieverResult):
            metadata = result_obj.metadata
            return list(result_obj.items), lambda items: RetrieverResult(items=items, metadata=metadata)
        if isinstance(result_obj, RawSearchResult):
            metadata = result_obj.metadata
            return list(result_obj.records), lambda items: RawSearchResult(records=items, metadata=metadata)
        if isinstance(result_obj, list):
            return list(result_obj), lambda items: items
        return [], lambda items: result_obj

    def _expand_results(self, results, query_text: str = ""):
        if self._window <= 0:
            return results
        items, rebuild = self._unwrap(results)
        if not items:
            return rebuild(items)
        with self._driver.session() as session:
            for hit in items:
                doc_key = _get_field(hit, "document_key")
                idx = _get_field(hit, "chunk_index")
                if doc_key is None or idx is None:
                    continue
                pub_key = None
                if isinstance(doc_key, str) and ":" in doc_key:
                    pub_key = doc_key.split(":", 1)[0]
                texts = _fetch_neighbor_texts(session, doc_key, idx, self._window)
                if not texts:
                    continue
                combined = "\n".join(texts)
                update_texts = gather_update_snippets(
                    session=session,
                    query_text=query_text,
                    base_text=combined,
                    window=self._window,
                    publication_key=pub_key,
                )
                if update_texts:
                    combined = combined + "\n--- update ---\n" + "\n".join(update_texts)
                _set_text(hit, combined)
        return rebuild(items)

    def search(self, *args, **kwargs):
        base = getattr(self._base, "search", None)
        if base is None and hasattr(self._base, "retrieve"):
            base = getattr(self._base, "retrieve")
        if base is None:
            return []
        query_text = kwargs.get("query_text") or (args[0] if args else "")
        return self._expand_results(base(*args, **kwargs), query_text)

    def retrieve(self, *args, **kwargs):
        base = getattr(self._base, "retrieve", None)
        if base is None and hasattr(self._base, "search"):
            base = getattr(self._base, "search")
        if base is None:
            return []
        query_text = kwargs.get("query_text") or (args[0] if args else "")
        return self._expand_results(base(*args, **kwargs), query_text)

    def get_search_results(self, *args, **kwargs):
        base = getattr(self._base, "get_search_results", None)
        if base is None and hasattr(self._base, "search"):
            base = getattr(self._base, "search")
        if base is None and hasattr(self._base, "retrieve"):
            base = getattr(self._base, "retrieve")
        if base is None:
            return []
        query_text = kwargs.get("query_text") or (args[0] if args else "")
        return self._expand_results(base(*args, **kwargs), query_text)


def wrap_with_neighbors(base_retriever, neo4j_driver, window=1):
    try:
        win = int(window)
    except Exception:
        win = 1
    if win <= 0:
        return base_retriever
    return NeighborWindowRetriever(base_retriever, neo4j_driver, win)


