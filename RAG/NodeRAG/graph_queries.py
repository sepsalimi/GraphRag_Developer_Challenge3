"""Query helpers that replicate NodeRAG's search pipeline (with a pure-Python HNSW fallback)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Sequence

import numpy as np

from KnowledgeGraph.NodeRag import noderag_runner, load_cached_outputs
from NodeRAG.search.Answer_base import Answer, Retrieval
from NodeRAG.storage.graph_mapping import Mapper
from NodeRAG.storage.storage import storage
from NodeRAG.utils.PPR import sparse_PPR


@dataclass(frozen=True)
class _MetadataState:
    semantic_to_text: Dict[str, str]
    text_lookup: Dict[str, Dict[str, Any]]
    doc_lookup: Dict[str, Dict[str, Any]]


@lru_cache(maxsize=1)
def _metadata_state() -> _MetadataState:
    """Load lightweight lookup tables for semantic units → text units → documents."""
    outputs = load_cached_outputs()

    semantic_map: Dict[str, str] = {}
    if not outputs.semantic_units.empty:
        sem_df = outputs.semantic_units.copy()
        sem_df["hash_id"] = sem_df["hash_id"].astype(str)
        for record in sem_df.to_dict("records"):
            hash_id = record.get("hash_id")
            text_hash = record.get("text_hash_id")
            if hash_id:
                semantic_map[str(hash_id)] = str(text_hash) if text_hash else ""

    text_lookup: Dict[str, Dict[str, Any]] = {}
    if not outputs.text_units.empty:
        text_df = outputs.text_units.copy()
        text_df["hash_id"] = text_df["hash_id"].astype(str)
        text_lookup = {record["hash_id"]: record for record in text_df.to_dict("records")}

    doc_lookup: Dict[str, Dict[str, Any]] = {}
    if not outputs.documents.empty and "doc_hash_id" in outputs.documents.columns:
        doc_df = outputs.documents.copy()
        doc_df["doc_hash_id"] = doc_df["doc_hash_id"].astype(str)
        doc_lookup = {record["doc_hash_id"]: record for record in doc_df.to_dict("records")}

    return _MetadataState(semantic_map, text_lookup, doc_lookup)


class SimpleHNSW:
    """Minimal stand-in for NodeRAG's HNSW index (cosine similarity over cached embeddings)."""

    def __init__(self, config) -> None:
        self.config = config
        self.ids, self.vectors = self._load_vectors()

    def _load_vectors(self) -> tuple[list[str], np.ndarray]:
        df = storage.load(self.config.embedding)
        ids: List[str] = []
        vectors: List[np.ndarray] = []
        for record in df.to_dict("records"):
            hash_id = record.get("hash_id")
            embedding = record.get("embedding")
            if not hash_id or embedding is None or (isinstance(embedding, str) and embedding == "done"):
                continue
            arr = np.asarray(embedding, dtype=np.float32)
            if arr.size == 0:
                continue
            ids.append(str(hash_id))
            vectors.append(arr)
        if not ids:
            raise RuntimeError("NodeRAG embedding cache is empty; rerun the embedding pipeline.")
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        return ids, matrix

    def search(self, query_embedding: np.ndarray, HNSW_results: int | None = None):
        if HNSW_results is None:
            HNSW_results = self.config.top_k
        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(query)
        if norm:
            query = query / norm
        sims = self.vectors @ query
        count = min(HNSW_results, len(sims))
        indices = np.argpartition(sims, -count)[-count:]
        indices = indices[np.argsort(sims[indices])[::-1]]
        return [(float(1 - sims[idx]), self.ids[idx]) for idx in indices]


class LiteNodeSearch:
    """NodeRAG search with a simple cosine/PPR fallback in place of the missing compiled index."""

    def __init__(self, config) -> None:
        self.config = config
        self.mapper = self._load_mapper()
        self.hnsw = SimpleHNSW(config)
        self.G = self._load_graph()
        self.id_to_type = {node_id: self.G.nodes[node_id].get("type") for node_id in self.G.nodes}
        self.id_to_text, self.accurate_id_to_text = self.mapper.generate_id_to_text(
            ["entity", "high_level_element_title"]
        )
        self.sparse_PPR = sparse_PPR(self.G)

    def _load_mapper(self) -> Mapper:
        paths = [
            self.config.semantic_units_path,
            self.config.entities_path,
            self.config.relationship_path,
            self.config.attributes_path,
            getattr(self.config, "high_level_elements_path", None),
            getattr(self.config, "high_level_elements_titles_path", None),
            self.config.text_path,
        ]
        existing = [path for path in paths if path and os.path.exists(path)]
        if not existing:
            raise RuntimeError("NodeRAG mapper cache is missing. Rebuild the pipeline.")
        return Mapper(existing)

    def _load_graph(self):
        candidates = [
            getattr(self.config, "base_graph_path", None),
            getattr(self.config, "graph_path", None),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return storage.load(path)
        raise RuntimeError("No NodeRAG graph pickle found; rerun the graph pipeline.")

    def search(self, query: str) -> Retrieval:
        retrieval = Retrieval(self.config, self.id_to_text, self.accurate_id_to_text, self.id_to_type)
        query_embedding = np.array(self.config.embedding_client.request([query])[0], dtype=np.float32)
        hnsw_results = [
            (distance, node_id)
            for distance, node_id in self.hnsw.search(query_embedding, HNSW_results=self.config.HNSW_results)
            if node_id in self.id_to_type
        ]
        retrieval.HNSW_results_with_distance = hnsw_results

        decomposed_entities = self.decompose_query(query)
        accurate_results = self.accurate_search(decomposed_entities)
        retrieval.accurate_results = accurate_results

        personalization = {ids: self.config.similarity_weight for ids in retrieval.HNSW_results}
        personalization.update({node_id: self.config.accuracy_weight for node_id in retrieval.accurate_results})
        weighted_nodes = self.graph_search(personalization)
        retrieval = self.post_process_top_k(weighted_nodes, retrieval)
        return retrieval

    def decompose_query(self, query: str) -> List[str]:
        prompt = self.config.prompt_manager.decompose_query.format(query=query)
        response = self.config.API_client.request(
            {"query": prompt, "response_format": self.config.prompt_manager.decomposed_text_json}
        )
        return response.get("elements", [])

    def accurate_search(self, entities: List[str]) -> List[str]:
        accurate_results: List[str] = []
        for entity in entities:
            words = entity.lower().split()
            if not words:
                continue
            pattern = re.compile(r"\b" + r"\s+".join(map(re.escape, words)) + r"\b")
            result = [
                node_id
                for node_id, text in self.accurate_id_to_text.items()
                if node_id in self.id_to_type and pattern.search(text.lower())
            ]
            if result:
                accurate_results.extend(result)
        return accurate_results

    def graph_search(self, personalization: Dict[str, float]) -> List[str]:
        valid = {node: weight for node, weight in personalization.items() if node in self.G}
        if not valid:
            return []
        ranked = self.sparse_PPR.PPR(valid, alpha=self.config.ppr_alpha, max_iter=self.config.ppr_max_iter)
        return [node_id for node_id, _ in ranked]

    def post_process_top_k(self, weighted_nodes: List[str], retrieval: Retrieval) -> Retrieval:
        entity_list: List[str] = []
        relationship_list: List[str] = []
        high_level_titles: List[str] = []
        addition_node = 0

        for node in weighted_nodes:
            if node not in retrieval.search_list:
                node_type = self.G.nodes[node].get("type")
                match node_type:
                    case "entity":
                        if node not in entity_list and len(entity_list) < self.config.Enode:
                            entity_list.append(node)
                    case "relationship":
                        if node not in relationship_list and len(relationship_list) < self.config.Rnode:
                            relationship_list.append(node)
                    case "high_level_element_title":
                        if node not in high_level_titles and len(high_level_titles) < self.config.Hnode:
                            high_level_titles.append(node)
                    case _:
                        if addition_node < self.config.cross_node and node not in retrieval.unique_search_list:
                            retrieval.search_list.append(node)
                            retrieval.unique_search_list.add(node)
                            addition_node += 1

                if (
                    addition_node >= self.config.cross_node
                    and len(entity_list) >= self.config.Enode
                    and len(relationship_list) >= self.config.Rnode
                    and len(high_level_titles) >= self.config.Hnode
                ):
                    break

        for entity in entity_list:
            attributes = self.G.nodes[entity].get("attributes")
            if attributes:
                for attribute in attributes:
                    if attribute not in retrieval.unique_search_list:
                        retrieval.search_list.append(attribute)
                        retrieval.unique_search_list.add(attribute)

        for high_level_title in high_level_titles:
            related_node = self.G.nodes[high_level_title].get("related_node")
            if related_node and related_node not in retrieval.unique_search_list:
                retrieval.search_list.append(related_node)
                retrieval.unique_search_list.add(related_node)

        retrieval.relationship_list = list(set(relationship_list))
        return retrieval


_SEARCH_INSTANCE: LiteNodeSearch | None = None


def _get_search() -> LiteNodeSearch:
    global _SEARCH_INSTANCE
    if _SEARCH_INSTANCE is None:
        settings = noderag_runner._load_settings()
        node_config = noderag_runner._build_node_config(settings)
        if node_config.embedding_client is None:
            raise RuntimeError("NodeRAG embedding client is not configured.")
        _SEARCH_INSTANCE = LiteNodeSearch(node_config)
    return _SEARCH_INSTANCE


def _collect_neighbors(graph, node_id: str, limit: int) -> List[Dict[str, Any]]:
    if node_id not in graph or limit <= 0:
        return []
    neighbors: List[Dict[str, Any]] = []
    for neighbor in graph.neighbors(node_id):
        data = graph.nodes[neighbor]
        edge = graph.get_edge_data(node_id, neighbor, default={})
        neighbors.append(
            {
                "hash_id": neighbor,
                "type": data.get("type"),
                "context": data.get("context"),
                "human_readable_id": data.get("human_readable_id"),
                "weight": data.get("weight"),
                "edge_weight": edge.get("weight"),
            }
        )
        if len(neighbors) >= limit:
            break
    return neighbors


def _format_doc_info(row: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not row:
        return None
    subset = {key: row.get(key) for key in ("title", "document_id", "document_date", "source_path", "publisher") if key in row}
    return subset or None


def _build_passages(
    search: LiteNodeSearch, retrieval: Retrieval, top_k: int, neighbor_limit: int
) -> List[Dict[str, Any]]:
    meta = _metadata_state()
    passages: List[Dict[str, Any]] = []
    seen = 0
    for node_id in retrieval.search_list:
        if seen >= top_k:
            break
        node_type = search.id_to_type.get(node_id)
        if node_type != "semantic_unit":
            continue
        try:
            record = search.mapper.get(node_id)
        except KeyError:
            continue
        text_hash = meta.semantic_to_text.get(node_id) or record.get("text_hash_id")
        text_info = meta.text_lookup.get(str(text_hash)) if text_hash else None
        doc_hash = (text_info or {}).get("doc_hash_id") or record.get("document_hash_id")
        doc_info = _format_doc_info(meta.doc_lookup.get(str(doc_hash)) if doc_hash else None)

        passages.append(
            {
                "hash_id": node_id,
                "type": node_type,
                "context": record.get("context"),
                "human_readable_id": record.get("human_readable_id"),
                "document": doc_info,
                "text_unit": text_info,
                "neighbors": _collect_neighbors(search.G, node_id, neighbor_limit),
            }
        )
        seen += 1
    return passages


def query_node_rag(
    question: str,
    *,
    top_k: int = 5,
    neighbor_limit: int = 5,
    include_answer: bool = True,
) -> Dict[str, Any]:
    search = _get_search()
    original_hnsw_results = search.config.HNSW_results
    search.config.HNSW_results = max(top_k, original_hnsw_results or top_k)
    try:
        retrieval = search.search(question)
    finally:
        search.config.HNSW_results = original_hnsw_results

    answer_text = None
    if include_answer:
        ans = Answer(question, retrieval)
        retrieved_info = ans.structured_prompt
        prompt = search.config.prompt_manager.answer.format(info=retrieved_info, query=question)
        try:
            response = search.config.API_client.request({"query": prompt})
            answer_text = response if isinstance(response, str) else str(response)
        except Exception:
            answer_text = None

    passages = _build_passages(search, retrieval, top_k=top_k, neighbor_limit=neighbor_limit)
    return {
        "question": question,
        "answer": answer_text,
        "passages": passages,
        "retrieval_prompt": retrieval.structured_prompt,
    }


def batch_query_node_rag(
    questions: Sequence[str],
    *,
    top_k: int = 5,
    neighbor_limit: int = 5,
    include_answer: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for question in questions:
        results.append(
            query_node_rag(
                question,
                top_k=top_k,
                neighbor_limit=neighbor_limit,
                include_answer=include_answer,
            )
        )
    return results


__all__ = ["query_node_rag", "batch_query_node_rag"]
