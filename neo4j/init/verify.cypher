// Demonstrate that the Chunk embedding index participates in query planning.

EXPLAIN
CALL db.index.vector.queryNodes(
  "chunk_embedding_hnsw",
  [value IN range(0, toInteger($embed_dim) - 1) | 0.0],
  10
)
YIELD node, score
RETURN node.id AS chunk_id, score
LIMIT 5;
