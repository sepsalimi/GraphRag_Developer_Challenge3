// Base GraphRAG schema. Parameter `$embed_dim` is injected by scripts/init-neo4j.sh.
// Statements are idempotent so re-running the init container is safe.

// Core node identifiers
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (chunk:Chunk)
REQUIRE chunk.id IS UNIQUE;

CREATE CONSTRAINT section_id_unique IF NOT EXISTS
FOR (section:Section)
REQUIRE section.id IS UNIQUE;

CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (doc:Document)
REQUIRE doc.id IS UNIQUE;

CREATE CONSTRAINT gazette_id_unique IF NOT EXISTS
FOR (gazette:Gazette)
REQUIRE gazette.id IS UNIQUE;

// Helpful property indexes
CREATE INDEX document_title_idx IF NOT EXISTS
FOR (doc:Document)
ON (doc.title);

CREATE INDEX publication_date_idx IF NOT EXISTS
FOR (gazette:Gazette)
ON (gazette.publication_date);

// Vector indexes for embeddings (HNSW cosine by default)
CREATE VECTOR INDEX chunk_embedding_hnsw IF NOT EXISTS
FOR (chunk:Chunk)
ON (chunk.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: toInteger($embed_dim),
    `vector.similarity_function`: 'cosine',
    `vector.index_engine`: 'hnsw',
    `vector.hnsw.m`: 16,
    `vector.hnsw.ef_construction`: 128
  }
};

CREATE VECTOR INDEX section_embedding_hnsw IF NOT EXISTS
FOR (section:Section)
ON (section.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: toInteger($embed_dim),
    `vector.similarity_function`: 'cosine',
    `vector.index_engine`: 'hnsw',
    `vector.hnsw.m`: 16,
    `vector.hnsw.ef_construction`: 128
  }
};

// Wait for the indexes to finish building before returning.
CALL db.awaitIndexes(300);
