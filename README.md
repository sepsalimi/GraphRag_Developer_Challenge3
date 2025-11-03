### GraphRAG Developer Challenge

A Neo4j-backed GraphRAG pipeline: prepare data, build a knowledge graph with embeddings, answer questions, and score results.

### Pipeline
- Ingest: place source markdown in `KnowledgeGraph/source_data/` and chunk it.
- Build KG: connect to Neo4j, create nodes/chunks, create vector index, embed text.
- Retrieve/Answer: run hybrid Graph+Vector RAG to answer questions with citations.
- Evaluate: run batch questions and score against answer keys.

### Key files and folders
- **Main Functions/**
  - `prep.ipynb`: data prep to make documents ingest-ready.
  - `main.ipynb`: Single query / multi-query functionality.
  - `VectorRAG.ipynb`: vector‑only baseline demo.

- **KnowledgeGraph/**
  - `source_data/`: markdown inputs to ingest.
  - `config.py`: loads `.env` and returns a Neo4j graph handle and OpenAI config.
  - `chunking.py`: basic JSON/MD chunking into text chunks with metadata.
  - `chunking_strategy.py`: table‑aware markdown chunking for better retrieval.
  - `knowledgegraph.py`: helpers to create nodes, chunks, relationships, vector index, and embed nodes.
  - `pipeline_utils.py`: constructs a simple KG builder pipeline (Neo4j GraphRAG).

- **RAG/**
  - `GraphRAG.py`: main hybrid pipeline (graph + vector) with neighbor window, anchor gating, and citations.
  - `HybridRAG.py`: hybrid retriever with MMR and optional reranking.
  - `VectorRAG.py`: pure vector retriever baseline.
  - `GetNeighbor.py`: expands hits with adjacent chunks from the same document.
  - `AnchorUtils.py`: detects tender/practice anchors; slot guard helpers.
  - `Citations.py`: builds Vancouver‑style references; appends citations to answers.
  - `MMR.py`: maximal marginal relevance selection helpers.
  - `Lucene.py`: escapes special characters for full‑text queries.
  - `UseCatalog.py`: card‑first gate using `RAG/card_catalog.jsonl` to restrict scope or answer directly.
  - `CheckUpdates.py`, `CheckSupplement.py`: include update/supplement snippets when relevant.
  - `batch_query.py`: batch‑answer questions and save to `AB Testing/Output Answers/`.
  - `normalize.py`: normalizes currencies, dates, and phrasing in answers.

- **AB Testing/**
  - `questions_*.json` / `answer_key_*.json`: question sets and expected answers.
  - `Output Answers/`: model answers saved by batch runs.
  - `mark_answers.py`: scores with GPT‑graded F1; writes `Test Results/test_results_*.json`.

- **Infra & config**
  - `docker-compose.yml`: local Neo4j (with APOC + GenAI) on 7474/7687.
  - `neo4j-README.md`: official Neo4j GraphRAG docs (reference).
  - `pyproject.toml` / `poetry.lock`: dependencies.
  - `Scripts/poetryexport.sh`: export dependencies.

### Quick start
1) Start Neo4j: `docker compose up -d`
2) Install dependencies: `poetry install` (uses `poetry.lock` for pinned versions)
3) Create `.env` in project root (min): `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_FULLTEXT_INDEX`, `NEO4J_VECTOR_INDEX`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBED_MODEL`, and MMR settings (`MMR_k`, `MMR_LAMBDA`, `ALWAYS_KEEP_TOP`).
4) Single question: `python "Main Functions/main.py"`
5) Batch + score (typical): run batch via `RAG/batch_query.py`, then run `AB Testing/mark_answers.py` with the matching `answer_key_*.json`.
