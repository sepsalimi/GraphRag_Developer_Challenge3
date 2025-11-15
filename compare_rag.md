# LightRAG vs. RAG-Anything Quickstart

## 1. Install dependencies (Poetry)

```bash
poetry add python-dotenv
poetry add git+https://github.com/HKUDS/LightRAG.git
# RAG-Anything is vendored under vendor/RAG-Anything – no extra Poetry add required.
```

> The RAG-Anything repository is checked out in `vendor/RAG-Anything` (to avoid a conflicting `json-repair` pin).  
> No additional install step is required beyond the clone already present in the repo.

## 2. Prepare LightRAG data

Open and run `Main Functions/prep_lightrag.ipynb`.  
This notebook loads your `.env`, reads Markdown files from `KnowledgeGraph/source_data/`, and ingests them into `./.rag/lightrag/`.

## 3. Compare inside `Main Functions/main.ipynb`

Two new sections were added:

- **LightRAG**  
  Toggle `LIGHTRAG_REBUILD` to control re-ingestion.  
  Run the single-query cell to sanity check, then the batch cell to save answers under `AB Testing/Output Answers/answers_lightrag_*.json`.

- **RAG-Anything**  
  Toggle `RAGANYTHING_REBUILD` as needed (defaults to using the existing store).  
  Single and batch query cells mirror the LightRAG flow and save to `answers_raganything_*.json`.

Both sections rely on the same Markdown corpus, so you can compare answer quality and latency easily.

## 4. Optional CLI

For a quick terminal check:

```bash
poetry run python Scripts/try_rag.py \
  --backend lightrag \
  --question "Summarize the key decrees from April 13, 2025."
```

Options:

- `--backend {lightrag, raganything}`
- `--mode` for retrieval (`mix`, `global`, `local`, etc.)
- `--skip-ingest` if you only want to query existing stores
- `--split-character` / `--split-only` to tweak chunking

## 5. Environment variables

Ensure `.env` contains at least:

```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_BASE_URL=...   # optional
```

You can override defaults:

- `LIGHTRAG_LLM_MODEL` (defaults to `gpt-5-mini`)
- `LIGHTRAG_EMBED_MODEL` (defaults to `text-embedding-3-large`; `text-embedding-3-small` is cost-effective)
- `LIGHTRAG_EMBED_DIM` (optional – inferred from the model if omitted)
- `LIGHTRAG_SOURCE_DIR` if you store Markdown elsewhere

When you swap between embedding sizes (small ↔ large), delete `./.rag/lightrag/` and rebuild any Neo4j vector indexes so stored vectors match the new dimension. `RAG-Anything` shares the same LightRAG core, so the settings apply to both backends.

