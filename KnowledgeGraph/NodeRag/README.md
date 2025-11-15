# NodeRAG Cache Runner

This folder contains the light wrapper we use to run the vendored [NodeRAG](https://github.com/Terry-Xu-666/NodeRAG) library directly on the markdown files in `KnowledgeGraph/source_data`. The goal is to stay as close to the upstream defaults as possible:

- The pipeline writes everything to `.noderag/` (documents, semantic units, entities, relationships, embeddings and the pickled heterogeneous graph).
- Neo4j is **not** part of this flow anymore. We rely on NodeRAG's own cache and graph files for inspection.
- The vendored library lives in `vendor/NodeRAG`. Please keep it there so the custom patches (tokenizer fallback, asyncio fixes, etc.) stay in one place.
- `vendor/RAG-Anything` is unrelated to NodeRAG. It is kept around for the LightRAG/RAG-Anything experiments referenced in `KnowledgeGraph/LightRAG`. Feel free to ignore it when working on NodeRAG.

## Running the pipeline

1. Make sure the required environment variables are present in `.env`:
   - `NODERAG_SOURCE_DIR` – defaults to `KnowledgeGraph/source_data`
   - `NODERAG_WORK_DIR` – defaults to `.noderag`
   - `NODERAG_LLM_MODEL` – e.g. `gpt-5-mini`
   - `OPENAI_API_KEY` – only needed if you are not using local HF embeddings
   - Optional overrides such as `NODERAG_HF_MODEL`, `NODERAG_EMBED_BACKEND`, `NODERAG_CHUNK_TARGET_TOKENS`, etc.
2. Run the CLI:
   ```bash
   poetry run python "Main Functions/prep_noderag.py" \
       [--reset-cache] [--skip-sync] [--graph-stats]
   ```
   - `--reset-cache` deletes `.noderag/cache` and `.noderag/info` before the run.
   - `--skip-sync` reuses whatever is already inside `.noderag/input`.
   - `--graph-stats` prints a quick node/edge breakdown after the run.
3. To inspect the graph later without re-running the expensive stages:
   ```bash
   poetry run python Scripts/noderag_graph_stats.py
   ```

## Notebook helpers

`Main Functions/prep_noderag.ipynb` calls the same helper functions that the CLI uses:

```python
from KnowledgeGraph.NodeRag import (
    get_cache_dir,
    load_cached_outputs,
    print_graph_overview,
    run_noderag,
    summarize_outputs,
)
```

That notebook keeps a reference to the latest `outputs` object so you can poke around the semantic units/entities in-place. When you only need cached stats, call `load_cached_outputs()` to avoid an eight-hour rerun.

## Visualization

For a text-only overview, call `print_graph_overview(outputs.graph)` (available in both the CLI and notebook). It prints node counts by type, edge counts by type, and the highest-degree nodes. Future visualizations (PyVis, Cytoscape, etc.) can read directly from `.noderag/cache/graph.pkl` without involving Neo4j.

