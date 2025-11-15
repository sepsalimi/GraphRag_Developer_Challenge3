"""
Stub module for `hnswlib_noderag` so the project can run on Python 3.12.

The vendor wheel only exists for CPython ≤3.11. Anything that actually
needs the compiled index should be run under 3.11 or with the HNSW stage
disabled (NODERAG_DISABLE_HNSW=true). Importing this module keeps the rest
of NodeRAG usable by raising a clear RuntimeError when someone tries to
instantiate the missing Index.
"""


class Index:
    def __init__(self, *_, **__):
        raise RuntimeError(
            "hnswlib_noderag is unavailable on Python 3.12. "
            "Disable the HNSW stage (set NODERAG_DISABLE_HNSW=true) or run "
            "that part of the pipeline in a Python 3.11 environment that has "
            "the official wheel."
        )


