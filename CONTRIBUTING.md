# Contributing to rag_cache

We welcome contributions! The library is designed to be highly extensible.

## Adding a new Vector Store

1. Look at `rag_cache/interfaces/vector_store.py` for the methods you need to implement.
2. Create a new file in `rag_cache/integrations/vector_stores/` (e.g., `qdrant.py`).
3. Add a test in `tests/integrations/`.

## Adding a new Embedding Provider

1. Look at `rag_cache/interfaces/embedding.py`.
2. Create a new file in `rag_cache/integrations/embeddings/` (e.g., `voyage_ai.py`).
3. Add a test in `tests/integrations/`.
