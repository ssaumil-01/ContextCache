# ContextCache Quick Start Guide

Get ContextCache up and running in your Retrieval-Augmented Generation (RAG) applications in less than 5 minutes.

---

## 📦 Step 1: Installation

ContextCache requires Python 3.10 or higher. Install the base package with production-ready Redis support:

```bash
pip install rag-cachex
```

### Enable Vector Support (FAISS & sentence-transformers)
To utilize local FAISS database storage and offline sentence embedding execution natively, install all extra backends:
```bash
pip install "rag-cachex[all]"
```

---

## ⚡ Step 2: Running with the Facade API (Recommended)

The high-level `ContextCache` facade wraps your existing RAG pipeline with a single function call, managing document fetches, semantic searches, overlap checks, and LLM queries automatically.

Ensure you have a Redis instance running locally at `redis://localhost:6379/0` (or set the `REDIS_URL` env variable), then copy this script:

```python
from rag_cache import ContextCache

# 1. Initialize the dual-layer cache with sensible defaults
cache = ContextCache()

# 2. Mock downstream systems (VectorDB and LLM)
def retrieve_docs(query: str):
    print("  🔍 Querying Vector Database...")
    return ["document_id_101", "document_id_102"]

def query_llm(query: str, doc_ids: list):
    print("  🧠 Querying Expensive LLM API...")
    return "LLMs process prompt contexts to form completions."

# 3. Wrap pipeline
def run_pipeline(query: str):
    print(f"\n=> User Query: '{query}'")
    
    # Executes the pipeline with caching
    result = cache.run(
        query=query,
        retriever=retrieve_docs,
        llm=query_llm
    )
    
    print(f"   Cache Hit: {result['cache_hit']} | Source: {result['source']}")
    print(f"   Answer: {result['answer']}")

# Execution Runs:
# Run 1: Cold Cache (Miss L1 -> Miss L2 -> Hits DB & LLM)
run_pipeline("What is retrieval-augmented generation?")

# Run 2: Exact String match (Hits L1 -> Bypasses DB & LLM completely)
run_pipeline("What is retrieval-augmented generation?")

# Run 3: Semantically identical query (Misses L1 -> Hits L2 Semantic cache)
run_pipeline("Explain retrieval-augmented generation.")
```

---

## 🛠️ Step 3: Custom Flow (Step-by-Step API)

If you need granular control over your retrieval database queries and LLM generation phases, you can use the low-level step-by-step methods:

```python
from rag_cache import ContextCache

cache = ContextCache()

query = "What are context-aware caching systems?"

# Step A: Check L1 exact string cache to avoid VectorDB search
doc_ids = cache.get_docs(query)

if doc_ids:
    print("⚡ L1 Cache Hit! Skipping VectorDB search.")
else:
    # Perform standard vector search
    doc_ids = ["doc_alpha", "doc_beta"]
    print("🔍 VectorDB Miss. Document IDs fetched.")

# Step B: Check L2 semantic cache to avoid LLM call
answer = cache.get_answer(query, doc_ids)

if answer:
    print("✅ L2 Cache Hit! Skipping LLM generation.")
else:
    # Generate response from LLM
    print("🧠 Cache Miss. Calling LLM...")
    answer = "Context-aware caches analyze document overlap to prevent wrong hits."
    
    # Step C: Save L1 and L2 caches for future queries
    cache.save(query, doc_ids, answer)

print(f"Final Answer: {answer}")
```

---

## ⚙️ Step 4: Developer-Friendly Configurations

ContextCache parses and applies configurations dynamically:

### Option A: Initialize directly from a YAML file
Create `config.yaml`:
```yaml
redis_url: "redis://localhost:6379/1"
use_faiss: true
min_embedding_similarity: 0.90
min_document_overlap: 0.80
```
Instantiate the cache using the loader:
```python
cache = ContextCache.from_config("config.yaml")
```

### Option B: Override using environment variables
```bash
export REDIS_URL="redis://production-host:6379/0"
export SIMILARITY_THRESHOLD="0.88"
export L1_TTL="3600"
```
Instantiate directly without arguments; the system automatically merges env variables:
```python
cache = ContextCache()
```

For complete parameter details, check out the [Architecture Documentation](file:///Users/utkarshbansal/Context-aware-cashe/docs/architecture.md).
