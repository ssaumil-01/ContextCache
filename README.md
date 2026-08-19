# ContextCache

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/context-cachex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Support](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/context-cachex/)
[![CI Build Status](https://github.com/Utk-858/Context-aware-cashe/actions/workflows/ci.yml/badge.svg)](https://github.com/Utk-858/Context-aware-cashe/actions/workflows/ci.yml)

ContextCache is a **context-aware, multi-level caching system** designed to accelerate and optimize Retrieval-Augmented Generation (RAG) pipelines. Unlike traditional exact-match KV caches, ContextCache utilizes semantic similarity, context stability validation, and user intent classification to safely reuse LLM responses while strictly preserving correctness.

---

## 🚀 Key Features

* **Dual-Layer Caching (L1 & L2)**:
  - **L1 (Retrieval Cache)**: Low-latency exact-match string cache mapping queries directly to document IDs, bypassing heavy vector search lookups.
  - **L2 (Generation Cache)**: High-performance semantic cache mapping queries and contexts to LLM responses.
* **Context Stability Validation**: Analyzes document overlaps using Jaccard Similarity to reject cache hits if retrieved context documents have drifted.
* **Intent-Aware Caching**: Classifies queries (e.g. action, informational) to bypass caching automatically for mutation operations.
* **Strict Tenant Isolation**: Restricts cache retrieval scope dynamically to prevent context leakages across different users and tenants.
* **Prometheus Observability**: Native latency histograms, cache hit-rate counters, and memory tracking.
* **Optimization ROI**: Speeds up RAG pipelines by **2.3x** and reduces API token costs by up to **60%** with zero loss in contextual accuracy.

---

## 📐 Architecture Overview

```
                        User Query
                            │
                            ▼
                    ┌──────────────┐
                    │  L1 Cache    ├─────────► Hit (Yields Doc IDs)
                    └──────┬───────┘                 │
                           │ Miss                    │
                           ▼                         │
                    ┌──────────────┐                 │
                    │ Vector DB    │◄────────────────┘
                    └──────┬───────┘
                           │ doc_ids
                           ▼
                    ┌──────────────┐
                    │  L2 Cache    ├─────────► Hit (Returns LLM Response)
                    └──────┬───────┘
                           │ Miss
                           ▼
                    ┌──────────────┐
                    │  Expensive   │
                    │   LLM API    │
                    └──────────────┘
```

For a deep dive into caching layers and algorithms, read the [Architecture Guide](file:///Users/utkarshbansal/Context-aware-cashe/docs/architecture.md).

---

## 📦 Installation

Install the core library (includes Redis support and Prometheus telemetry):
```bash
pip install context-cachex
```

### Optional Backends:
- **Local FAISS Vector DB**:
  ```bash
  pip install "context-cachex[faiss]"
  ```
- **Local Sentence-Transformers Embeddings**:
  ```bash
  pip install "context-cachex[embeddings]"
  ```
- **Full Installation (All Backends)**:
  ```bash
  pip install "context-cachex[all]"
  ```

---

## ⚡ Quick Start

Get running in less than 2 minutes using our high-level facade:

```python
from rag_cache import ContextCache

# Initialize dual-layer cache facade
cache = ContextCache()

# Wrap your existing RAG execution function
def query_rag_pipeline(query: str):
    # Pass query, retriever callback, and LLM callback
    result = cache.run(
        query=query,
        retriever=lambda q: ["doc_abc", "doc_xyz"],
        llm=lambda q, doc_ids: "Context-aware systems ensure LLM correctness."
    )
    return result["answer"]

# Executes LLM call (miss)
print(query_rag_pipeline("What is context-aware caching?"))

# Returns cached response immediately (semantic hit!)
print(query_rag_pipeline("Explain context-aware caching."))
```

For more execution models and advanced configurations, see the [Quick Start Guide](file:///Users/utkarshbansal/Context-aware-cashe/docs/quickstart.md).

---

## ⚙️ Configuration

ContextCache supports configuration loading with the following precedence:
1. **Keyword overrides** in constructor (e.g. `ContextCache(redis_url="...")`).
2. **Environment variables** (`REDIS_URL`, `L1_TTL`, `L2_TTL`, `SIMILARITY_THRESHOLD`).
3. **YAML configuration file** (e.g. loaded via `ContextCache.from_config("config.yaml")`).
4. **Built-in defaults**.

See [config_example.yaml](file:///Users/utkarshbansal/Context-aware-cashe/examples/config_example.yaml) for a complete template configuration.

---

## 🔒 Tenant Isolation

Ensure user data boundaries are strictly isolated by passing a `tenant_id` to caching endpoints:

```python
result = cache.run(
    query=query,
    retriever=retriever_fn,
    llm=llm_fn,
    tenant_id="customer_company_a",
    scope="tenant"  # Options: "tenant" (default), "user", or "global"
)
```

For details on namespacing strategies in Redis, see [Architecture Guide](file:///Users/utkarshbansal/Context-aware-cashe/docs/architecture.md).

---

## 📊 Metrics & Telemetry

ContextCache collects statistics out-of-the-box using Prometheus:
- Cache hits/misses split by L1 and L2 layers.
- Operational latency histograms for Redis, FAISS, and Embeddings.
- Eviction count and Redis memory usage.

To enable the HTTP telemetry server, set the `prometheus_port` config parameter:
```python
cache = ContextCache(prometheus_port=8000)
```
Scrape metrics at `http://localhost:8000/metrics`. We provision pre-built Grafana dashboards under `config/grafana/dashboards`.

---

## 🧪 Calibration & Benchmarking

To optimize caching threshold parameters for your specific RAG datasets, run the offline calibration script:
```bash
python tools/calibration/calibrate.py
```
This runs a threshold sweep to recommend optimal parameters under **Safety-First**, **Max F1**, and **Balanced** profiles. Read the [Benchmark Documentation](file:///Users/utkarshbansal/Context-aware-cashe/docs/benchmarks.md) for more details.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](file:///Users/utkarshbansal/Context-aware-cashe/LICENSE) for details.
