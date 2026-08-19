# ContextCache Load Testing Framework

This document describes how to execute concurrency load testing using **k6** against a **FastAPI** wrapper, integrate metrics into Grafana, and analyze potential scaling bottlenecks.

---

## 📐 Load Testing Architecture

ContextCache is a Python library. To run external load tests (e.g. simulating 500 concurrent users), we wrap the library in a lightweight FastAPI application that runs ContextCache requests and exposes telemetry.

```
┌──────────────┐                       ┌──────────────┐
│  k6 Clients  │ ────[HTTP POST]─────► │ FastAPI App  │
└──────────────┘      (/query)         │   (Uvicorn)  │
                                       └──────┬───────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │   ContextCache   │
                                       └──────┬───────┘
                                              │
                             ┌────────────────┴────────────────┐
                             ▼                                 ▼
                      ┌─────────────┐                   ┌──────────────┐
                      │ Redis Cache │                   │ FAISS Vector │
                      └─────────────┘                   └──────────────┘
```

---

## 🚀 Execution Guide

### 1. Start the Target API Server
Ensure Redis is running (e.g., via Docker on port `6379`), install dependencies, and start the FastAPI server:
```bash
# Install performance dependencies
pip install -e .[all,perf]

# Start the server (bind to port 8000)
uvicorn tools.load_testing.server:app --host 0.0.0.0 --port 8000
```

### 2. Generate the Workload Dataset
Before executing the load test, compile the realistic RAG query distribution (30% exact repeats, 40% semantic variants, 30% novel queries) by running:
```bash
python tools/load_testing/generate_workload.py
```
This generates the workload file `tools/load_testing/benchmark_workload.json` consisting of 1,000 queries.

### 3. Execute k6 Load Test
Install k6 on your system (e.g., using Homebrew on macOS `brew install k6` or run via Docker).

Run the k6 test script:
```bash
k6 run tools/load_testing/load_test.js
```

During execution, k6 will load `benchmark_workload.json` and print the following custom counters at the end of the run under `CUSTOM`:
- `exact_query_count`: Total count of exact query matches executed.
- `semantic_query_count`: Total count of semantic query variants executed.
- `novel_query_count`: Total count of novel/pioneering queries executed.


---

## 📉 Benchmark Methodology

The load test sweeps virtual users (VUs) through 4 distinct stages to analyze system behavior under increasing concurrency stress:

1. **Warm-up / Baseline (10 VUs)**: Captures optimal performance of hits and misses with minimal scheduling overhead.
2. **Normal Load (50 VUs)**: Typical operational capacity.
3. **Scale Capacity (100 VUs)**: Approaching high capacity limits.
4. **Stress Level (500 VUs)**: Evaluates queuing delay, connection exhaustion, and threshold limits.

### Metrics Recorded
- **k6 metrics**:
  - `http_reqs`: Throughput (requests/sec).
  - `http_req_duration`: End-to-end HTTP response latency (p50, p90, p95, p99).
  - `http_req_failed`: Request error rate.
- **ContextCache Prometheus metrics**:
  - `rag_cache_redis_latency_seconds`: Redis read/write latency.
  - `rag_cache_faiss_search_latency_seconds`: FAISS vector search operations.
  - `rag_cache_embedding_latency_seconds`: Simulated/real embedding generation.

---

## 📊 Dashboard Recommendations

To monitor the performance under load, we recommend creating the following panels on your Grafana dashboard:

### 1. Caching Efficiency (Hits vs Misses)
- **Type**: Time Series
- **Query**: `sum(rate(rag_cache_l1_hits_total[1m])) by (tenant_id)` vs `sum(rate(rag_cache_l1_misses_total[1m]))`
- **Goal**: Measures retrieval L1 cache efficiency.

### 2. Latency Breakdown
- **Type**: Heatmap / Time Series
- **Queries**:
  - Redis Operations: `histogram_quantile(0.95, sum(rate(rag_cache_redis_latency_seconds_bucket[1m])) by (le))`
  - FAISS Searches: `histogram_quantile(0.95, sum(rate(rag_cache_faiss_search_latency_seconds_bucket[1m])) by (le))`
- **Goal**: Pinpoints whether Redis connection pools or FAISS indices are introducing scaling overhead.

### 3. Throughput & Error Rates
- **Type**: Gauge / Single Stat
- **Query**: `sum(rate(http_requests_total[1m]))` and `sum(rate(http_requests_failed_total[1m]))`

---

## 🔍 Likely Bottlenecks & Mitigations

Under 500 concurrent virtual users, Python applications often encounter performance degradation. Here are the likely bottlenecks and how to configure ContextCache to bypass them:

### 1. Python Global Interpreter Lock (GIL) Contention
- **Symptom**: High CPU usage on a single core; Uvicorn throughput plateaus under high concurrency.
- **Mitigation**: Uvicorn runs single-threaded. In production, run the wrapper with multiple worker processes to utilize multi-core hosts:
  ```bash
  uvicorn tools.load_testing.server:app --host 0.0.0.0 --port 8000 --workers 4
  ```

### 2. Redis Connection Pool Exhaustion
- **Symptom**: Redis operation latency spikes; client requests fail with timeout exceptions.
- **Mitigation**: The default connection pool is set to 50 connections. Under 500 concurrent users, scale up `max_connections` inside your `pyproject.toml` or YAML configuration:
  ```yaml
  # config.yaml overrides
  redis_url: "redis://localhost:6379/0"
  max_connections: 500
  ```

### 3. FAISS Thread Contention
- **Symptom**: FAISS search latency spikes and causes scheduling blocks.
- **Mitigation**: OpenMP threads clash when multiple Python processes query the C++ FAISS library simultaneously. Ensure `faiss.omp_set_num_threads(1)` is set (ContextCache does this automatically) to keep vector lookups running single-threaded inside each worker.
