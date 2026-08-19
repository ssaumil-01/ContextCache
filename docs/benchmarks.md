# ContextCache Calibration & Benchmarks

This guide describes how to measure the performance of ContextCache and tune its validation thresholds to balance correct response reuse and high hit rates.

---

## ⚖️ Threshold Calibration

Caching too aggressively causes incorrect cache hits (serving the wrong answer), while caching too conservatively reduces the cache hit rate (wasting LLM costs). ContextCache provides an offline calibration tool to sweep similarity thresholds on a custom dataset and find the optimal parameters.

### Calibration Tool
The calibration script is located at `tools/calibration/calibrate.py`. It runs Cosine Similarity sweeps on a JSON dataset of query pairs containing expected semantic matches (label `1`) and mismatches (label `0`).

#### Run Calibration
To start the calibration process:
```bash
python tools/calibration/calibrate.py
```

#### Output Recommendations
The tool automatically exports reports to `tools/calibration/calibration_report.json` and `calibration_report.csv`, outputting three recommended profiles:

| Profile | Selection Criteria | Typical Threshold | Recommended For |
| :--- | :--- | :--- | :--- |
| **Safety-First** | Maximizes Precision ($\ge 95\%$) where Recall $> 40\%$. | `0.85 - 0.90` | Strict applications where incorrect answers are unacceptable (legal, medical, financial). |
| **Balanced** | Maximizes F1-Score while ensuring Precision is at least $80\%$. | `0.70 - 0.80` | Standard customer support, document search, and general information bots. |
| **Max F1** | Absolute highest F1-Score (balances Precision and Recall equally). | `0.60 - 0.70` | Casual assistants, chat applications, or highly creative pipelines. |

---

## 📈 Performance Benchmarks

Below are the benchmark results simulated on a representative RAG workload mixing exact string repetitions, semantic phrasing variations, and new topic pioneering queries over 10 execution cycles.

### Execution Settings
- **Workload mix**: 30% Repetitive Queries, 40% Semantic Queries, 30% Pioneering Queries.
- **L1 TTL**: 24 hours.
- **L2 Similarity Threshold**: `0.85`.
- **L2 Overlap Threshold**: `0.85`.

### Results Comparison

```
Latency Comparison (10 Query Workload)
──────────────────────────────────────────────────────────────────────────
Baseline (No Cache):   ██████████████████████████████  15.00s
With ContextCache:         ████████─────  6.50s (2.3x Speedup)
──────────────────────────────────────────────────────────────────────────
```

### Metrics Dashboard
When metrics telemetry is enabled, you can monitor performance dynamically on the Grafana dashboard:

- **LLM Call Elimination Rate**: **60%** (6 out of 10 LLM calls completely avoided).
- **L1 Hit Rate**: **30%** (bypassed Document VectorDB lookups).
- **L2 Hit Rate**: **30%** (bypassed LLM generation APIs).
- **Average Latency Reduction**: Latency dropped from an average of `1.50s` per query to `0.65s` per query.

---

## ⚖️ Load Testing Workload Analysis

When evaluating cache performance under concurrent stress using load testing, using small synthetic query pools (e.g., < 10 queries) leads to unrealistic cache hit rates approaching 100%. To prevent this and simulate a realistic production environment, ContextCache utilizes a dynamically generated query workload dataset of 1,000+ entries (`benchmark_workload.json`).

### Workload Composition
The query workload is generated following typical user query behaviors and drift in production:

*   **30% Exact Repeats** (L1 Hit Target): Simulates users querying the exact same questions (e.g. system status, office hours). These queries hit the fast L1 cache directly, bypassing vector database lookups and LLM generation.
*   **40% Semantic Variants / Paraphrases** (L2 Hit Target): Simulates users asking the same questions using different phrasing (e.g., "What is the WFH policy?" vs. "Can I work from home?"). These queries bypass L1 but hit the L2 semantic cache, bypassing the LLM.
*   **30% Novel Queries** (Cache Miss Target): Simulates users asking completely unrelated, new topic questions. These queries miss both L1 and L2 caches, triggering the full retriever and LLM generation pipeline.

### Target Performance Metrics
Under a full load test run with the staggered pacing (sleep between 50ms and 150ms) and multi-tenant isolation, the expected target metrics are:

| Metric | Target Range | Key System Path Tested |
| :--- | :--- | :--- |
| **L1 Cache Hit Rate** | `20% - 40%` | Fast Redis L1 key-value lookup path. |
| **L2 Cache Hit Rate** | `30% - 60%` | FAISS index semantic similarity validation. |
| **Cache Miss Rate** | `10% - 30%` | Backend database retrieval, LLM call, and cache stampede mutex protection. |

For detailed information on configuring and running these load tests, refer to the [LOAD_TESTING.md](file:///Users/utkarshbansal/Context-aware-cashe/docs/LOAD_TESTING.md) guide.

---


## 🛠️ Step-by-Step Optimization Process

To get the highest speedups and savings in production:

1. **Collect query history**: Export a log of typical user queries and their returned document IDs.
2. **Formulate a validation dataset**: Create a JSON file in `tools/calibration/dataset.json` following this structure:
   ```json
   [
     {
       "query_a": "How do I request paid time off?",
       "query_b": "What is the process to apply for PTO?",
       "expected_label": 1,
       "description": "Semantic matches"
     },
     {
       "query_a": "Apply policy",
       "query_b": "What is the policy?",
       "expected_label": 0,
       "description": "Mismatched intent"
     }
   ]
   ```
3. **Execute calibration**: Run `python tools/calibration/calibrate.py`.
4. **Update configuration**: Set the recommended threshold value in your `pyproject.toml` config, environment variables, or YAML configuration file.
