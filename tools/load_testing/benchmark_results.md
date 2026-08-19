# ContextCache Concurrency Stress Test Results & Bottleneck Analysis

*Generated at: 2026-08-19 23:40:25 (Scenario Duration: 1m)*

## 1. Single Worker Benchmark Results (`--workers 1`, Mock Embeddings)

| VUs | Throughput | Avg Latency | P50 (Med) | P90 | P95 | Errors | Sys CPU | Proc CPU | RSS Mem | Redis Latency | FAISS Latency | Embedding Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 7.10 req/s | 1.287s | 1.034s | 2.722s | 2.875s | 0.0% | 45.7% | 55.6% | 71.7 MB | 1.76ms | 0.00ms | 1.03ms |
| 50 | 5.40 req/s | 8.748s | 7.623s | 10.833s | 26.877s | 2.2% | 35.0% | 50.3% | 72.2 MB | 2.32ms | 0.00ms | 1.59ms |
| 100 | 10.94 req/s | 8.553s | 4.468s | 9.471s | 59.983s | 5.6% | 36.6% | 58.9% | 74.7 MB | 1.68ms | 0.00ms | 0.82ms |
| 500 | 1631.87 req/s | 0.110s | 0.000s | 0.000s | 0.000s | 99.4% | 43.7% | 56.1% | 82.8 MB | 1.11ms | 0.00ms | 0.63ms |

## 2. Multi-Worker Benchmark Results (`--workers 4`, Mock Embeddings)

| VUs | Throughput | Avg Latency | P50 (Med) | P90 | P95 | Errors | Sys CPU | Proc CPU | RSS Mem | Redis Latency | FAISS Latency | Embedding Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 50.68 req/s | 0.096s | 0.077s | 0.179s | 0.217s | 0.0% | 46.7% | 221.6% | 317.7 MB | 0.75ms | 0.00ms | 0.56ms |
| 50 | 50.67 req/s | 0.851s | 0.521s | 1.936s | 2.044s | 0.0% | 51.1% | 2.9% | 270.6 MB | 0.84ms | 0.00ms | 0.56ms |
| 100 | 48.68 req/s | 1.913s | 1.817s | 2.148s | 3.344s | 0.0% | 53.7% | 0.5% | 273.5 MB | 0.87ms | 0.00ms | 0.57ms |
| 500 | 68.23 req/s | 5.676s | 4.676s | 19.418s | 21.458s | 43.1% | 51.2% | 0.8% | 272.8 MB | 0.92ms | 0.00ms | 0.59ms |

## 3. Multi-Worker Benchmark Results (`--workers 4`, Real SentenceTransformer)

| VUs | Throughput | Avg Latency | P50 (Med) | P90 | P95 | Errors | Sys CPU | Proc CPU | RSS Mem | Redis Latency | FAISS Latency | Embedding Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 40.97 req/s | 0.143s | 0.110s | 0.269s | 0.331s | 0.0% | 58.6% | 132.7% | 921.2 MB | 0.92ms | 0.00ms | 0.63ms |
| 50 | 14.11 req/s | 3.260s | 1.555s | 6.153s | 11.541s | 0.0% | 60.6% | 134.6% | 1506.7 MB | 1.58ms | 0.00ms | 1.40ms |
| 100 | 9.25 req/s | 9.797s | 9.939s | 12.082s | 12.666s | 0.0% | 57.9% | 181.8% | 664.9 MB | 2.12ms | 0.00ms | 1.90ms |
| 500 | 9.38 req/s | 37.228s | 42.238s | 59.907s | 60.060s | 28.0% | 57.7% | 159.1% | 679.5 MB | 2.27ms | 0.00ms | 2.42ms |

## 4. Worker Contention Comparison (Single vs. Multi-Worker, Mock Embeddings)

| VUs | 1 Worker Throughput | 4 Workers Throughput | Speedup | 1 Worker P95 | 4 Workers P95 | P95 Latency Reduction | 1 Worker CPU | 4 Workers CPU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 7.10 req/s | 50.68 req/s | 7.14x | 2.875s | 0.217s | 92.5% | 55.6% | 221.6% |
| 50 | 5.40 req/s | 50.67 req/s | 9.38x | 26.877s | 2.044s | 92.4% | 50.3% | 2.9% |
| 100 | 10.94 req/s | 48.68 req/s | 4.45x | 59.983s | 3.344s | 94.4% | 58.9% | 0.5% |
| 500 | 1631.87 req/s | 68.23 req/s | 0.04x | 0.000s | 21.458s | 0.0% | 56.1% | 0.8% |

## 5. Embedding Overhead Comparison (Mock vs. Real Embeddings under 4 Workers)

| VUs | Mock Throughput | Real Throughput | Throughput Retention | Mock P95 | Real P95 | Latency Overhead | Mock CPU | Real CPU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 50.68 req/s | 40.97 req/s | 80.8% | 0.217s | 0.331s | 52.4% | 221.6% | 132.7% |
| 50 | 50.67 req/s | 14.11 req/s | 27.8% | 2.044s | 11.541s | 464.6% | 2.9% | 134.6% |
| 100 | 48.68 req/s | 9.25 req/s | 19.0% | 3.344s | 12.666s | 278.8% | 0.5% | 181.8% |
| 500 | 68.23 req/s | 9.38 req/s | 13.7% | 21.458s | 60.060s | 179.9% | 0.8% | 159.1% |

## 6. Bottleneck & Latency Analysis

### Quantitative Diagnoses:
- **Uvicorn Worker CPU Utilisation (Test D, 500 VUs)**:
  - 1 Worker (Mock): **56.1%**
  - 4 Workers (Mock): **0.8%**
  - 4 Workers (Real SentenceTransformer): **159.1%**
- **Sub-Component Latency Comparison (Test D, 500 VUs)**:
  - Redis Latency: Mock = **0.92ms**, Real = **2.27ms**
  - FAISS Latency: Mock = **0.00ms**, Real = **0.00ms**
  - Embedding Generation Latency: Mock = **0.59ms**, Real = **2.42ms**
- **Total API Service Time vs. Client-Perceived Latency (Test D, 500 VUs)**:
  - Mock Embeddings: Service Time = **1.51ms**, Client Latency (Avg) = **5.676s**
  - Real Embeddings: Service Time = **4.69ms**, Client Latency (Avg) = **37.228s**

### Analysis & Conclusion:
1. **Is Embedding Generation a Bottleneck?** **NO, it remains lightweight under all-MiniLM-L6-v2.**
   - **Evidence**: Embedding latency remained at **2.42ms**, showing that the CPU cost of the neural model is minor compared to connection management and loop overheads.
2. **Uvicorn Worker Contention**: The GIL and single-process architecture of Uvicorn remain the dominant bottleneck.