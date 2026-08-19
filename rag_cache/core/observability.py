import time
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ---------------------------------------------------------
# Prometheus Metric Definitions
# ---------------------------------------------------------

# L1 (Exact Match) Cache Metrics
L1_HITS = Counter("rag_cache_l1_hits_total", "Total number of L1 cache hits", ["tenant_id"])
L1_MISSES = Counter("rag_cache_l1_misses_total", "Total number of L1 cache misses", ["tenant_id"])

# L2 (Semantic) Cache Metrics
L2_HITS = Counter("rag_cache_l2_hits_total", "Total number of L2 cache hits", ["tenant_id"])
L2_MISSES = Counter("rag_cache_l2_misses_total", "Total number of L2 cache misses", ["tenant_id"])

# ROI / Savings Metrics
LLM_CALLS_SAVED = Counter(
    "rag_cache_llm_calls_saved_total",
    "Total number of LLM calls saved by L2 cache hits",
    ["tenant_id"],
)
RETRIEVER_CALLS_SAVED = Counter(
    "rag_cache_retriever_calls_saved_total",
    "Total number of retriever calls saved by L1 cache hits",
    ["tenant_id"],
)

# Latency Histograms
FAISS_SEARCH_LATENCY = Histogram(
    "rag_cache_faiss_search_latency_seconds",
    "Latency of FAISS search operations in seconds",
    ["tenant_id"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)

REDIS_LATENCY = Histogram(
    "rag_cache_redis_latency_seconds",
    "Latency of Redis operations in seconds",
    ["tenant_id", "operation"],
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5),
)

EMBEDDING_LATENCY = Histogram(
    "rag_cache_embedding_latency_seconds",
    "Latency of embedding generation in seconds",
    ["tenant_id"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)

DECISION_ENGINE_LATENCY = Histogram(
    "rag_cache_decision_engine_latency_seconds",
    "Latency of the Decision Engine evaluation in seconds",
    ["tenant_id"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
)

TOTAL_REQUEST_LATENCY = Histogram(
    "rag_cache_total_request_latency_seconds",
    "Total request latency of ContextCache run in seconds",
    ["tenant_id", "cache_hit"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)

DECISION_ENGINE_REJECTIONS = Counter(
    "rag_cache_decision_engine_rejections_total",
    "Total number of rejections by the Decision Engine, categorized by reason",
    ["tenant_id", "reason"],
)

# Global Gauges
CACHE_ENTRIES = Gauge("rag_cache_entries_total", "Total number of keys/entries in Redis")

REDIS_MEMORY_BYTES = Gauge("rag_cache_redis_memory_bytes", "Redis used memory in bytes")

FAISS_VECTORS_TOTAL = Gauge(
    "rag_cache_faiss_vectors_total", "Total number of vectors in the FAISS index"
)

# ---------------------------------------------------------
# Server Control & Helpers
# ---------------------------------------------------------

_prometheus_started = False


def start_metrics_server(port: int = 8000, addr: str = "0.0.0.0") -> None:
    """Starts the Prometheus metrics HTTP server if not already started."""
    global _prometheus_started
    if not _prometheus_started:
        start_http_server(port, addr=addr)
        _prometheus_started = True


def get_tenant_label(tenant_id: Optional[str]) -> str:
    """Coerces tenant_id to 'global' if None to satisfy Prometheus label requirements."""
    return tenant_id if tenant_id is not None else "global"
