import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_cache.core.cache import GenerationCache, RetrievalCache
from rag_cache.core.decision_engine import DecisionEngine, DecisionRuleConfig
from rag_cache.core.default_intent import RuleBasedIntentClassifier
from rag_cache.core.models import ResolveInput, StoreInput
from rag_cache.integrations.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_cache.integrations.key_value_stores.redis import RedisKeyValueStore
from rag_cache.integrations.vector_stores.in_memory import InMemoryVectorStore

# ---------------------------------------------------------
# Workload Generator Setup
# ---------------------------------------------------------
# Define a set of topics, each with a base query, a semantic variant, and a doc ID.
TOPICS = [
    {
        "base": "What is the company refund policy?",
        "variant": "How do I get a refund?",
        "doc": "policy_doc_refund",
        "intent": "informational",
    },
    {
        "base": "Tell me about the engineering team.",
        "variant": "Who works in engineering?",
        "doc": "team_doc_eng",
        "intent": "informational",
    },
    {
        "base": "What is the remote work policy?",
        "variant": "Can I work from home?",
        "doc": "remote_doc_work",
        "intent": "informational",
    },
    {
        "base": "Explain the vacation policy.",
        "variant": "How many PTO days do I get?",
        "doc": "pto_doc_vacation",
        "intent": "informational",
    },
    {
        "base": "How do I reset my password?",
        "variant": "I forgot my password, how to recover it?",
        "doc": "security_doc_pass",
        "intent": "informational",
    },
    {
        "base": "What is the office dress code?",
        "variant": "What are we allowed to wear at work?",
        "doc": "office_doc_dress",
        "intent": "informational",
    },
    {
        "base": "How do I submit an expense report?",
        "variant": "Where do I file my expenses?",
        "doc": "finance_doc_expense",
        "intent": "informational",
    },
    {
        "base": "What are the working hours?",
        "variant": "When do I need to be in the office?",
        "doc": "office_doc_hours",
        "intent": "informational",
    },
    {
        "base": "Is parking free at the office?",
        "variant": "Do I have to pay to park my car at work?",
        "doc": "office_doc_parking",
        "intent": "informational",
    },
    {
        "base": "What is the health insurance policy?",
        "variant": "Tell me about health benefits.",
        "doc": "hr_doc_health",
        "intent": "informational",
    },
]

# We will generate a list of novel queries that are completely unrelated to these topics
NOVEL_QUERIES = [
    ("How do I cook pasta?", "food_doc_pasta"),
    ("Who was the first man on the moon?", "history_doc_moon"),
    ("What is the capital of France?", "geo_doc_france"),
    ("How does a diesel engine work?", "mech_doc_diesel"),
    ("What is the distance to the sun?", "space_doc_sun"),
    ("Why is the sky blue?", "science_doc_sky"),
    ("Who wrote Hamlet?", "lit_doc_hamlet"),
    ("How to plant a tree?", "garden_doc_tree"),
    ("What is quantum computing?", "physics_doc_quantum"),
    ("Explain photosyntheis in plants.", "bio_doc_plant"),
    ("How do I learn Python?", "coding_doc_python"),
    ("What is the highest mountain?", "geo_doc_everest"),
    ("How to brew coffee?", "food_doc_coffee"),
    ("What is inflation in economics?", "finance_doc_inflation"),
    ("Who painted the Mona Lisa?", "art_doc_monalisa"),
]


def generate_workload(total_queries=1000):
    """
    Generates a list of 1000 queries with:
    - 30% exact repeats (300)
    - 40% semantic variants (400)
    - 30% novel queries (300)
    """
    import random

    random.seed(42)  # Consistent workload across thresholds

    workload = []
    seen_bases = []

    # Pre-populate some history so we can do repeats and variants
    for t in TOPICS:
        workload.append({"query": t["base"], "doc_ids": [t["doc"]], "type": "initial"})
        seen_bases.append(t)

    for q, d in NOVEL_QUERIES[:5]:
        workload.append({"query": q, "doc_ids": [d], "type": "initial"})

    # Generate the remaining queries
    exact_count = 0
    variant_count = 0
    novel_count = 0

    target_exact = int(total_queries * 0.3)
    target_variant = int(total_queries * 0.4)
    target_novel = total_queries - len(workload) - target_exact - target_variant

    while len(workload) < total_queries:
        r = random.random()

        # Exact repeat (30%)
        if r < 0.3 and exact_count < target_exact:
            # Pick a previously seen base query
            chosen = random.choice(seen_bases)
            workload.append(
                {"query": chosen["base"], "doc_ids": [chosen["doc"]], "type": "exact_repeat"}
            )
            exact_count += 1

        # Semantic variant (40%)
        elif r < 0.7 and variant_count < target_variant:
            # Pick a previously seen base topic, but execute its variant
            chosen = random.choice(seen_bases)
            workload.append(
                {"query": chosen["variant"], "doc_ids": [chosen["doc"]], "type": "semantic_variant"}
            )
            variant_count += 1

        # Novel query (30%)
        else:
            # Generate a new query
            q_text, q_doc = random.choice(NOVEL_QUERIES)
            # Add some randomness to make them unique
            suffix = f" (run {random.randint(1, 100000)})"
            workload.append({"query": q_text + suffix, "doc_ids": [q_doc], "type": "novel"})
            novel_count += 1

    return workload


# ---------------------------------------------------------
# Latency Simulation Setup
# ---------------------------------------------------------
# To prevent CPU-bound embedding generation from slowing down the benchmark,
# we pre-compute all unique embeddings once, making the search runs blazing fast.
def precompute_embeddings(workload, embedder):
    print("Pre-computing embeddings for all unique query strings...")
    unique_queries = list(set([item["query"] for item in workload]))
    print(f"Total unique queries: {len(unique_queries)}")

    embedding_cache = {}
    start = time.time()
    # Batch compute to make it even faster
    for q in unique_queries:
        embedding_cache[q] = embedder.embed_query(q)
    print(f"Pre-computation completed in {time.time() - start:.2f} seconds.")
    return embedding_cache


# Simulated network times (consistent with user specs)
RETRIEVER_LATENCY = 0.30  # 300 ms
LLM_LATENCY = 1.20  # 1200 ms


def simulate_retriever(query):
    return ["mock_doc_id"]


def simulate_llm(query, doc_ids):
    return "Simulated LLM response"


def calculate_percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return sorted_data[int(idx)]
    return sorted_data[low] * (high - idx) + sorted_data[high] * (idx - low)


# ---------------------------------------------------------
# Benchmarking Engine
# ---------------------------------------------------------
def run_benchmark_for_store(store_type, workload, embedding_cache, embedder):
    print(f"\n>>> Running Benchmark for {store_type} (Threshold = 0.65) ...")

    shared_kv = RedisKeyValueStore()
    shared_kv.client.flushdb()  # Clear KV cache

    if store_type == "FAISS":
        from rag_cache.integrations.vector_stores.faiss import FaissVectorStore

        if os.path.exists("faiss_index.bin"):
            try:
                os.remove("faiss_index.bin")
            except Exception:
                pass
        vector_db = FaissVectorStore(
            dimension=384, kv_store=shared_kv, index_filepath="faiss_index.bin"
        )
    else:
        vector_db = InMemoryVectorStore()

    # Configure Decision Engine with 0.65 threshold
    config = DecisionRuleConfig(min_embedding_similarity=0.65)
    decision_engine = DecisionEngine(config=config)

    # L1 and L2 setups
    retrieval_cache = RetrievalCache(kv_store=shared_kv)
    generation_cache = GenerationCache(
        embedder=embedder,
        vector_store=vector_db,
        kv_store=shared_kv,
        intent_classifier=RuleBasedIntentClassifier(),
        decision_engine=decision_engine,
        debug_mode=False,
    )

    # Override embedder to use precomputed values (bypassing CPU model calls)
    def fast_embed(text):
        return embedding_cache[text]

    generation_cache.embedder.embed_query = fast_embed

    latencies = []
    vector_search_latencies = []
    l1_hits = 0
    l2_hits = 0
    retriever_calls = 0
    llm_calls = 0

    # Instrument search latency
    original_search = vector_db.search

    def timed_search(vector, top_k=5, *args, **kwargs):
        s = time.perf_counter()
        res = original_search(vector, top_k, *args, **kwargs)
        vector_search_latencies.append(time.perf_counter() - s)
        return res

    vector_db.search = timed_search

    for idx, q_item in enumerate(workload):
        q = q_item["query"]
        doc_ids = q_item["doc_ids"]

        start_time = time.perf_counter()

        # Step 1: L1 Retrieval check
        cached_docs = retrieval_cache.resolve(q)
        retriever_time = 0.0
        if cached_docs:
            l1_hits += 1
            doc_ids = cached_docs
        else:
            doc_ids = simulate_retriever(q)
            retriever_time = RETRIEVER_LATENCY
            retriever_calls += 1
            retrieval_cache.store(q, doc_ids)

        # Step 2: L2 Generation check
        cache_result, _ = generation_cache.resolve(ResolveInput(query=q, doc_ids=doc_ids))

        llm_time = 0.0
        if cache_result.hit:
            l2_hits += 1
        else:
            ans = simulate_llm(q, doc_ids)
            llm_time = LLM_LATENCY
            llm_calls += 1
            generation_cache.store(StoreInput(query=q, response=ans, doc_ids=doc_ids))

        latency = (time.perf_counter() - start_time) + retriever_time + llm_time
        latencies.append(latency * 1000.0)  # Convert to ms

    total_runtime = sum(latencies) / 1000.0  # back to seconds
    avg_latency = sum(latencies) / len(latencies)
    avg_search_latency_ms = (
        (sum(vector_search_latencies) / len(vector_search_latencies)) * 1000.0
        if vector_search_latencies
        else 0.0
    )

    # Compute memory footprint in KB
    if store_type == "FAISS":
        num_vectors = vector_db.index.ntotal
        vector_memory = num_vectors * 384 * 4  # flat index float32 arrays
        redis_mapping_memory = num_vectors * 128  # approximate overhead
        memory_usage_kb = (vector_memory + redis_mapping_memory) / 1024.0
    else:
        import sys

        memory_size = sys.getsizeof(vector_db.storage)
        for item in vector_db.storage:
            memory_size += (
                sys.getsizeof(item)
                + sys.getsizeof(item.get("id", ""))
                + sys.getsizeof(item.get("vector", []))
            )
        memory_usage_kb = memory_size / 1024.0

    p50 = calculate_percentile(latencies, 0.50)
    p95 = calculate_percentile(latencies, 0.95)
    p99 = calculate_percentile(latencies, 0.99)

    total_queries = len(workload)
    l1_rate = l1_hits / total_queries
    l2_rate = l2_hits / total_queries
    llm_saved = total_queries - llm_calls
    retriever_saved = total_queries - retriever_calls

    results = {
        "store_type": store_type,
        "total_queries": total_queries,
        "l1_hits": l1_hits,
        "l2_hits": l2_hits,
        "l1_hit_rate": l1_rate,
        "l2_hit_rate": l2_rate,
        "retriever_calls_saved": retriever_saved,
        "llm_calls_saved": llm_saved,
        "total_runtime_sec": total_runtime,
        "avg_latency_ms": avg_latency,
        "avg_search_latency_ms": avg_search_latency_ms,
        "memory_usage_kb": memory_usage_kb,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }

    print(f"    - Total Runtime       : {total_runtime:.2f} seconds")
    print(f"    - Avg Latency         : {avg_latency:.2f} ms")
    print(f"    - Avg Search Latency  : {avg_search_latency_ms:.4f} ms")
    print(f"    - Memory Footprint    : {memory_usage_kb:.2f} KB")
    print(f"    - L1 Hits             : {l1_hits} ({l1_rate*100:.1f}%)")
    print(f"    - L2 Hits             : {l2_hits} ({l2_rate*100:.1f}%)")
    print(f"    - LLM Calls Saved     : {llm_saved}")
    return results


def main():
    print("==========================================================")
    # 1. Generate workload
    workload = generate_workload(1000)
    print(f"Generated workload containing {len(workload)} queries.")

    # 2. Pre-compute embeddings
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    embedding_cache = precompute_embeddings(workload, embedder)

    # 3. Compare stores
    stores = ["InMemory", "FAISS"]
    all_results = []

    for st in stores:
        res = run_benchmark_for_store(st, workload, embedding_cache, embedder)
        all_results.append(res)

    print("\n================ VECTOR STORE SUMMARY COMPARISON ================")
    print(
        f"{'Store Type':<12} | {'L1 Hit %':<8} | {'L2 Hit %':<8} | {'LLM Saved':<9} | {'Avg Lat':<7} | {'Avg Search':<10} | {'Memory (KB)':<11} | {'P50':<6} | {'P95':<6}"
    )
    print("-" * 105)
    for r in all_results:
        print(
            f"{r['store_type']:<12} | {r['l1_hit_rate']*100:<7.1f}% | {r['l2_hit_rate']*100:<7.1f}% | {r['llm_calls_saved']:<9} | {r['avg_latency_ms']:<7.1f} | {r['avg_search_latency_ms']:<10.4f} | {r['memory_usage_kb']:<11.2f} | {r['p50_ms']:<6.1f} | {r['p95_ms']:<6.1f}"
        )
    print("=================================================================")

    # Export reports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "../tools/calibration/benchmark_report.json")
    csv_path = os.path.join(script_dir, "../tools/calibration/benchmark_report.csv")

    # Ensure directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # JSON export
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nExported JSON report to {json_path}")

    # CSV export
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Store Type",
                "L1 Hit Rate",
                "L2 Hit Rate",
                "Retriever Calls Saved",
                "LLM Calls Saved",
                "Total Runtime (s)",
                "Avg Latency (ms)",
                "Avg Search Latency (ms)",
                "Memory Usage (KB)",
                "P50 (ms)",
                "P95 (ms)",
                "P99 (ms)",
            ]
        )
        for r in all_results:
            writer.writerow(
                [
                    r["store_type"],
                    f"{r['l1_hit_rate']:.4f}",
                    f"{r['l2_hit_rate']:.4f}",
                    r["retriever_calls_saved"],
                    r["llm_calls_saved"],
                    f"{r['total_runtime_sec']:.4f}",
                    f"{r['avg_latency_ms']:.4f}",
                    f"{r['avg_search_latency_ms']:.4f}",
                    f"{r['memory_usage_kb']:.4f}",
                    f"{r['p50_ms']:.4f}",
                    f"{r['p95_ms']:.4f}",
                    f"{r['p99_ms']:.4f}",
                ]
            )
    print(f"Exported CSV report to {csv_path}")


if __name__ == "__main__":
    main()
