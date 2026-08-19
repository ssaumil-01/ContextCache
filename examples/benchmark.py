import os
import sys
import time

# Add the project root to the python path so it runs out-of-the-box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_cache.core.cache import GenerationCache, RetrievalCache
from rag_cache.core.decision_engine import DecisionEngine, DecisionRuleConfig
from rag_cache.core.default_intent import RuleBasedIntentClassifier
from rag_cache.core.models import ResolveInput, StoreInput
from rag_cache.integrations.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_cache.integrations.key_value_stores.redis import RedisKeyValueStore
from rag_cache.integrations.vector_stores.in_memory import InMemoryVectorStore

# ---------------------------------------------------------
# Sample Dataset
# ---------------------------------------------------------
# A realistic mix of exact repeats, semantic duplicates, and new topics
SAMPLE_QUERIES = [
    "What is the company refund policy?",  # 1. New (Baseline Cost)
    "How do I get a refund?",  # 2. Semantic Match to 1 (L2 Hit)
    "What is the company refund policy?",  # 3. Exact Match to 1 (L1 Hit)
    "Tell me about the engineering team.",  # 4. New (Baseline Cost)
    "Who works in engineering?",  # 5. Semantic Match to 4 (L2 Hit)
    "What is the remote work policy?",  # 6. New (Baseline Cost)
    "Can I work from home?",  # 7. Semantic Match to 6 (L2 Hit)
    "Explain the vacation policy.",  # 8. New (Baseline Cost)
    "How many PTO days do I get?",  # 9. Semantic Match to 8 (L2 Hit)
    "Tell me about the engineering team.",  # 10. Exact Match to 4 (L1 Hit)
]


# ---------------------------------------------------------
# Fake RAG Network Infrastructure (Latencies)
# ---------------------------------------------------------
def fake_retriever(query: str):
    time.sleep(0.3)  # Simulate Pinecone/Weaviate network latency
    if "refund" in query.lower():
        return ["policy_doc"]
    elif "engineer" in query.lower():
        return ["team_doc"]
    elif "work" in query.lower() or "home" in query.lower():
        return ["remote_doc"]
    else:
        return ["pto_doc"]


def mock_llm_generation(query: str, doc_ids: list):
    time.sleep(1.2)  # Simulate OpenAI/Anthropic generation latency
    return f"Generated answer"


# ---------------------------------------------------------
# Evaluators
# ---------------------------------------------------------
def run_baseline():
    """Runs a standard RAG pipeline without any caching."""
    print(f"\n[1/2] Running Baseline Pipeline (No Cache) over {len(SAMPLE_QUERIES)} queries...")
    start_time = time.time()

    for q in SAMPLE_QUERIES:
        docs = fake_retriever(q)
        ans = mock_llm_generation(q, docs)

    total_time = time.time() - start_time
    print(f"      -> Completed in {total_time:.2f} seconds")
    return total_time


def run_with_cache():
    """Runs the exact same workload masked by the Duel-Layer ContextCache."""
    print(f"\n[2/2] Running Optimized Pipeline (ContextCache) over {len(SAMPLE_QUERIES)} queries...")

    print("      -> Initializing Semantic Embedder...")
    shared_kv = RedisKeyValueStore()
    shared_kv.client.flushdb()
    retrieval_cache = RetrievalCache(kv_store=shared_kv)
    generation_cache = GenerationCache(
        embedder=SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2"),
        vector_store=InMemoryVectorStore(),
        kv_store=shared_kv,
        intent_classifier=RuleBasedIntentClassifier(),
        decision_engine=DecisionEngine(config=DecisionRuleConfig(min_embedding_similarity=0.65)),
        debug_mode=False,
    )

    start_time = time.time()

    for q in SAMPLE_QUERIES:
        # L1: Retrieval
        doc_ids = retrieval_cache.resolve(q)
        if not doc_ids:
            doc_ids = fake_retriever(q)
            retrieval_cache.store(q, doc_ids)

        # L2: Generation
        cache_result, _ = generation_cache.resolve(ResolveInput(query=q, doc_ids=doc_ids))
        if not cache_result.hit:
            ans = mock_llm_generation(q, doc_ids)
            generation_cache.store(StoreInput(query=q, response=ans, doc_ids=doc_ids))

    total_time = time.time() - start_time
    print(f"      -> Completed in {total_time:.2f} seconds")

    return total_time, retrieval_cache, generation_cache


# ---------------------------------------------------------
# Main Reporter
# ---------------------------------------------------------
def main():
    print("===================================================")
    print("           ContextCache Load Benchmark                 ")
    print("===================================================")

    baseline_time = run_baseline()
    cache_time, r_cache, g_cache = run_with_cache()

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Total Simulated Queries: {len(SAMPLE_QUERIES)}")
    print(f"Baseline System Latency: {baseline_time:.2f} seconds")
    print(f"ContextCache System Latency: {cache_time:.2f} seconds")

    speedup = baseline_time / cache_time if cache_time > 0 else 0
    print(f"Speedup Factor:          {speedup:.2f}x faster")
    print(f"Net Time Saved:          {baseline_time - cache_time:.2f} seconds")

    print("\n--- Cache Routing Analytics ---")
    l1_stats = r_cache.get_stats()
    l2_stats = g_cache.get_stats()

    print(f"L1 (Exact Match) Hits:     {l1_stats['hits']}")
    print(f"L2 (Semantic) Hits:        {l2_stats['hits']}")

    # 4 misses out of 10 means 6 hits total. 6 / 10 = 60% LLM Call Elimination
    elimination_rate = ((l1_stats["hits"] + l2_stats["hits"]) / len(SAMPLE_QUERIES)) * 100
    print(f"Total API Calls Mitigated: {elimination_rate:.0f}% Elimination Rate")
    print("===================================================\n")


if __name__ == "__main__":
    main()
