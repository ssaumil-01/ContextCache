import os
import sys
import time

# Add the project root to the python path so it runs out-of-the-box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# rag_cache imports
from rag_cache.core.cache import GenerationCache, RetrievalCache
from rag_cache.core.decision_engine import DecisionEngine, DecisionRuleConfig
from rag_cache.core.default_intent import RuleBasedIntentClassifier
from rag_cache.core.models import ResolveInput, StoreInput
from rag_cache.integrations.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_cache.integrations.key_value_stores.redis import RedisKeyValueStore
from rag_cache.integrations.vector_stores.in_memory import InMemoryVectorStore

# ---------------------------------------------------------
# 1. Initialize Cache Layers
# ---------------------------------------------------------
print("Loading SentenceTransformer weights (may take a moment on first run)...")

# Setup synchronized DBs (Redis handles its own eviction)
vector_db = InMemoryVectorStore()
shared_kv = RedisKeyValueStore()

# L1: Retrieval Cache (Lightweight string lookup wrapper for KV Store)
retrieval_cache = RetrievalCache(kv_store=shared_kv)

# L2: Generation Cache (Complex context-aware semantic caching)
generation_cache = GenerationCache(
    embedder=SentenceTransformerEmbedder(),
    vector_store=vector_db,
    kv_store=shared_kv,
    intent_classifier=RuleBasedIntentClassifier(),
    decision_engine=DecisionEngine(config=DecisionRuleConfig(min_embedding_similarity=0.65)),
    debug_mode=True,  # <--- Explicit Tracing Enabled
)


# ---------------------------------------------------------
# 2. Fake Application Components
# ---------------------------------------------------------
def fake_retriever(query: str) -> list[str]:
    """Mocks fetching document IDs from your main Document VectorDB."""
    time.sleep(0.5)  # Network latency to VectorDB
    if "attendance" in query:
        return ["doc_att_1", "doc_att_2"]
    elif "grading" in query:
        return ["doc_grade_1", "doc_grade_2"]
    else:
        return ["doc_default"]


def mock_llm_generation(query: str, doc_ids: list[str]) -> str:
    """Mocks waiting for Anthropic/OpenAI to generate a response."""
    time.sleep(1.5)  # Network latency & Token Streaming
    return f"Based on context from {doc_ids}, the resulting answer is: 42."


# ---------------------------------------------------------
# 3. Main RAG Pipeline Execution Flow
# ---------------------------------------------------------
def execute_rag_pipeline(user_query: str) -> str:
    print(f"\n[Incoming Query]: '{user_query}'")

    # ------------------------------
    # L1 CACHE: Check exact match
    # ------------------------------
    doc_ids = retrieval_cache.resolve(user_query)

    if doc_ids:
        print("⚡ L1 Retrieval Hit! (Skipped Document VectorDB Network Call)")
    else:
        # Actually hit the RAG Document VectorDB
        doc_ids = fake_retriever(user_query)
        # Store silently in L1 Cache
        retrieval_cache.store(user_query, doc_ids)

    # ------------------------------
    # L2 CACHE: Check semantic match
    # ------------------------------
    cache_result, reason = generation_cache.resolve(ResolveInput(query=user_query, doc_ids=doc_ids))

    # Fast Path !
    if cache_result.hit:
        print(f"✅ L2 Cache Hit! ({reason})")
        return cache_result.response

    # Slow Path
    print(f"❌ Cache Miss ({reason})")
    print("... Falling back to expensive LLM Generation ...")

    start_time = time.time()
    llm_answer = mock_llm_generation(user_query, doc_ids)
    print(f"... Done generating in {time.time() - start_time:.2f} seconds.")

    # Securely Save into L2 Cache
    generation_cache.store(StoreInput(query=user_query, response=llm_answer, doc_ids=doc_ids))

    return llm_answer


# ---------------------------------------------------------
# Run Example Workflow
# ---------------------------------------------------------
def main():
    print("\nStarting Main App...")

    # First request: Empty cache, will generate via LLM (~2.0 seconds total)
    execute_rag_pipeline("What is attendance policy?")

    # Second request: Semantically similar but exact string changes!
    # L1 Retrieval Cache MISSES (Strings differ!) -> Hits external Document DB for chunks.
    # L2 Generation Cache HITS (Semantics are 0.94 matching, Context is exact match!)
    execute_rag_pipeline("Explain attendance policy")

    # Third request: Exact string match of earlier query!
    # L1 Retrieval Cache HITS (Bypasses VectorDB Document Fetch completely!)
    # L2 Generation Cache HITS (Bypasses LLM Generation completely!)
    execute_rag_pipeline("What is attendance policy?")

    print("\n[Observability] L1 Retrieval Performance Metrics:")
    print(retrieval_cache.get_stats())

    print("\n[Observability] L2 Generation Performance Metrics:")
    print(generation_cache.get_stats())


if __name__ == "__main__":
    main()
