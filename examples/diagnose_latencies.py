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

SAMPLE_QUERIES = [
    "What is the company refund policy?",  # 1. New (Baseline Cost)
    "How do I get a refund?",  # 2. Semantic Match to 1 (L2 Hit - Miss due to low sim)
    "What is the company refund policy?",  # 3. Exact Match to 1 (L1 Hit)
    "Tell me about the engineering team.",  # 4. New (Baseline Cost)
    "Who works in engineering?",  # 5. Semantic Match to 4 (L2 Hit - Miss due to low sim)
    "What is the remote work policy?",  # 6. New (Baseline Cost)
    "Can I work from home?",  # 7. Semantic Match to 6 (L2 Hit - Miss due to low sim)
    "Explain the vacation policy.",  # 8. New (Baseline Cost)
    "How many PTO days do I get?",  # 9. Semantic Match to 8 (L2 Hit - Miss due to low sim)
    "Tell me about the engineering team.",  # 10. Exact Match to 4 (L1 Hit)
]


def fake_retriever(query: str):
    time.sleep(0.3)
    if "refund" in query.lower():
        return ["policy_doc"]
    elif "engineer" in query.lower():
        return ["team_doc"]
    elif "work" in query.lower() or "home" in query.lower():
        return ["remote_doc"]
    else:
        return ["pto_doc"]


def mock_llm_generation(query: str, doc_ids: list):
    time.sleep(1.2)
    return f"Generated answer"


def run_diagnostics():
    print("Initializing components...")
    shared_kv = RedisKeyValueStore()
    shared_kv.client.flushdb()

    # Trace instrumentation helpers
    redis_times = []
    embed_times = []
    vector_times = []
    decision_times = []

    # Monkey-patch RedisKeyValueStore to measure times
    original_get = shared_kv.get
    original_set = shared_kv.set

    def timed_get(key):
        start = time.perf_counter()
        res = original_get(key)
        redis_times.append(time.perf_counter() - start)
        return res

    def timed_set(key, value, ttl_seconds=None):
        start = time.perf_counter()
        original_set(key, value, ttl_seconds)
        redis_times.append(time.perf_counter() - start)

    shared_kv.get = timed_get
    shared_kv.set = timed_set

    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    original_embed = embedder.embed_query

    def timed_embed(text):
        start = time.perf_counter()
        res = original_embed(text)
        embed_times.append(time.perf_counter() - start)
        return res

    embedder.embed_query = timed_embed

    vector_store = InMemoryVectorStore()
    original_search = vector_store.search

    def timed_search(vector, top_k=5):
        start = time.perf_counter()
        res = original_search(vector, top_k)
        vector_times.append(time.perf_counter() - start)
        return res

    vector_store.search = timed_search

    decision_engine = DecisionEngine(config=DecisionRuleConfig(min_embedding_similarity=0.65))
    original_evaluate = decision_engine.evaluate_candidates

    def timed_evaluate(*args, **kwargs):
        start = time.perf_counter()
        res = original_evaluate(*args, **kwargs)
        decision_times.append(time.perf_counter() - start)
        return res

    decision_engine.evaluate_candidates = timed_evaluate

    retrieval_cache = RetrievalCache(kv_store=shared_kv)
    generation_cache = GenerationCache(
        embedder=embedder,
        vector_store=vector_store,
        kv_store=shared_kv,
        intent_classifier=RuleBasedIntentClassifier(),
        decision_engine=decision_engine,
        debug_mode=False,
    )

    print("\nStarting execution trace...")
    print(
        f"{'Query':<36} | {'Total':<6} | {'Embed':<6} | {'Redis':<6} | {'VecSrc':<6} | {'DecEng':<6} | {'Retr':<6} | {'LLM':<6} | {'Result':<6}"
    )
    print("-" * 105)

    for idx, q in enumerate(SAMPLE_QUERIES):
        query_start = time.perf_counter()

        # Reset metric times for this query
        redis_times.clear()
        embed_times.clear()
        vector_times.clear()
        decision_times.clear()

        # L1 Retrieval check
        doc_ids = retrieval_cache.resolve(q)
        retriever_time = 0.0
        if not doc_ids:
            retriever_start = time.perf_counter()
            doc_ids = fake_retriever(q)
            retriever_time = time.perf_counter() - retriever_start
            retrieval_cache.store(q, doc_ids)

        # L2 Generation check
        cache_result, _ = generation_cache.resolve(ResolveInput(query=q, doc_ids=doc_ids))

        llm_time = 0.0
        outcome = "L2 HIT"
        if not cache_result.hit:
            llm_start = time.perf_counter()
            ans = mock_llm_generation(q, doc_ids)
            llm_time = time.perf_counter() - llm_start
            generation_cache.store(StoreInput(query=q, response=ans, doc_ids=doc_ids))
            outcome = "MISS"
        elif idx in [2, 9]:
            outcome = "L1 HIT"  # L1 hit bypassing retriever, L2 also hit

        total_time = time.perf_counter() - query_start

        # Sum up measured component times
        embed_total = sum(embed_times)
        redis_total = sum(redis_times)
        vector_total = sum(vector_times)
        decision_total = sum(decision_times)

        # Format printing
        q_short = q[:34] + ".." if len(q) > 34 else q
        print(
            f"{q_short:<36} | {total_time:.3f} | {embed_total:.3f} | {redis_total:.3f} | {vector_total:.3f} | {decision_total:.3f} | {retriever_time:.3f} | {llm_time:.3f} | {outcome:<6}"
        )


if __name__ == "__main__":
    run_diagnostics()
