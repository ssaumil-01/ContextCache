import os
import random
import sys
import time

import redis

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_cache import ContextCache as UnifiedContextCache


def mock_retriever(query):
    # Simulate retriever latency
    time.sleep(0.3)
    return [f"doc_{hash(query) % 100}"]


def mock_llm(query, doc_ids):
    # Simulate LLM latency
    time.sleep(1.2)
    return f"LLM generated response for query: {query}"


def run_observability_demo():
    print("==========================================================")
    # Clear Redis db for clean demo
    r = redis.Redis(host="localhost", port=6379, db=0)
    try:
        r.flushdb()
        print("Cleared Redis DB.")
    except Exception as e:
        print(f"Warning: Could not connect to Redis at localhost:6379: {e}")

    # Initialize ContextCache with Prometheus metrics server on port 8000
    print("Initializing UnifiedContextCache with Prometheus endpoint on port 8000...")
    cache = UnifiedContextCache(use_faiss=True, prometheus_port=8000, debug=True)

    # List of queries for simulation
    query_pool = [
        "What is the remote work policy?",
        "How do I request time off?",
        "Where can I find my payslip?",
        "What is the company policy on expense reports?",
        "How do I reset my account password?",
    ]

    tenants = ["tenant_a", "tenant_b", None]

    print("\n---------------------------------------------------------")
    print("Observability Demo is now running!")
    print("1. Metrics Endpoint: http://localhost:8000/metrics")
    print("2. Grafana Dashboard: http://localhost:3000 (admin/admin)")
    print("Press Ctrl+C to stop the simulation.")
    print("---------------------------------------------------------\n")

    try:
        iteration = 1
        while True:
            # Randomly pick tenant, scope, and query type
            tenant = random.choice(tenants)
            query = random.choice(query_pool)
            scope = "global" if tenant is None else "tenant"

            # 30% chance to modify the query slightly to simulate semantic variation (L2 hit/miss test)
            if random.random() < 0.3:
                query = query.replace("policy", "procedures").replace(
                    "How do I", "Can you explain how to"
                )

            print(f"[{iteration}] Processing: '{query}' for Tenant: '{tenant}'")

            # Execute cache.run
            res = cache.run(
                query=query, retriever=mock_retriever, llm=mock_llm, tenant_id=tenant, scope=scope
            )

            print(f"      Result Source: {res['source']} | Cache Hit: {res['cache_hit']}")
            print(f"      Response: {res['answer'][:60]}...")
            print("-" * 50)

            # Sleep a short duration before next request to simulate staggered traffic
            time.sleep(random.uniform(0.5, 2.0))
            iteration += 1

    except KeyboardInterrupt:
        print("\nStopping demo simulation...")


if __name__ == "__main__":
    run_observability_demo()
