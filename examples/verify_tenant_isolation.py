import os
import sys

import redis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_cache import ContextCache as UnifiedContextCache


def mock_retriever(query):
    return ["doc_leave_policy"]


def mock_llm(query, doc_ids):
    # Returns tenant-specific response if keyword in query, or default response
    if "leave policy" in query:
        return "20 days of annual paid leave."
    return "This is a generic LLM response."


def verify_tenant_isolation():
    print("==========================================================")
    print("       ContextCache Tenant Isolation Verification Test        ")
    print("==========================================================")

    # Clear Redis database to have clean state
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushdb()
    print("Cleared Redis DB.")

    # Remove old index file if exists
    if os.path.exists("faiss_index.bin"):
        try:
            os.remove("faiss_index.bin")
        except Exception:
            pass

    # Instantiate cache facade
    cache = UnifiedContextCache(use_faiss=True, debug=True)

    # 1. Store cache under Tenant A
    print("\n--- 1. Querying under Tenant A (Expected: MISS & STORE) ---")
    res_a1 = cache.run(
        query="What is our leave policy?",
        retriever=mock_retriever,
        llm=lambda q, docs: "Tenant A Response: 20 days PTO.",
        tenant_id="tenant_a",
        scope="tenant",
    )
    print(f"Result A1: {res_a1}")
    assert res_a1["cache_hit"] is False, "A1 should be a miss"
    assert res_a1["answer"] == "Tenant A Response: 20 days PTO."

    # 2. Query identical query under Tenant B
    print("\n--- 2. Querying under Tenant B (Expected: MISS - Strict Isolation) ---")
    res_b1 = cache.run(
        query="What is our leave policy?",
        retriever=mock_retriever,
        llm=lambda q, docs: "Tenant B Response: Unlimited PTO.",
        tenant_id="tenant_b",
        scope="tenant",
    )
    print(f"Result B1: {res_b1}")
    assert res_b1["cache_hit"] is False, "B1 must be a cache miss due to tenant isolation"
    assert res_b1["answer"] == "Tenant B Response: Unlimited PTO."

    # 3. Query identical query under Tenant A again
    print("\n--- 3. Querying under Tenant A again (Expected: HIT) ---")
    res_a2 = cache.run(
        query="What is our leave policy?",
        retriever=mock_retriever,
        llm=lambda q, docs: "Unused LLM response",
        tenant_id="tenant_a",
        scope="tenant",
    )
    print(f"Result A2: {res_a2}")
    assert res_a2["cache_hit"] is True, "A2 should be a cache hit"
    assert res_a2["answer"] == "Tenant A Response: 20 days PTO."

    # 4. Query identical query under Tenant B again
    print("\n--- 4. Querying under Tenant B again (Expected: HIT) ---")
    res_b2 = cache.run(
        query="What is our leave policy?",
        retriever=mock_retriever,
        llm=lambda q, docs: "Unused LLM response",
        tenant_id="tenant_b",
        scope="tenant",
    )
    print(f"Result B2: {res_b2}")
    assert res_b2["cache_hit"] is True, "B2 should be a cache hit"
    assert res_b2["answer"] == "Tenant B Response: Unlimited PTO."

    # 5. Global scope test
    print("\n--- 5. Global Scope Caching Test (Expected: Shared Access) ---")
    # Store global query under admin (tenant_id = None, scope = "global")
    res_g1 = cache.run(
        query="What is the capital of France?",
        retriever=lambda q: ["doc_france"],
        llm=lambda q, docs: "Paris.",
        tenant_id=None,
        scope="global",
    )
    print(f"Result G1 (Global store): {res_g1}")
    assert res_g1["cache_hit"] is False

    # Fetch global query under Tenant A
    print("\nFetch Global query under Tenant A (Expected: HIT)")
    res_ag = cache.run(
        query="What is the capital of France?",
        retriever=lambda q: ["doc_france"],
        llm=lambda q, docs: "Wrong answer.",
        tenant_id="tenant_a",
        scope="global",
    )
    print(f"Result AG: {res_ag}")
    assert res_ag["cache_hit"] is True, "Should hit global cache entry"
    assert res_ag["answer"] == "Paris."

    # Fetch global query under Tenant B
    print("\nFetch Global query under Tenant B (Expected: HIT)")
    res_bg = cache.run(
        query="What is the capital of France?",
        retriever=lambda q: ["doc_france"],
        llm=lambda q, docs: "Wrong answer.",
        tenant_id="tenant_b",
        scope="global",
    )
    print(f"Result BG: {res_bg}")
    assert res_bg["cache_hit"] is True, "Should hit global cache entry"
    assert res_bg["answer"] == "Paris."

    # 6. Verify Redis keys schema
    print("\n--- 6. Checking Redis database key namespacing ---")
    all_keys = [k.decode("utf-8") for k in r.keys("*")]
    print("Found keys in Redis:")
    for k in sorted(all_keys):
        print(f"  - {k}")

    # Assert namespaces exist
    assert any(
        k.startswith("tenant:tenant_a:l1:") for k in all_keys
    ), "L1 keys for tenant_a must be prefixed"
    assert any(
        k.startswith("tenant:tenant_b:l1:") for k in all_keys
    ), "L1 keys for tenant_b must be prefixed"
    assert any(
        k.startswith("tenant:tenant_a:l2:") for k in all_keys
    ), "L2 keys for tenant_a must be prefixed"
    assert any(
        k.startswith("tenant:tenant_b:l2:") for k in all_keys
    ), "L2 keys for tenant_b must be prefixed"
    assert any(
        k.startswith("tenant:tenant_a:map:") for k in all_keys
    ), "Map keys for tenant_a must be prefixed"
    assert any(
        k.startswith("tenant:tenant_b:map:") for k in all_keys
    ), "Map keys for tenant_b must be prefixed"

    # Assert global key structure
    assert any(k.startswith("l1:") for k in all_keys), "Global L1 key should not have tenant prefix"
    assert any(k.startswith("l2:") for k in all_keys), "Global L2 key should not have tenant prefix"

    print("\n==========================================================")
    print("        🎉 ALL TENANT ISOLATION TESTS PASSED! 🎉           ")
    print("==========================================================")


if __name__ == "__main__":
    verify_tenant_isolation()
