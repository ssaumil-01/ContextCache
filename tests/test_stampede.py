import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import redis

from rag_cache import ContextCache


def test_cache_stampede_protection():
    """Verify that multiple concurrent queries trigger only 1 retriever and 1 LLM execution."""
    # Initialize cache using mock embedder and local Redis
    cache = ContextCache(use_local_embeddings=False, debug=True, poll_interval_ms=50)

    # Flush Redis DB to ensure clean state
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushdb()

    retriever_calls = 0
    llm_calls = 0

    def mock_retriever(query):
        nonlocal retriever_calls
        time.sleep(0.5)  # Simulate slow database call
        retriever_calls += 1
        return ["doc_1", "doc_2"]

    def mock_llm(query, doc_ids):
        nonlocal llm_calls
        time.sleep(0.5)  # Simulate slow LLM generation
        llm_calls += 1
        return "Coalesced generation answer."

    # Spin up 5 concurrent requests for the same query
    query = "Concurrent stampede query"
    num_threads = 5

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(cache.run, query=query, retriever=mock_retriever, llm=mock_llm)
            for _ in range(num_threads)
        ]
        results = [f.result() for f in futures]

    # Assertions
    assert len(results) == num_threads
    for res in results:
        assert res["answer"] == "Coalesced generation answer."

    # Verify that the downstream systems were called exactly once
    assert retriever_calls == 1, f"Retriever called {retriever_calls} times instead of 1"
    assert llm_calls == 1, f"LLM called {llm_calls} times instead of 1"

    # Verify leader vs follower distribution
    leaders = [r for r in results if r["cache_hit"] is False]
    followers = [r for r in results if r["cache_hit"] is True]

    assert len(leaders) == 1, f"Expected 1 leader, got {len(leaders)}"
    assert (
        len(followers) == num_threads - 1
    ), f"Expected {num_threads - 1} followers, got {len(followers)}"


def test_isolated_tenant_stampedes():
    """Verify concurrent requests from different tenants execute independently and do not lock or cache-leak."""
    cache = ContextCache(use_local_embeddings=False, debug=True, poll_interval_ms=50)

    # Flush Redis DB
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushdb()

    retriever_calls = {}
    llm_calls = {}

    def mock_retriever(query, tenant):
        time.sleep(0.4)
        retriever_calls[tenant] = retriever_calls.get(tenant, 0) + 1
        return [f"doc_{tenant}"]

    def mock_llm(query, doc_ids, tenant):
        time.sleep(0.4)
        llm_calls[tenant] = llm_calls.get(tenant, 0) + 1
        return f"Response for {tenant}"

    # We will trigger concurrent requests for Tenant A and Tenant B
    # Tenant A will have 3 concurrent queries
    # Tenant B will have 2 concurrent queries
    tenants_list = ["tenant_a", "tenant_a", "tenant_a", "tenant_b", "tenant_b"]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for t in tenants_list:
            # Note: lambda capture requires passing t as default argument
            fut = executor.submit(
                cache.run,
                query="Shared isolation query",
                retriever=lambda q, tenant=t: mock_retriever(q, tenant),
                llm=lambda q, docs, tenant=t: mock_llm(q, docs, tenant),
                tenant_id=t,
                scope="tenant",
            )
            futures.append((t, fut))

        results = [(t, f.result()) for t, f in futures]

    # Verify answers are correct
    for t, res in results:
        assert res["answer"] == f"Response for {t}"

    # Verify each tenant calls retriever & LLM exactly once (stampede protected within tenant namespace)
    assert retriever_calls["tenant_a"] == 1
    assert retriever_calls["tenant_b"] == 1
    assert llm_calls["tenant_a"] == 1
    assert llm_calls["tenant_b"] == 1

    # Verify leader vs follower within Tenant A (3 requests -> 1 leader, 2 followers)
    tenant_a_results = [r for t, r in results if t == "tenant_a"]
    leaders_a = [r for r in tenant_a_results if r["cache_hit"] is False]
    followers_a = [r for r in tenant_a_results if r["cache_hit"] is True]
    assert len(leaders_a) == 1
    assert len(followers_a) == 2

    # Verify leader vs follower within Tenant B (2 requests -> 1 leader, 1 follower)
    tenant_b_results = [r for t, r in results if t == "tenant_b"]
    leaders_b = [r for r in tenant_b_results if r["cache_hit"] is False]
    followers_b = [r for r in tenant_b_results if r["cache_hit"] is True]
    assert len(leaders_b) == 1
    assert len(followers_b) == 1
