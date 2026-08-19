import hashlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from rag_cache.core.decision_engine import DecisionEngine
from rag_cache.core.metrics import MetricsTracker
from rag_cache.core.models import CacheEntry, CacheResult, ResolveInput, StoreInput
from rag_cache.interfaces.embedding import Embedder
from rag_cache.interfaces.intent import IntentClassifier
from rag_cache.interfaces.key_value_store import KeyValueStore
from rag_cache.interfaces.vector_store import VectorStore


class RetrievalCache:
    """
    L1 Cache Layer.
    A purely exact-match Key-Value cache mapping identical user queries
    directly to retrieved document IDs, bypassing Document VectorDB lookups.
    """

    def __init__(self, kv_store: KeyValueStore, ttl_seconds: Optional[int] = 86400):
        self.kv_store = kv_store
        self.metrics = MetricsTracker()
        self.ttl_seconds = ttl_seconds

    def _generate_cache_key(self, query: str) -> str:
        return f"retrieval:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"

    def resolve(self, query: str, tenant_id: Optional[str] = None) -> Optional[List[str]]:
        """Fetch previously retrieved Document IDs for this exact string."""
        key = self._generate_cache_key(query)
        data = self.kv_store.get(key, tenant_id=tenant_id)

        from rag_cache.core.observability import (
            L1_HITS,
            L1_MISSES,
            RETRIEVER_CALLS_SAVED,
            get_tenant_label,
        )

        tenant_label = get_tenant_label(tenant_id)

        if data and "doc_ids" in data:
            self.metrics.record_hit()
            L1_HITS.labels(tenant_id=tenant_label).inc()
            RETRIEVER_CALLS_SAVED.labels(tenant_id=tenant_label).inc()
            return data["doc_ids"]

        self.metrics.record_miss()
        L1_MISSES.labels(tenant_id=tenant_label).inc()
        return None

    def store(self, query: str, doc_ids: List[str], tenant_id: Optional[str] = None) -> None:
        """Store the Document IDs associated with this query string."""
        key = self._generate_cache_key(query)
        self.kv_store.set(
            key, {"doc_ids": doc_ids}, ttl_seconds=self.ttl_seconds, tenant_id=tenant_id
        )

    def get_stats(self) -> Dict[str, Any]:
        return self.metrics.get_stats()


class GenerationCache:
    """
    L2 Cache Layer.
    A context-aware, semantic caching system orchestrator.
    Maps a semantic embedding to an LLM response, but strictly requires
    DecisionEngine checks (Context stability, Intent) before hitting.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        kv_store: KeyValueStore,
        intent_classifier: IntentClassifier,
        decision_engine: DecisionEngine,
        bypass_intents: Optional[List[str]] = None,
        debug_mode: bool = False,
        ttl_seconds: Optional[int] = 1209600,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.kv_store = kv_store
        self.intent_classifier = intent_classifier
        self.decision_engine = decision_engine

        self.debug_mode = debug_mode
        if self.debug_mode:
            self.decision_engine.config.debug_mode = True

        self.bypass_intents = bypass_intents or ["action"]
        self.metrics = MetricsTracker()
        self.ttl_seconds = ttl_seconds

    def get_stats(self) -> Dict[str, Any]:
        """Exposes metrics for observability/dashboards."""
        return self.metrics.get_stats()

    def _generate_cache_key(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def resolve(self, resolve_input: ResolveInput) -> Tuple[CacheResult, str]:
        import time

        from rag_cache.core.observability import (
            DECISION_ENGINE_LATENCY,
            EMBEDDING_LATENCY,
            L2_HITS,
            L2_MISSES,
            LLM_CALLS_SAVED,
            get_tenant_label,
        )

        tenant_label = get_tenant_label(resolve_input.tenant_id)

        if self.debug_mode:
            print(f"\n[DEBUG L2] Resolving semantic cache for: '{resolve_input.query}'")

        # Step 1: Classify Intent
        if resolve_input.intent == "default":
            resolve_input.intent = self.intent_classifier.classify(resolve_input.query)

        if resolve_input.intent in self.bypass_intents:
            self.metrics.record_miss()
            L2_MISSES.labels(tenant_id=tenant_label).inc()
            from rag_cache.core.observability import DECISION_ENGINE_REJECTIONS

            DECISION_ENGINE_REJECTIONS.labels(tenant_id=tenant_label, reason="intent").inc()
            return CacheResult(hit=False), "Miss: Intent configured for cache bypass."

        # Step 2: Generate Embedding
        start_emb = time.time()
        query_vector = self.embedder.embed_query(resolve_input.query)
        EMBEDDING_LATENCY.labels(tenant_id=tenant_label).observe(time.time() - start_emb)

        # Step 3: Search Semantic Vector Store
        # Wider search window (top_k=20) to mitigate recall masking under shared index
        search_k = 20 if resolve_input.tenant_id is not None else 5
        search_results = self.vector_store.search(
            query_vector, top_k=search_k, tenant_id=resolve_input.tenant_id
        )

        if not search_results:
            reason = "Miss: No semantically similar queries found in VectorStore."
            if self.debug_mode:
                print(f"[DEBUG L2] Vector search returned empty -> {reason}")

            self.metrics.record_miss()
            L2_MISSES.labels(tenant_id=tenant_label).inc()
            from rag_cache.core.observability import DECISION_ENGINE_REJECTIONS

            DECISION_ENGINE_REJECTIONS.labels(tenant_id=tenant_label, reason="similarity").inc()
            return CacheResult(hit=False), reason

        # Step 4: Fetch Candidate Payloads
        candidates_with_scores = []
        for result in search_results:
            score = result.get("score", 0.0)
            cache_key = result.get("id")
            if not cache_key:
                continue

            cached_data = self.kv_store.get(cache_key, tenant_id=resolve_input.tenant_id)
            if not cached_data and resolve_input.tenant_id is not None:
                # Fallback to global namespace mapping for global scope entries
                cached_data = self.kv_store.get(cache_key, tenant_id=None)

            if cached_data:
                try:
                    entry = CacheEntry(**cached_data)
                    candidates_with_scores.append((entry, score))
                except Exception:
                    pass

        # Step 5: Filter through Decision Engine thresholds
        start_de = time.time()
        best_candidate, reason, confidence = self.decision_engine.evaluate_candidates(
            current_intent=resolve_input.intent,
            current_doc_ids=resolve_input.doc_ids,
            candidates_with_scores=candidates_with_scores,
            current_doc_versions=resolve_input.doc_versions,
            current_tenant_id=resolve_input.tenant_id,
            current_user_id=resolve_input.user_id,
        )
        DECISION_ENGINE_LATENCY.labels(tenant_id=tenant_label).observe(time.time() - start_de)

        if best_candidate:
            self.metrics.record_hit()
            L2_HITS.labels(tenant_id=tenant_label).inc()
            LLM_CALLS_SAVED.labels(tenant_id=tenant_label).inc()
            return (
                CacheResult(
                    hit=True,
                    response=best_candidate.response,
                    metadata=best_candidate.metadata,
                    confidence=confidence,
                ),
                reason,
            )

        self.metrics.record_miss()
        L2_MISSES.labels(tenant_id=tenant_label).inc()

        # Classify the reject reason
        reject_reason = "similarity"  # default fallback
        reason_lower = reason.lower()
        if "similarity" in reason_lower:
            reject_reason = "similarity"
        elif "overlap" in reason_lower:
            reject_reason = "overlap"
        elif "intent" in reason_lower:
            reject_reason = "intent"
        elif "tenant" in reason_lower or "scope" in reason_lower:
            reject_reason = "tenant"
        elif "version" in reason_lower or "drift" in reason_lower:
            reject_reason = "version_drift"

        from rag_cache.core.observability import DECISION_ENGINE_REJECTIONS

        DECISION_ENGINE_REJECTIONS.labels(tenant_id=tenant_label, reason=reject_reason).inc()

        return CacheResult(hit=False, confidence=confidence), reason

    def store(self, store_input: StoreInput) -> str:
        if store_input.intent == "default":
            store_input.intent = self.intent_classifier.classify(store_input.query)

        if store_input.intent in self.bypass_intents:
            return "skipped_bypass_intent"

        import time

        from rag_cache.core.observability import EMBEDDING_LATENCY, get_tenant_label

        start_emb = time.time()
        query_vector = self.embedder.embed_query(store_input.query)
        EMBEDDING_LATENCY.labels(tenant_id=get_tenant_label(store_input.tenant_id)).observe(
            time.time() - start_emb
        )
        cache_key = self._generate_cache_key(store_input.query)

        entry = CacheEntry(
            query=store_input.query,
            response=store_input.response,
            doc_ids=store_input.doc_ids,
            intent=store_input.intent,
            tenant_id=store_input.tenant_id,
            user_id=store_input.user_id,
            scope=store_input.scope,
            doc_versions=store_input.doc_versions,
            metadata=store_input.metadata,
        )

        # Dual-Write
        self.kv_store.set(
            cache_key, asdict(entry), ttl_seconds=self.ttl_seconds, tenant_id=store_input.tenant_id
        )
        self.vector_store.upsert(
            ids=[cache_key],
            vectors=[query_vector],
            metadata=[
                {
                    "query": store_input.query,
                    "intent": store_input.intent,
                    "tenant_id": store_input.tenant_id,
                }
            ],
            tenant_id=store_input.tenant_id,
        )

        return cache_key
