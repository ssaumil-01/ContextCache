import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

from rag_cache.core.cache import GenerationCache, RetrievalCache
from rag_cache.core.config import ContextCacheConfig
from rag_cache.core.decision_engine import DecisionEngine
from rag_cache.core.default_intent import RuleBasedIntentClassifier
from rag_cache.core.models import ResolveInput, StoreInput
from rag_cache.interfaces.embedding import Embedder
from rag_cache.interfaces.vector_store import VectorStore


class UnifiedContextCache:
    """
    High-level facade for context-aware multi-level RAG caching.

    Provides a simple, plug-and-play API:

        answer = cache.run(query, retriever, llm)

    Handles:
    - L1 Retrieval Cache (query → doc_ids)
    - L2 Generation Cache (query + context → response)
    - Fallback to retriever + LLM
    - Automatic caching
    """

    def __init__(self, config: Optional[ContextCacheConfig] = None, **kwargs):
        # 1. Load config if not explicitly provided
        if config is None:
            config = ContextCacheConfig.load()
        else:
            config = config.model_copy()

        # 2. Override with kwargs for backward compatibility
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.debug = config.debug

        # Local imports to keep dependencies optional
        from rag_cache.integrations.embeddings.mock import MockEmbedder
        from rag_cache.integrations.key_value_stores.redis import RedisKeyValueStore

        kv = RedisKeyValueStore(redis_url=config.redis_url, tenant_id=config.tenant_id)
        self.kv = kv

        if config.prometheus_port is not None:
            from rag_cache.core.observability import start_metrics_server

            start_metrics_server(port=config.prometheus_port)

        vector_db: VectorStore
        if config.use_faiss:
            from rag_cache.integrations.vector_stores.faiss import FaissVectorStore

            vector_db = FaissVectorStore(
                dimension=config.faiss_dimension,
                redis_url=config.redis_url,
                tenant_id=config.tenant_id,
                index_filepath=config.faiss_index_filepath,
                kv_store=kv,
            )
        else:
            from rag_cache.integrations.vector_stores.in_memory import InMemoryVectorStore

            vector_db = InMemoryVectorStore()

        # -----------------------------
        # L1: Retrieval Cache
        # -----------------------------
        self.retrieval_layer = RetrievalCache(kv_store=kv, ttl_seconds=config.l1_ttl_seconds)

        # -----------------------------
        # L2: Embedding Setup
        # -----------------------------
        embedder: Embedder = MockEmbedder()
        if config.use_local_embeddings:
            try:
                from rag_cache.integrations.embeddings.sentence_transformer import (
                    SentenceTransformerEmbedder,
                )

                embedder = SentenceTransformerEmbedder()
            except ImportError:
                warnings.warn(
                    "sentence-transformers not installed. "
                    "Semantic caching is disabled and MockEmbedder will be used. "
                    "Install with: pip install 'rag-cachex[all]'"
                )

        # -----------------------------
        # L2: Generation Cache
        # -----------------------------
        self.generation_layer = GenerationCache(
            embedder=embedder,
            vector_store=vector_db,
            kv_store=kv,
            intent_classifier=RuleBasedIntentClassifier(),
            decision_engine=DecisionEngine(config=config),
            bypass_intents=config.bypass_intents,
            debug_mode=config.debug,
            ttl_seconds=config.l2_ttl_seconds,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "UnifiedContextCache":
        """
        Instantiates ContextCache directly using a YAML config file.
        """
        config = ContextCacheConfig.load(config_path)
        return cls(config=config)

    def get_docs(self, query: str, tenant_id: Optional[str] = None) -> Optional[List[str]]:
        """
        Step-by-step L1 retrieval cache lookup.
        Returns the cached list of doc_ids for this exact query string if found, otherwise None.
        """
        active_tenant = tenant_id if tenant_id is not None else self.kv.tenant_id
        return self.retrieval_layer.resolve(query, tenant_id=active_tenant)

    def get_answer(
        self,
        query: str,
        doc_ids: List[str],
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        scope: str = "tenant",
    ) -> Optional[str]:
        """
        Step-by-step L2 semantic generation cache lookup.
        Returns the cached response string if hit and validated, otherwise None.
        """
        active_tenant = tenant_id if tenant_id is not None else self.kv.tenant_id
        result, _ = self.generation_layer.resolve(
            ResolveInput(
                query=query, doc_ids=doc_ids, tenant_id=active_tenant, user_id=user_id, scope=scope
            )
        )
        return result.response if result.hit else None

    def save(
        self,
        query: str,
        doc_ids: List[str],
        response: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        scope: str = "tenant",
    ) -> None:
        """
        Step-by-step dual-write cache store.
        Stores query -> doc_ids in L1, and query + doc_ids + response in L2.
        """
        active_tenant = tenant_id if tenant_id is not None else self.kv.tenant_id
        self.retrieval_layer.store(query, doc_ids, tenant_id=active_tenant)
        self.generation_layer.store(
            StoreInput(
                query=query,
                doc_ids=doc_ids,
                response=response,
                tenant_id=active_tenant,
                user_id=user_id,
                scope=scope,
            )
        )

    def _execute_with_stampede_protection(
        self,
        lock_name: str,
        execute_fn: Callable[[], Any],
        check_cache_fn: Callable[[], Any],
        tenant_id: Optional[str] = None,
    ) -> Tuple[Any, bool]:
        """
        Executes a downstream function with distributed lock coalescing.
        If stampede protection is disabled or locking fails to initiate,
        falls back to direct execution.
        Returns (result, hit_during_wait).
        """
        import time
        import uuid

        if not self.config.stampede_protection:
            return execute_fn(), False

        # Generate a unique lock token for this request
        lock_token = str(uuid.uuid4())

        # Try to acquire lock
        acquired = self.kv.acquire_lock(
            key=lock_name,
            value=lock_token,
            expire_ms=self.config.lock_timeout_ms,
            tenant_id=tenant_id,
        )

        if acquired:
            try:
                # We are the leader: execute the computation and return
                return execute_fn(), False
            finally:
                # Release lock safely
                self.kv.release_lock(key=lock_name, value=lock_token, tenant_id=tenant_id)
        else:
            # We are the follower: poll the cache until leader completes
            start_time = time.time()
            timeout = self.config.lock_timeout_ms / 1000.0
            poll_interval = self.config.poll_interval_ms / 1000.0

            while time.time() - start_time < timeout:
                time.sleep(poll_interval)
                # Check if cache has been populated
                cached_val = check_cache_fn()
                if cached_val is not None:
                    return cached_val, True

            # Safety fallback: leader timed out or crashed, execute directly
            return execute_fn(), False

    # ---------------------------------------------------------
    # 🔥 MAIN ENTRYPOINT (THIS IS WHAT USERS USE)
    # ---------------------------------------------------------
    def run(
        self,
        query: str,
        retriever: Callable[[str], List[str]],
        llm: Callable[[str, List[str]], str],
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        scope: str = "tenant",
    ) -> Dict[str, Any]:
        """
        Executes full RAG pipeline with caching.

        Args:
            query: user query
            retriever: function(query) → List[doc_ids]
            llm: function(query, doc_ids) → response
            tenant_id: optional tenant context override
            user_id: optional user context override
            scope: target lookup cache scope

        Returns:
            dict with:
                - answer
                - cache_hit (bool)
                - source ("L1", "L2", "LLM")
        """
        import time

        from rag_cache.core.observability import TOTAL_REQUEST_LATENCY, get_tenant_label

        start_req = time.time()

        # Resolve the active tenant context
        active_tenant = tenant_id if tenant_id is not None else self.kv.tenant_id

        if self.debug:
            print(f"\n[Query]: {query} (tenant: {active_tenant}, scope: {scope})")

        # -----------------------------
        # Step 1: L1 Retrieval Cache
        # -----------------------------
        doc_ids = self.retrieval_layer.resolve(query, tenant_id=active_tenant)

        if doc_ids:
            if self.debug:
                print("⚡ L1 HIT: Retrieved cached documents")
        else:
            if self.debug:
                print("❌ L1 MISS: Calling retriever")

            import hashlib

            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()

            def run_retriever():
                docs = retriever(query)
                self.retrieval_layer.store(query, docs, tenant_id=active_tenant)
                return docs

            doc_ids, hit_during_wait = self._execute_with_stampede_protection(
                lock_name=f"retrieval:{query_hash}",
                execute_fn=run_retriever,
                check_cache_fn=lambda: self.retrieval_layer.resolve(query, tenant_id=active_tenant),
                tenant_id=active_tenant,
            )

        if doc_ids is None:
            doc_ids = []

        # -----------------------------
        # Step 2: L2 Generation Cache
        # -----------------------------
        result, reason = self.generation_layer.resolve(
            ResolveInput(
                query=query, doc_ids=doc_ids, tenant_id=active_tenant, user_id=user_id, scope=scope
            )
        )

        if result.hit:
            if self.debug:
                print(f"⚡ L2 HIT: {reason}")

            # Increment hit counter in Redis
            self.kv.incr("metrics:hits", tenant_id=active_tenant)

            TOTAL_REQUEST_LATENCY.labels(
                tenant_id=get_tenant_label(active_tenant), cache_hit="true"
            ).observe(time.time() - start_req)
            self._update_gauges()
            return {"answer": result.response, "cache_hit": True, "source": "L2"}

        if self.debug:
            print(f"❌ L2 MISS: {reason}")
            print("... Falling back to LLM ...")

        import hashlib

        doc_str = "".join(doc_ids)
        l2_lock_name = hashlib.sha256((query + ":" + doc_str).encode("utf-8")).hexdigest()

        def run_llm():
            # Increment miss counter in Redis for leader
            self.kv.incr("metrics:misses", tenant_id=active_tenant)
            ans = llm(query, doc_ids)
            self.generation_layer.store(
                StoreInput(
                    query=query,
                    doc_ids=doc_ids,
                    response=ans,
                    tenant_id=active_tenant,
                    user_id=user_id,
                    scope=scope,
                )
            )
            return ans

        def check_l2_cache():
            res, _ = self.generation_layer.resolve(
                ResolveInput(
                    query=query,
                    doc_ids=doc_ids,
                    tenant_id=active_tenant,
                    user_id=user_id,
                    scope=scope,
                )
            )
            return res.response if res.hit else None

        response, hit_during_wait = self._execute_with_stampede_protection(
            lock_name=f"l2:{l2_lock_name}",
            execute_fn=run_llm,
            check_cache_fn=check_l2_cache,
            tenant_id=active_tenant,
        )

        if hit_during_wait:
            self.kv.incr("metrics:hits", tenant_id=active_tenant)
            TOTAL_REQUEST_LATENCY.labels(
                tenant_id=get_tenant_label(active_tenant), cache_hit="true"
            ).observe(time.time() - start_req)
            self._update_gauges()
            return {"answer": response, "cache_hit": True, "source": "L2"}
        else:
            TOTAL_REQUEST_LATENCY.labels(
                tenant_id=get_tenant_label(active_tenant), cache_hit="false"
            ).observe(time.time() - start_req)
            self._update_gauges()
            return {"answer": response, "cache_hit": False, "source": "LLM"}

    def _update_gauges(self) -> None:
        try:
            from rag_cache.core.observability import (
                CACHE_ENTRIES,
                FAISS_VECTORS_TOTAL,
                REDIS_MEMORY_BYTES,
            )
            from rag_cache.integrations.key_value_stores.redis import RedisKeyValueStore

            if isinstance(self.kv, RedisKeyValueStore):
                CACHE_ENTRIES.set(self.kv.client.dbsize())
                info = self.kv.client.info("memory")
                REDIS_MEMORY_BYTES.set(info.get("used_memory", 0))

            # Get ntotal from vector store
            if hasattr(self.generation_layer.vector_store, "index"):
                FAISS_VECTORS_TOTAL.set(self.generation_layer.vector_store.index.ntotal)
        except Exception:
            pass

    # ---------------------------------------------------------
    # Optional: Metrics passthrough
    # ---------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Returns cache performance metrics."""
        return self.generation_layer.get_stats()
