import json
from typing import Any, Dict, Optional

import redis

from rag_cache.interfaces.key_value_store import KeyValueStore


class RedisKeyValueStore(KeyValueStore):
    """
    A production-ready, thread-safe Redis Key-Value store integration.
    Implements KeyValueStore with connection pooling, custom naming strategies,
    flexible TTL options, robust JSON serialization, and tenant isolation preparation.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: Optional[int] = None,
        tenant_id: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
    ):
        self.default_ttl = default_ttl
        self.tenant_id = tenant_id

        # Initialize the connection pool
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            decode_responses=True,  # Automatically decodes Redis replies to unicode strings
        )
        self.client = redis.Redis(connection_pool=self.pool)

    def _resolve_redis_key(self, key: str, tenant_id: Optional[str] = None) -> str:
        """
        Translates high-level keys to namespaced Redis keys:
        - If key starts with "metrics:", keeps metrics:<key> structure
        - If key starts with "retrieval:", maps retrieval:<hash> -> l1:<hash>
        - If key starts with "map:", maps map:<key> -> map:<key>
        - Otherwise, maps -> l2:<hash>
        - Prepend tenant prefix if tenant_id is configured.
        """
        if key.startswith("metrics:"):
            key_type = "metrics"
            base_key = key[len("metrics:") :]
        elif key.startswith("retrieval:"):
            key_type = "l1"
            base_key = key[len("retrieval:") :]
        elif key.startswith("map:"):
            key_type = "map"
            base_key = key[len("map:") :]
        elif key.startswith("lock:"):
            key_type = "lock"
            base_key = key[len("lock:") :]
        else:
            key_type = "l2"
            base_key = key

        resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id

        # Construct isolation path
        if resolved_tenant:
            return f"tenant:{resolved_tenant}:{key_type}:{base_key}"

        if key_type == "metrics":
            return f"metrics:{base_key}"
        if key_type == "l1":
            return f"l1:{base_key}"
        if key_type == "map":
            return f"map:{base_key}"
        if key_type == "lock":
            return f"lock:{base_key}"
        return f"l2:{base_key}"

    def get(self, key: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch value from Redis, deserializing it back to dict. Fail-safe on connection errors."""
        redis_key = self._resolve_redis_key(key, tenant_id=tenant_id)
        import time

        from rag_cache.core.observability import REDIS_LATENCY, get_tenant_label

        start_time = time.time()
        try:
            val = self.client.get(redis_key)
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="get").observe(
                duration
            )
            if val is not None:
                return json.loads(val)
        except Exception:
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="get").observe(
                duration
            )
            # Fallback gracefully (behave as a cache miss) in production env
            pass
        return None

    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Save serialized JSON string to Redis. Expiry is set based on hierarchy: write TTL > default TTL."""
        redis_key = self._resolve_redis_key(key, tenant_id=tenant_id)
        expire = ttl_seconds if ttl_seconds is not None else self.default_ttl
        import time

        from rag_cache.core.observability import REDIS_LATENCY, get_tenant_label

        serialized = json.dumps(value)
        start_time = time.time()
        try:
            self.client.set(redis_key, serialized, ex=expire)
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="set").observe(
                duration
            )
        except Exception:
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="set").observe(
                duration
            )
            pass

    def delete(self, key: str, tenant_id: Optional[str] = None) -> bool:
        """Remove key from Redis. Returns True if deleted successfully."""
        redis_key = self._resolve_redis_key(key, tenant_id=tenant_id)
        import time

        from rag_cache.core.observability import REDIS_LATENCY, get_tenant_label

        start_time = time.time()
        try:
            res = bool(self.client.delete(redis_key))
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="delete").observe(
                duration
            )
            return res
        except Exception:
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="delete").observe(
                duration
            )
            return False

    def exists(self, key: str, tenant_id: Optional[str] = None) -> bool:
        """Check if key exists in Redis."""
        redis_key = self._resolve_redis_key(key, tenant_id=tenant_id)
        import time

        from rag_cache.core.observability import REDIS_LATENCY, get_tenant_label

        start_time = time.time()
        try:
            res = bool(self.client.exists(redis_key))
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="exists").observe(
                duration
            )
            return res
        except Exception:
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="exists").observe(
                duration
            )
            return False

    def incr(self, key: str, tenant_id: Optional[str] = None) -> int:
        """Atomic integer increment helper for metrics logging."""
        redis_key = self._resolve_redis_key(key, tenant_id=tenant_id)
        import time

        from rag_cache.core.observability import REDIS_LATENCY, get_tenant_label

        start_time = time.time()
        try:
            res = self.client.incr(redis_key)
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="incr").observe(
                duration
            )
            return res
        except Exception:
            duration = time.time() - start_time
            REDIS_LATENCY.labels(tenant_id=get_tenant_label(tenant_id), operation="incr").observe(
                duration
            )
            return 0

    def acquire_lock(
        self, key: str, value: str, expire_ms: int, tenant_id: Optional[str] = None
    ) -> bool:
        """
        Attempts to acquire a distributed lock in Redis.
        Uses non-blocking SET NX PX.
        """
        redis_key = self._resolve_redis_key(f"lock:{key}", tenant_id=tenant_id)
        try:
            return bool(self.client.set(redis_key, value, nx=True, px=expire_ms))
        except Exception:
            return False

    def release_lock(self, key: str, value: str, tenant_id: Optional[str] = None) -> bool:
        """
        Releases a distributed lock in Redis using a safe Lua script
        to ensure atomicity (only deletes if the lock token matches).
        """
        redis_key = self._resolve_redis_key(f"lock:{key}", tenant_id=tenant_id)
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        try:
            return bool(self.client.eval(lua_script, 1, redis_key, value))
        except Exception:
            return False
