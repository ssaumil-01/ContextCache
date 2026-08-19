from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class KeyValueStore(ABC):
    @abstractmethod
    def get(self, key: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve the cached LLM response and associated metadata for a specific payload key.
        Returns None on a cache miss.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Store the generated response and metadata for a given key, with an optional Time-To-Live."""
        pass

    @abstractmethod
    def delete(self, key: str, tenant_id: Optional[str] = None) -> bool:
        """Remove a specific key from the cache (used for manual invalidation). Returns True if deleted."""
        pass

    @abstractmethod
    def exists(self, key: str, tenant_id: Optional[str] = None) -> bool:
        """Check if a specific key exists in the cache. Returns True if it exists."""
        pass

    def acquire_lock(
        self, key: str, value: str, expire_ms: int, tenant_id: Optional[str] = None
    ) -> bool:
        """
        Acquires a lease lock. Returns True if acquired, False otherwise.
        Defaults to True to allow execution to proceed if locking is unsupported.
        """
        return True

    def release_lock(self, key: str, value: str, tenant_id: Optional[str] = None) -> bool:
        """Releases the lease lock. Defaults to True."""
        return True
