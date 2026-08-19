import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from rag_cache.integrations.key_value_stores.redis import RedisKeyValueStore
from rag_cache.interfaces.vector_store import VectorStore


class FaissVectorStore(VectorStore):
    """
    A production-grade, highly-scalable FAISS-based Vector Store integration.
    Wraps faiss.IndexIDMap(faiss.IndexFlatIP(dimension)) to support exact cosine similarity
    searches and manual vector deletion by ID.
    Metadata mapping and persistent 64-bit integer ID allocation is delegated to Redis.
    """

    def __init__(
        self,
        dimension: int = 384,
        redis_url: str = "redis://localhost:6379/0",
        tenant_id: Optional[str] = None,
        index_filepath: Optional[str] = "faiss_index.bin",
        kv_store: Optional[RedisKeyValueStore] = None,
    ):
        self.dimension = dimension
        self.tenant_id = tenant_id
        self.index_filepath = index_filepath

        # Initialize Redis client for ID mapping
        if kv_store is not None:
            self.kv = kv_store
        else:
            self.kv = RedisKeyValueStore(redis_url=redis_url, tenant_id=tenant_id)

        # Set OpenMP threads to 1 to prevent runtime crashes (e.g. on macOS with PyTorch conflicts)
        faiss.omp_set_num_threads(1)

        # Initialize FAISS IndexIDMap with Flat Inner Product
        self.flat_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(self.flat_index)

        # Try to load existing index from disk if present
        if self.index_filepath and os.path.exists(self.index_filepath):
            try:
                self.index = faiss.read_index(self.index_filepath)
            except Exception:
                pass

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalizes a single numpy array vector to unit length."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """
        Inserts new embeddings. Maps string cache keys to 64-bit integers dynamically using Redis.
        If a key already exists, its old vector is deleted from the index before writing the new one (UPSERT).
        """
        vector_list = []
        id_list = []

        for i, cache_key in enumerate(ids):
            vec = vectors[i]
            norm_vec = self._normalize(np.array(vec, dtype=np.float32))

            # Retrieve tenant_id dynamically from metadata or override
            item_tenant = tenant_id
            if not item_tenant and metadata and i < len(metadata):
                item_tenant = metadata[i].get("tenant_id")

            # Retrieve or generate 64-bit integer ID mapping from Redis
            mapped_data = self.kv.get(f"map:cache_key:{cache_key}", tenant_id=item_tenant)
            active_tenant = item_tenant
            if not mapped_data and item_tenant is not None:
                fallback_data = self.kv.get(f"map:cache_key:{cache_key}", tenant_id=None)
                if fallback_data:
                    mapped_data = fallback_data
                    active_tenant = None

            if mapped_data and "int_id" in mapped_data:
                int_id = int(mapped_data["int_id"])
                # Remove the old vector with this ID to implement update logic
                try:
                    self.index.remove_ids(np.array([int_id], dtype=np.int64))
                except Exception:
                    pass
            else:
                # Allocate a new atomic counter-based 64-bit ID from Redis using global namespace to avoid collisions
                int_id = self.kv.incr("metrics:next_vector_id", tenant_id=None)
                # Save mapping directories under the active tenant namespace
                self.kv.set(
                    f"map:cache_key:{cache_key}", {"int_id": int_id}, tenant_id=active_tenant
                )
                query_str = metadata[i].get("query", "") if (metadata and i < len(metadata)) else ""
                self.kv.set(
                    f"map:vector_id:{int_id}",
                    {"cache_key": cache_key, "query": query_str, "tenant_id": active_tenant},
                    tenant_id=active_tenant,
                )

            vector_list.append(norm_vec)
            id_list.append(int_id)

        if vector_list:
            vectors_np = np.vstack(vector_list).astype(np.float32)
            ids_np = np.array(id_list, dtype=np.int64)
            self.index.add_with_ids(vectors_np, ids_np)
            self.save_index()

    def search(
        self, query_vector: List[float], top_k: int = 5, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches FAISS for similar vectors. Translates returned integer IDs back
        to string cache keys using Redis, and returns matches with scores.
        """
        import time

        from rag_cache.core.observability import FAISS_SEARCH_LATENCY, get_tenant_label

        start_time = time.time()
        try:
            if self.index.ntotal == 0:
                return []

            norm_query = self._normalize(np.array(query_vector, dtype=np.float32)).reshape(1, -1)
            scores, indices = self.index.search(norm_query, top_k)

            results = []
            if indices.size > 0:
                for i, int_id in enumerate(indices[0]):
                    if int_id == -1:
                        continue  # FAISS padding index

                    # Map 64-bit integer back to original string cache key
                    mapped_data = self.kv.get(f"map:vector_id:{int_id}", tenant_id=tenant_id)
                    if not mapped_data and tenant_id is not None:
                        # Fallback to global namespace mapping for global scope entries
                        mapped_data = self.kv.get(f"map:vector_id:{int_id}", tenant_id=None)

                    if mapped_data and "cache_key" in mapped_data:
                        cache_key = mapped_data["cache_key"]
                        query_str = mapped_data.get("query", "")
                        results.append(
                            {
                                "id": cache_key,
                                "score": float(scores[0][i]),
                                "metadata": {"query": query_str},
                            }
                        )
            return results
        finally:
            duration = time.time() - start_time
            FAISS_SEARCH_LATENCY.labels(tenant_id=get_tenant_label(tenant_id)).observe(duration)

    def delete(self, doc_id: str, tenant_id: Optional[str] = None) -> bool:
        """Removes a vector from the index using its string cache key."""
        cache_key = doc_id
        mapped_data = self.kv.get(f"map:cache_key:{cache_key}", tenant_id=tenant_id)
        active_tenant = tenant_id
        if not mapped_data and tenant_id is not None:
            fallback_data = self.kv.get(f"map:cache_key:{cache_key}", tenant_id=None)
            if fallback_data:
                mapped_data = fallback_data
                active_tenant = None

        if mapped_data and "int_id" in mapped_data:
            int_id = int(mapped_data["int_id"])
            try:
                self.index.remove_ids(np.array([int_id], dtype=np.int64))
            except Exception:
                return False

            # Delete Redis mappings
            self.kv.delete(f"map:cache_key:{cache_key}", tenant_id=active_tenant)
            self.kv.delete(f"map:vector_id:{int_id}", tenant_id=active_tenant)
            self.save_index()
            return True
        return False

    def save_index(self) -> None:
        """Saves current index state to disk."""
        if self.index_filepath:
            try:
                faiss.write_index(self.index, self.index_filepath)
            except Exception:
                pass
