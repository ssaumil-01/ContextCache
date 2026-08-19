from typing import Any, Dict, List, Optional

from rag_cache.interfaces.vector_store import VectorStore


class InMemoryVectorStore(VectorStore):
    """
    A simple, zero-dependency, in-memory Vector Store.
    Performs exhaustive exact nearest neighbor (KNN) search.
    Extremely useful for testing and very small caching layers.
    """

    def __init__(self):
        # Underlying storage
        # Structure: [{"id": str, "vector": List[float], "metadata": dict}]
        self.storage: List[Dict[str, Any]] = []

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Computes true cosine similarity between two vectors without relying on numpy."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Inserts new embeddings. If an ID already exists, it is overwritten (UPSERT).
        """
        for i, doc_id in enumerate(ids):
            # Ensure no duplicates by sweeping out the old entry with this ID (if it exists)
            self.storage = [item for item in self.storage if item["id"] != doc_id]

            # Add the new entry
            self.storage.append(
                {
                    "id": doc_id,
                    "vector": vectors[i],
                    "metadata": metadata[i] if metadata and i < len(metadata) else {},
                }
            )

    def search(
        self, query_vector: List[float], top_k: int = 5, *args, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity of the query against every vector in storage (O(N)),
        then returns the top-k highest scoring matches.
        """
        scored_results = []
        for item in self.storage:
            sim_score = self._cosine_similarity(query_vector, item["vector"])

            # Only return the data needed by the Orchestrator (it doesn't need to see the dense float array again)
            scored_results.append(
                {"id": item["id"], "score": sim_score, "metadata": item["metadata"]}
            )

        # Sort scores in descending order (highest similarity first)
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        return scored_results[:top_k]

    def delete(self, doc_id: str, *args, **kwargs) -> bool:
        """Removes a specific embedding by its ID."""
        original_length = len(self.storage)
        self.storage = [item for item in self.storage if item["id"] != doc_id]
        return len(self.storage) < original_length
