import hashlib
import random
from typing import List

from rag_cache.interfaces.embedding import Embedder


class MockEmbedder(Embedder):
    """
    A lightweight, deterministic Mock Embedder.
    Generates normalized pseudo-random vectors based on the text hash.
    Because it is deterministic, identical strings produce identical vectors,
    making it completely functional for End-to-End testing without importing PyTorch.
    """

    def __init__(self, dimension: int = 1536):
        # Defaulting to 1536 to match OpenAI's text-embedding-ada-002 / text-embedding-3-small
        self.dimension = dimension

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        # Hash the string to create a consistent integer seed
        text_hash = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(text_hash)

        # Generate raw pseudo-random floats
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]

        # Normalize the vector so it behaves properly in similarity searches
        # (cosine similarity will match the dot product)
        magnitude = sum(x**2 for x in vector) ** 0.5
        if magnitude == 0:
            return vector

        return [x / magnitude for x in vector]

    def embed_query(self, text: str) -> List[float]:
        """Implements Embedder interface."""
        return self._generate_deterministic_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Implements Embedder interface."""
        return [self._generate_deterministic_vector(t) for t in texts]
