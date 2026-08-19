from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Convert a single user query string into a dense vector representation."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of document chunks into a list of vector representations."""
        pass
