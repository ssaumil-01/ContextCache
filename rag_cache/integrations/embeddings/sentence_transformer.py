from typing import List

from rag_cache.interfaces.embedding import Embedder


class SentenceTransformerEmbedder(Embedder):
    """
    A production-ready offline embedding integration using sentence-transformers.
    Defaults to 'all-MiniLM-L6-v2'.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' library is not installed. "
                "Please install it via `pip install sentence-transformers`."
            )

        # The model will be downloaded automatically the first time this is instantiated
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        """Convert a single query into a dense normalized vector."""
        # normalize_embeddings=True guarantees cosine similarity matches dot-product
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of chunks into dense normalized vectors."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
