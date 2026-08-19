import os
import sys

# Add the project root to the python path so it runs out-of-the-box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_cache.integrations.embeddings.sentence_transformer import SentenceTransformerEmbedder


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Since SentenceTransformer returns normalized embeddings, dot product is exactly equal to Cosine Similarity."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def main():
    print("Loading SentenceTransformer Embedder (This may download weights on first run)...")
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    q1 = "What is the meaning of life?"
    q2 = "Explain meaning of life"
    q3 = "What is grading policy?"

    print("Generating embeddings...")
    v1 = embedder.embed_query(q1)
    v2 = embedder.embed_query(q2)
    v3 = embedder.embed_query(q3)

    print("\n--- Semantic Similarity Results ---\n")

    sim_1_2 = cosine_similarity(v1, v2)
    print(f"[{sim_1_2:.4f}] Similarity between:")
    print(f"  A) '{q1}'")
    print(f"  B) '{q2}'")
    print("  Outcome: STRONG MATCH (Should trigger Cache Check)\n")

    sim_1_3 = cosine_similarity(v1, v3)
    print(f"[{sim_1_3:.4f}] Similarity between:")
    print(f"  A) '{q1}'")
    print(f"  B) '{q3}'")
    print("  Outcome: NO MATCH (Radically different topic)\n")

    sim_2_3 = cosine_similarity(v2, v3)
    print(f"[{sim_2_3:.4f}] Similarity between:")
    print(f"  A) '{q2}'")
    print(f"  B) '{q3}'")
    print("  Outcome: NO MATCH \n")


if __name__ == "__main__":
    main()
