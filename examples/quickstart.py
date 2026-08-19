import os
import sys

# Adds the project root so this runs out-of-the-box locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------------------------------------------
# ContextCache Quickstart
# -------------------------------------------------------------
from rag_cache import ContextCache

# 1. Initialize the dual-layer cache (requires zero configuration!)
print("Loading semantic model...")
cache = ContextCache()


# Fake Network Systems
def fetch_from_vector_db(query: str):
    return ["chunk_alpha", "chunk_beta"]


def generate_from_llm(query: str, doc_ids: list):
    return "Distributed context-aware systems use embedded thresholds to validate hits."


def process_query(query: str):
    """A standard RAG pipeline wrapped with Context-Aware caching."""
    print(f"\n=> Query: '{query}'")

    # [Step A]: O(1) L1 Retrieval Lookup
    doc_ids = cache.get_docs(query)
    if doc_ids:
        print("  ⚡ L1 Hit! (Bypassed Vector Database)")
    else:
        doc_ids = fetch_from_vector_db(query)
        print("  🔍 VectorDB Searched.")

    # [Step B]: Complex L2 Semantic Context Lookup
    answer = cache.get_answer(query, doc_ids)
    if answer:
        print("  ✅ L2 Hit! (Bypassed LLM Generation)")
        return answer

    print("  🧠 Generating via LLM...")
    answer = generate_from_llm(query, doc_ids)

    # [Step C]: Dual-Write Saving
    cache.save(query, doc_ids, answer)

    return answer


if __name__ == "__main__":
    # Run 1: Cold Cache (Miss L1 -> Miss L2 -> Execute Pipeline)
    process_query("What is context-aware caching?")

    # Run 2: Semantically identical string (Miss L1 -> Hit L2 Semantic Match)
    process_query("Explain context-aware caching.")

    # Run 3: Exact string replay (Hit L1 Exact string)
    # (Notice how it doesn't even search the VectorDB!)
    process_query("Explain context-aware caching.")
