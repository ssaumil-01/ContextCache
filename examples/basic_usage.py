import os
import sys

# Add the project root to the python path so it runs out-of-the-box
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_cache import GenerationCache as ContextCache
from rag_cache import ResolveInput, StoreInput
from rag_cache.core.decision_engine import DecisionEngine, DecisionRuleConfig
from rag_cache.core.default_intent import RuleBasedIntentClassifier

# Import our available integrations
from rag_cache.integrations.embeddings.mock import MockEmbedder
from rag_cache.integrations.key_value_stores.redis import RedisKeyValueStore
from rag_cache.integrations.vector_stores.in_memory import InMemoryVectorStore


def main():
    print("--- 1. Initializing ContextCache ---")
    # Initialize all concrete dependencies.
    # Notice how easy it is to swap MockEmbedder for OpenAIEmbedder later!
    cache = ContextCache(
        embedder=MockEmbedder(),
        vector_store=InMemoryVectorStore(),
        kv_store=RedisKeyValueStore(),
        intent_classifier=RuleBasedIntentClassifier(),
        decision_engine=DecisionEngine(config=DecisionRuleConfig(min_embedding_similarity=0.65)),
    )

    query = "What is our company's Q3 revenue?"
    retrieved_docs = ["doc_ID_alpha", "doc_ID_beta"]

    print("\n--- 2. First Request (Cache Miss) ---")
    result, reason = cache.resolve(ResolveInput(query=query, doc_ids=retrieved_docs))
    print(f"Hit: {result.hit} | Reason: {reason}")

    # ... Your Application would normally call the LLM here ...
    llm_answer = "The Q3 revenue reached a record $5.2 million."
    print("... Calling Expensive LLM API ...")

    print("\n--- 3. Storing to Cache ---")
    cache.store(StoreInput(query=query, response=llm_answer, doc_ids=retrieved_docs))
    print("Successfully cached the response.")

    print("\n--- 4. Second Request (Cache Hit!) ---")
    # Another user asks the same question and gets the same docs assigned to them
    result2, reason2 = cache.resolve(ResolveInput(query=query, doc_ids=retrieved_docs))
    print(f"Hit: {result2.hit} | Reason: {reason2}")
    print(f"Cached Answer: {result2.response}")

    print("\n--- 5. Third Request (Context Drift - Cache Miss) ---")
    # The user asks the same question, but the VectorDB pulled a NEW document today
    result3, reason3 = cache.resolve(
        ResolveInput(query=query, doc_ids=["doc_ID_alpha", "doc_ID_MewNewData!"])
    )
    print(f"Hit: {result3.hit}")
    print(f"Reason: {reason3}")


if __name__ == "__main__":
    main()
