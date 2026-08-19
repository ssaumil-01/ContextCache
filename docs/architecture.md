# ContextCache Architectural Documentation

This document describes the internals, caching strategies, evaluation heuristics, and storage structures of ContextCache.

---

## 📐 Caching Topologies

ContextCache operates a two-tier caching strategy to optimize performance at both the vector retrieval and generation stages of RAG applications.

```
                    ┌────────────────────────────────────────┐
                    │               User Query               │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
      ┌─────────────────────────────────┴────────────────────────────────┐
      │                      L1: Retrieval Cache                         │
      │   - Exact String Match (Redis Key-Value)                        │
      │   - Bypasses expensive Vector Database lookups                   │
      └─────────────────────────────────┬────────────────────────────────┘
                                        │
                                 Miss   │   Hit (Yields Doc IDs)
                                        ▼   ───────────────────────────────┐
      ┌─────────────────────────────────┴────────────────────────────────┐ │
      │                      Vector DB Retrieval                         │ │
      │   - Fetches context documents (doc_ids) from Vector database      │ │
      └─────────────────────────────────┬────────────────────────────────┘ │
                                        │                                  │
                                        │◄─────────────────────────────────┘
                                        │ doc_ids
                                        ▼
      ┌─────────────────────────────────┴────────────────────────────────┐
      │                      L2: Generation Cache                        │
      │   - Semantic similarity query embedding matches                  │
      │   - Decision Engine context validation and intent checking       │
      │   - Bypasses heavy LLM generation APIs                           │
      └─────────────────────────────────┬────────────────────────────────┘
                                        │
                                 Miss   │   Hit (Returns Response)
                                        ▼   ───────────────────────────────┐
      ┌─────────────────────────────────┴────────────────────────────────┐ │
      │                  Expensive Downstream LLM API                    │ │
      │   - Invokes OpenAI/Anthropic/Local LLM generation                │ │
      └──────────────────────────────────────────────────────────────────┘ │
                                                                           │
                                                                           ▼
                                                                     User Response
```

---

## 1. L1 Cache (Retrieval Cache)

### Purpose
Bypasses the Document Vector Database query step for identical queries, saving execution latency and database workload.

### Strategy
- **Exact String Match**: Fast, O(1) hash table lookup.
- **Key Derivation**: Computes a SHA-256 hash of the query text to prevent naming conflicts or size issues in Redis.
- **Redis Namespace Key**: `l1:<hash>` (or `tenant:<tenant_id>:l1:<hash>` under tenant scope).
- **Payload**: JSON dictionary containing the retrieved document IDs:
  ```json
  {
    "doc_ids": ["doc_001", "doc_002"]
  }
  ```

---

## 2. L2 Cache (Generation Cache)

### Purpose
Reuses LLM generated answers for semantically identical or highly similar queries, checking that context documents and query intents match perfectly.

### Storage Split (Dual-Write)
1. **Vector Database (FAISS or InMemory)**:
   - Stores the query embeddings.
   - Maps the vector index ID directly to the unique cache key (`sha256(query)`).
2. **Key-Value Store (Redis)**:
   - Stores the actual `CacheEntry` JSON payload under the cache key `l2:<hash>` (or `tenant:<tenant_id>:l2:<hash>`).

### Cache Entry Schema
Every stored cache entry includes the following validated fields:
- `query`: The original user query.
- `response`: The LLM generated answer.
- `doc_ids`: Document IDs retrieved at generation time.
- `intent`: Classified query intent (e.g. informational).
- `tenant_id` / `user_id`: Namespace markers.
- `scope`: Access boundary (`global`, `tenant`, or `user`).
- `doc_versions`: Optional hashes of documents to verify version drift.
- `created_at`: Expiration timestamp.

---

## 3. Decision Engine

The core logic responsible for determining if a candidate cached response is safe to reuse. It protects the pipeline from serving stale or contextually mismatched answers.

### Confidence Score Formula
To rank semantic candidates, the Decision Engine calculates a weighted confidence score:
$$\text{Confidence} = (\text{Similarity} \times 0.4) + (\text{Overlap} \times 0.4) + (\text{Intent Match} \times 0.2)$$
where *Intent Match* is `1.0` for identical matches and `0.5` for compatible ones.

### Validation Checks
A candidate entry yields a **Cache Hit** if and only if it passes all the following criteria:

1. **Scope Check**: Asserts access rights. User-scoped cache entries cannot be fetched by other users, and tenant-scoped entries are isolated to their specific tenant.
2. **Intent Matching**: Checks if the query intent matches.
   - `strict`: Candidate and user intents must be identical.
   - `relaxed`: Any query intent matches.
   - `compatible`: Uses a lookup table to permit overlaps (e.g. `analytical` query can hit `informational` cache, but `action` queries bypass caching completely).
3. **Embedding Similarity**: Cosine similarity between user query embedding and cached embedding must exceed `min_embedding_similarity` (default: `0.85`).
4. **Document Overlap (Context Stability)**: Calculates the **Jaccard Similarity** of document IDs:
   $$\text{Jaccard Overlap} = \frac{|A \cap B|}{|A \cup B|}$$
   *Must exceed `min_document_overlap` (default: `0.85`).*
5. **Version Drift Check**: If document version hashes are provided, it verifies that no document has changed between the caching event and the current request.

---

## 4. Redis Key-Value Store Integration

Redis is the production-grade Key-Value database integration (`RedisKeyValueStore`).
- **Thread-safe Connection Pool**: Utilizes `redis.ConnectionPool` for connection recycling.
- **Fail-safe Operations**: All Redis operations are wrapped in try-except blocks; connection losses default to a cache-miss, preventing application crashes.
- **Auto-decoding**: Configured with `decode_responses=True` to fetch string formats directly.

---

## 5. FAISS Vector Store Integration

The `FaissVectorStore` provides high-performance local vector similarity indexing.
- **Structure**: Wraps `faiss.IndexFlatIP` (Inner Product/Cosine search) with `faiss.IndexIDMap` to link vectors directly to integer IDs.
- **Coordination**: FAISS natively supports only 64-bit integers as keys. The integration uses Redis atomic increments (`metrics:next_vector_id`) to map string cache keys (SHA-256 hashes) to integer IDs.
- **Disk Persistence**: Persists state to a file (`faiss_index.bin`) on inserts/deletions.

---

## 6. Tenant Isolation & Access Scopes

ContextCache protects data boundaries natively.

### Access Scopes
- **Global**: Entries are cached in the shared namespace. Available to all tenants and users.
- **Tenant**: Entries are isolated to the specific tenant ID.
- **User**: Entries are isolated to both the tenant ID and the specific user ID.

### Redis Naming Layout
Redis keys are automatically translated under the hood:
- **Global keys**:
  - L1: `l1:<hash>`
  - L2: `l2:<hash>`
  - Map: `map:<hash>`
- **Tenant-isolated keys**:
  - L1: `tenant:<tenant_id>:l1:<hash>`
  - L2: `tenant:<tenant_id>:l2:<hash>`
  - Map: `tenant:<tenant_id>:map:<hash>`
