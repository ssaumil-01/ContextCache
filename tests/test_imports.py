from rag_cache import (
    CacheResult,
    GenerationCache,
    ContextCache,
    ResolveInput,
    RetrievalCache,
    StoreInput,
)


def test_package_exports():
    """Ensure that all core components are importable from the package root."""
    assert ContextCache is not None
    assert RetrievalCache is not None
    assert GenerationCache is not None
    assert ResolveInput is not None
    assert StoreInput is not None
    assert CacheResult is not None


def test_facade_instantiation():
    """Ensure the facade can be imported and initialized without crashing when using mocks."""
    cache = ContextCache(use_local_embeddings=False)
    assert cache is not None
    assert cache.retrieval_layer is not None
    assert cache.generation_layer is not None
