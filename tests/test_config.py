import os
import tempfile

import pytest

from rag_cache import ContextCache, ContextCacheConfig


def test_default_config(monkeypatch):
    """Verify default config values."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("L1_TTL", raising=False)
    monkeypatch.delenv("L2_TTL", raising=False)
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)

    config = ContextCacheConfig.load()
    assert config.redis_url == "redis://localhost:6379/0"
    assert config.use_faiss is False
    assert config.l1_ttl_seconds == 86400
    assert config.l2_ttl_seconds == 1209600
    assert config.min_embedding_similarity == 0.85
    assert config.min_document_overlap == 0.85


def test_yaml_config_load(monkeypatch):
    """Verify loading from a YAML config file."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("L1_TTL", raising=False)
    monkeypatch.delenv("L2_TTL", raising=False)
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)

    yaml_content = """
redis_url: "redis://my-redis-host:6379/2"
use_faiss: true
l1_ttl_seconds: 3600
l2_ttl_seconds: 7200
min_embedding_similarity: 0.92
min_document_overlap: 0.75
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = ContextCacheConfig.load(config_path=temp_path)
        assert config.redis_url == "redis://my-redis-host:6379/2"
        assert config.use_faiss is True
        assert config.l1_ttl_seconds == 3600
        assert config.l2_ttl_seconds == 7200
        assert config.min_embedding_similarity == 0.92
        assert config.min_document_overlap == 0.75
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_env_overrides(monkeypatch):
    """Verify that environment variables override defaults and YAML configurations."""
    yaml_content = """
redis_url: "redis://my-redis-host:6379/2"
l1_ttl_seconds: 3600
min_embedding_similarity: 0.92
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Set environment variables
        monkeypatch.setenv("REDIS_URL", "redis://env-redis:6379/1")
        monkeypatch.setenv("L1_TTL", "999")
        monkeypatch.setenv("L2_TTL", "8888")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.68")

        config = ContextCacheConfig.load(config_path=temp_path)

        # Verify env overrides YAML/defaults
        assert config.redis_url == "redis://env-redis:6379/1"
        assert config.l1_ttl_seconds == 999
        assert config.l2_ttl_seconds == 8888
        assert config.min_embedding_similarity == 0.68
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_facade_kwarg_overrides(monkeypatch):
    """Verify constructor kwargs take highest precedence."""
    monkeypatch.setenv("REDIS_URL", "redis://env-redis:6379/1")

    # Instantiate facade with explicit kwargs overriding everything
    cache = ContextCache(redis_url="redis://kwarg-redis:6379/5", use_local_embeddings=False)

    assert cache.config.redis_url == "redis://kwarg-redis:6379/5"
    assert cache.config.use_local_embeddings is False


def test_facade_from_config(monkeypatch):
    """Verify instantiating the facade directly from a YAML config file."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("L1_TTL", raising=False)
    monkeypatch.delenv("L2_TTL", raising=False)
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)

    yaml_content = """
redis_url: "redis://yaml-facade:6379/3"
use_faiss: false
use_local_embeddings: false
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        cache = ContextCache.from_config(temp_path)
        assert cache.config.redis_url == "redis://yaml-facade:6379/3"
        assert cache.config.use_local_embeddings is False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
