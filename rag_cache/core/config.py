import os
from typing import Dict, List, Optional, Set

import yaml
from pydantic import BaseModel, Field


class ContextCacheConfig(BaseModel):
    """
    Unified configuration schema for ContextCache.
    Consolidates Redis, FAISS, Decision Engine thresholds, TTLs,
    and telemetry/debug parameters under a single validated object.
    """

    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379/0")

    # FAISS configuration
    use_faiss: bool = Field(default=False)
    faiss_index_filepath: Optional[str] = Field(default="faiss_index.bin")
    faiss_dimension: int = Field(default=384)

    # Decision Engine thresholds
    min_embedding_similarity: float = Field(default=0.85)
    min_document_overlap: float = Field(default=0.85)
    intent_match_mode: str = Field(default="compatible")
    intent_compatibility_matrix: Dict[str, Set[str]] = Field(default_factory=dict)
    bypass_intents: List[str] = Field(default_factory=lambda: ["action"])

    # Time-To-Live settings
    l1_ttl_seconds: int = Field(default=86400)
    l2_ttl_seconds: int = Field(default=1209600)

    # Tenant settings
    tenant_id: Optional[str] = Field(default=None)

    # Stampede Protection settings
    stampede_protection: bool = Field(default=True)
    lock_timeout_ms: int = Field(default=10000)
    poll_interval_ms: int = Field(default=100)

    # Observability & setup
    prometheus_port: Optional[int] = Field(default=None)
    use_local_embeddings: bool = Field(default=True)
    debug: bool = Field(default=False)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ContextCacheConfig":
        """
        Loads configuration from a YAML file and overrides it with environment variables.
        Precedence: Env Variables > YAML Config > Defaults.
        """
        config_data = {}

        # 1. Read YAML configuration if file path exists
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                if isinstance(yaml_data, dict):
                    config_data.update(yaml_data)

        # 2. Apply env variable overrides
        if os.getenv("REDIS_URL"):
            config_data["redis_url"] = os.getenv("REDIS_URL")

        l1_ttl = os.getenv("L1_TTL")
        if l1_ttl is not None:
            try:
                config_data["l1_ttl_seconds"] = int(l1_ttl)
            except ValueError:
                pass

        l2_ttl = os.getenv("L2_TTL")
        if l2_ttl is not None:
            try:
                config_data["l2_ttl_seconds"] = int(l2_ttl)
            except ValueError:
                pass

        similarity_threshold = os.getenv("SIMILARITY_THRESHOLD")
        if similarity_threshold is not None:
            try:
                config_data["min_embedding_similarity"] = float(similarity_threshold)
            except ValueError:
                pass

        return cls(**config_data)
