import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResolveInput:
    """Payload required to query the cache for an existing answer."""

    query: str
    doc_ids: List[str]
    intent: str = "default"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    scope: str = "tenant"
    doc_versions: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.tenant_id is None:
            self.scope = "global"


@dataclass
class StoreInput:
    """Payload required to save a newly generated LLM response into the cache."""

    query: str
    response: str
    doc_ids: List[str]
    intent: str = "default"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    scope: str = "tenant"
    doc_versions: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.tenant_id is None:
            self.scope = "global"


@dataclass
class CacheEntry:
    """The normalized object serialized and stored in the KeyValueStore."""

    query: str
    response: str
    doc_ids: List[str]
    intent: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    scope: str = "tenant"
    doc_versions: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.tenant_id is None:
            self.scope = "global"


@dataclass
class CacheResult:
    """The result returned to the user upon attempting to resolve cache."""

    hit: bool
    response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
