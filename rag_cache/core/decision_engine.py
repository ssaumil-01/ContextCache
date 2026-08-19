# --- stdlib imports ---
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# --- internal imports ---
from rag_cache.core.models import CacheEntry
from rag_cache.utils.hashing import compute_document_overlap


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
@dataclass
class DecisionRuleConfig:
    """Configurable thresholds for the Decision Engine."""

    min_embedding_similarity: float = 0.85
    min_document_overlap: float = 0.85
    intent_match_mode: str = "compatible"  # Options: "strict", "relaxed", or "compatible"

    # Maps an incoming intent to a set of acceptable cached intents.
    # Empty sets natively fallback to strict match mode.
    intent_compatibility_matrix: Dict[str, Set[str]] = field(default_factory=dict)

    debug_mode: bool = False


# ---------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------
class DecisionEngine:
    """
    Evaluates cache candidates based on semantic similarity,
    context overlap, and intent.

    Ensures that incorrect or stale cache hits are rejected.
    """

    def __init__(self, config: Optional[Any] = None):
        if config is None:
            self.config = DecisionRuleConfig()
        elif hasattr(config, "min_embedding_similarity") and not hasattr(config, "debug_mode"):
            # It's the new ContextCacheConfig; map it to a DecisionRuleConfig
            self.config = DecisionRuleConfig(
                min_embedding_similarity=config.min_embedding_similarity,
                min_document_overlap=config.min_document_overlap,
                intent_match_mode=config.intent_match_mode,
                intent_compatibility_matrix=config.intent_compatibility_matrix,
                debug_mode=getattr(config, "debug", False),
            )
        else:
            self.config = config

    def evaluate_candidates(
        self,
        current_intent: str,
        current_doc_ids: List[str],
        candidates_with_scores: List[Tuple[CacheEntry, float]],
        current_doc_versions: Optional[List[str]] = None,
        current_tenant_id: Optional[str] = None,
        current_user_id: Optional[str] = None,
    ) -> Tuple[Optional[CacheEntry], str, float]:
        """
        Evaluates a list of (CacheEntry, similarity_score) candidates.

        Returns:
            (best_candidate, reason, confidence_score)
        """

        if not candidates_with_scores:
            return None, "Miss: No candidates provided for evaluation.", 0.0

        # Sort by highest semantic similarity first
        sorted_candidates = sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)

        last_failure_reason = "Miss: No candidates passed evaluation."
        highest_miss_confidence = 0.0

        for candidate, sim_score in sorted_candidates:
            # -----------------------------
            # Check 0a: Scope Validation
            # -----------------------------
            scope_valid = False
            if candidate.scope == "global":
                scope_valid = True
            elif candidate.scope == "tenant":
                scope_valid = (
                    current_tenant_id is not None and candidate.tenant_id == current_tenant_id
                )
            elif candidate.scope == "user":
                scope_valid = (
                    current_tenant_id is not None
                    and candidate.tenant_id == current_tenant_id
                    and current_user_id is not None
                    and candidate.user_id == current_user_id
                )

            if not scope_valid:
                last_failure_reason = (
                    f"Miss: Scope validation mismatch (candidate.scope='{candidate.scope}', "
                    f"candidate.tenant_id='{candidate.tenant_id}', candidate.user_id='{candidate.user_id}'; "
                    f"current_tenant_id='{current_tenant_id}', current_user_id='{current_user_id}')"
                )
                if self.config.debug_mode:
                    print(f"    -> REJECTED: {last_failure_reason}")
                continue

            # -----------------------------
            # Check 0b: Tenant Validation
            # -----------------------------
            if candidate.scope != "global" and candidate.tenant_id != current_tenant_id:
                last_failure_reason = (
                    f"Miss: Tenant validation mismatch "
                    f"(candidate.tenant_id='{candidate.tenant_id}', current_tenant_id='{current_tenant_id}')"
                )
                if self.config.debug_mode:
                    print(f"    -> REJECTED: {last_failure_reason}")
                continue

            # Compute document overlap proactively
            overlap = compute_document_overlap(current_doc_ids, candidate.doc_ids)

            # -----------------------------
            # Confidence Formula
            # 40% Similarity + 40% Overlap + 20% Intent
            # -----------------------------
            intent_val = 1.0 if candidate.intent == current_intent else 0.5
            confidence = (max(0, sim_score) * 0.4) + (overlap * 0.4) + (intent_val * 0.2)

            if confidence > highest_miss_confidence:
                highest_miss_confidence = confidence

            if self.config.debug_mode:
                print(f"  [DEBUG Evaluator] Inspecting Semantic Candidate: '{candidate.query}'")
                print(f"    - Overlap: {overlap:.3f}")
                print(f"    - Similarity: {sim_score:.3f}")
                print(f"    - Confidence: {confidence:.3f}")

            # -----------------------------
            # Check 1: Intent Matching
            # -----------------------------
            intent_match = False

            if self.config.intent_match_mode == "strict":
                intent_match = candidate.intent == current_intent

            elif self.config.intent_match_mode == "compatible":
                # Get explicit allowed intents from matrix.
                allowed = self.config.intent_compatibility_matrix.get(current_intent, set())
                # If list is empty, it elegantly falls back to strict equality.
                intent_match = (candidate.intent == current_intent) or (candidate.intent in allowed)

            elif self.config.intent_match_mode == "relaxed":
                intent_match = True

            if not intent_match:
                last_failure_reason = (
                    f"Miss: Intent mismatch under '{self.config.intent_match_mode}' mode "
                    f"(expected='{current_intent}', got='{candidate.intent}')"
                )
                if self.config.debug_mode:
                    print(f"    -> REJECTED: {last_failure_reason}")
                continue

            # -----------------------------
            # Check 2: Embedding Similarity
            # -----------------------------
            if sim_score < self.config.min_embedding_similarity:
                last_failure_reason = (
                    f"Miss: Low similarity "
                    f"({sim_score:.3f} < {self.config.min_embedding_similarity})"
                )
                continue

            # -----------------------------
            # Check 3: Document Overlap
            # -----------------------------
            if overlap < self.config.min_document_overlap:
                last_failure_reason = (
                    f"Miss: Low doc overlap "
                    f"({overlap:.3f} < {self.config.min_document_overlap})"
                )
                if self.config.debug_mode:
                    print(f"    -> REJECTED: {last_failure_reason}")
                continue

            # -----------------------------
            # Check 4: Document Version Matching
            # -----------------------------
            if current_doc_versions and candidate.doc_versions:
                current_v_map = dict(zip(current_doc_ids, current_doc_versions))
                cand_v_map = dict(zip(candidate.doc_ids, candidate.doc_versions))

                version_drift = False
                for doc_id, version in current_v_map.items():
                    if doc_id in cand_v_map and cand_v_map[doc_id] != version:
                        version_drift = True
                        break

                if version_drift:
                    last_failure_reason = "Miss: Document versions (hashes) have drifted."
                    if self.config.debug_mode:
                        print(f"    -> REJECTED: {last_failure_reason}")
                    continue

            # -----------------------------
            # ✅ All checks passed → HIT
            # -----------------------------
            if self.config.debug_mode:
                print(f"    -> ACCEPTED: Threshold passed! Yielding Candidate.")
            reason = (
                f"Hit: similarity={sim_score:.3f} ≥ {self.config.min_embedding_similarity}, "
                f"overlap={overlap:.3f} ≥ {self.config.min_document_overlap}, "
                f"intent_match={intent_match}"
            )

            return candidate, reason, confidence

        # -----------------------------
        # ❌ No candidate passed
        # -----------------------------
        return None, last_failure_reason, highest_miss_confidence
