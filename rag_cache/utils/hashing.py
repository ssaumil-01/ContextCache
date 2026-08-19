from typing import List


def compute_document_overlap(actual_doc_ids: List[str], cached_doc_ids: List[str]) -> float:
    """
    Computes a Position-Weighted Overlap (Rank Biased) between retrieved docs.
    Matches at the top of the context window are weighted much more than those at the bottom.
    If a document shifts positions (e.g. from rank 1 to rank 3), it incurs a decay penalty.

    Returns:
        float: A normalized score between 0.0 (no overlap) and 1.0 (perfect spatial overlap).
    """
    if not actual_doc_ids and not cached_doc_ids:
        return 1.0

    if not actual_doc_ids or not cached_doc_ids:
        return 0.0

    # O(1) hash map lookup for cached document rankings
    cached_positions = {doc_id: i for i, doc_id in enumerate(cached_doc_ids)}

    total_possible_weight = 0.0
    achieved_weight = 0.0

    for i, doc_id in enumerate(actual_doc_ids):
        # Base importance (Harmonic decay: 1.0, 0.5, 0.33, 0.25, etc.)
        position_weight = 1.0 / (i + 1)
        total_possible_weight += position_weight

        if doc_id in cached_positions:
            # Apply a penalty if the ranking order shifted drastically
            shift_penalty = 1.0 / (1.0 + abs(i - cached_positions[doc_id]))
            achieved_weight += position_weight * shift_penalty

    # Normalize gracefully between 0.0 and 1.0
    return achieved_weight / total_possible_weight
