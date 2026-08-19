from typing import Any, Dict


class MetricsTracker:
    """
    A comprehensive tracker for monitoring cache performance and ROI.
    Tracks hits, calculates throughput metrics, and estimates time and token savings.
    """

    def __init__(
        self,
        avg_llm_generation_time_sec: float = 1.5,
        avg_tokens_per_response: int = 250,
        cache_overhead_sec: float = 0.015,
    ):
        self.hits = 0
        self.misses = 0

        # Heuristics for calculating savings
        self.avg_llm_generation_time_sec = avg_llm_generation_time_sec
        self.avg_tokens_per_response = avg_tokens_per_response
        self.cache_overhead_sec = cache_overhead_sec

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self) -> None:
        self.misses += 1

    def get_stats(self) -> Dict[str, Any]:
        """Calculates and returns current cache performance and efficiency metrics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0

        # ---------------------------------------------------------
        # Estimation Logic
        # ---------------------------------------------------------
        # Total time saved is the LLM time avoided MINUS the time spent checking the cache
        total_time_saved_sec = (self.hits * self.avg_llm_generation_time_sec) - (
            total * self.cache_overhead_sec
        )

        # Bound it to zero just in case cache overhead was magically higher than savings
        total_time_saved_sec = max(0.0, total_time_saved_sec)

        total_tokens_saved = self.hits * self.avg_tokens_per_response

        if total > 0:
            avg_time_saved_per_request = total_time_saved_sec / total

            # Efficiency is (Net Time Saved) / (Theoretical Time If No Cache Existed)
            theoretical_total_time = total * self.avg_llm_generation_time_sec
            cache_efficiency = total_time_saved_sec / theoretical_total_time
        else:
            avg_time_saved_per_request = 0.0
            cache_efficiency = 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_percentage": f"{round(hit_rate * 100, 2)}%",
            "roi_estimates": {
                "total_time_saved_sec": round(total_time_saved_sec, 3),
                "avg_latency_saved_per_request": round(avg_time_saved_per_request, 3),
                "estimated_tokens_saved": total_tokens_saved,
                "cache_efficiency_percentage": f"{round(cache_efficiency * 100, 2)}%",
            },
        }
