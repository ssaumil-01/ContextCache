import re

from rag_cache.interfaces.intent import IntentClassifier


class RuleBasedIntentClassifier(IntentClassifier):
    """
    A simple, fast keyword-based fallback intent classifier.
    Satisfies the IntentClassifier interface so it can be cleanly swapped out
    for an LLM-based or small-embedding-based classifier in production.
    """

    def __init__(self):
        # Heuristics mapped to intent categories. Checked in order.
        self.rules = {
            "action": [
                r"\b(create|update|delete|run|execute|start|stop|restart|add|remove|build|make)\b"
            ],
            "analytical": [
                r"\b(compare|analyze|evaluate|trend|stats|average|difference|metrics|performance)\b",
                r"\b(why did|how many|how much)\b",
            ],
            "navigation": [r"\b(go to|show me|navigate|open|where is|link|page|url|dashboard)\b"],
            "informational": [
                r"\b(what|who|when|where|how|explain|describe|define)\b",
                r"\b(is|are|details|info)\b",
            ],
        }

        # Precompile to avoid overhead per-query
        self.compiled_rules = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.rules.items()
        }

    def classify(self, query: str) -> str:
        """
        Determine the intent or semantic category of the query (e.g., 'factual', 'creative', 'bypass').
        Defaults to 'informational' if no other rules match.
        """
        for intent, patterns in self.compiled_rules.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent
        return "informational"
