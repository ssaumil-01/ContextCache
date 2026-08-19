from abc import ABC, abstractmethod


class IntentClassifier(ABC):
    @abstractmethod
    def classify(self, query: str) -> str:
        """
        Determine the intent or semantic category of the query (e.g., 'factual', 'creative', 'bypass').
        This informs the orchestrator whether it should even attempt a cache lookup.
        """
        pass
