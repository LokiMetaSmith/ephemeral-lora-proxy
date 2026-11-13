from abc import ABC, abstractmethod
from typing import List, Dict, Any


class TrainerBackend(ABC):
    """Abstract base interface for Ephemeral Weight Consolidation backends."""

    @abstractmethod
    def train(self, distilled_pairs: List[Dict[str, str]], session_id: str, config: Dict[str, Any]) -> bool:
        """
        Train on distilled Q&A pairs and atomically update the GGUF adapter.
        Returns True on success, False otherwise.
        """
        pass

    @abstractmethod
    def reset_adapter(self, config: Dict[str, Any]) -> bool:
        """
        Reset ephemeral adapter to seed/zero state.
        """
        pass
