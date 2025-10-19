# Intent based routing for medical queries classification and emergency detection

from enum import Enum
from typing import Dict, Any, Optional

class QueryType(Enum):
    STRUCTURED = "structured"
    COMPLEX = "complex"
    EMERGENCY = "emergency"

class MedicalRAGRouter:
    def __init__(self):
        # Sync with validator.py emergency_indicators
        self.emergency_keywords = {
            "emergency", "chest pain", "bleeding", "unconscious",
            "heart attack", "stroke", "severe allergic reaction"
        }

    def classify_query(self, query: str, intent: Optional[str] = None) -> Dict[str, Any]:
        """Classify query type, detect emergencies, and suggest metadata filters."""
        query_lower = query.lower()
        metadata_filter = {}

        # Emergency detection
        if any(keyword in query_lower for keyword in self.emergency_keywords):
            return {
                "query_type": QueryType.EMERGENCY,
                "emergency_flag": True,
                "metadata_filter": {"category": "emergency"}
            }

        # Structured queries (specific intents or keywords)
        if intent in ["ask_medication", "ask_interaction"] or "specific" in query_lower or "interaction" in query_lower:
            metadata_filter = {"category": "medication"}
            return {
                "query_type": QueryType.STRUCTURED,
                "emergency_flag": False,
                "metadata_filter": metadata_filter
            }

        # Complex queries (default to open-ended)
        if "treatment" in query_lower or "management" in query_lower:
            metadata_filter = {"category": "treatment", "last_updated": {"$gte": "2023-01-01"}}
        return {
            "query_type": QueryType.COMPLEX,
            "emergency_flag": False,
            "metadata_filter": metadata_filter
        }

if __name__ == "__main__":
    router = MedicalRAGRouter()
    print(router.classify_query("aspirin and warfarin"))  # Structured
    print(router.classify_query("chest pain"))  # Emergency
    print(router.classify_query("diabetes management"))  # Complex