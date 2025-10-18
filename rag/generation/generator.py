from typing import Dict
from langchain_community.chat_models import ChatDeepSeek  # Assuming custom wrapper
import os
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

class MedicalGenerator:
    def __init__(self):
        self.llm = ChatDeepSeek(model="deepseek-r1", api_key=DEEPSEEK_API_KEY)

    def generate(self, augmented_input: Dict) -> Dict:
        """Generate response with safety checks."""
        response = self.llm.invoke(augmented_input["prompt"])
        output = {
            "response": response.content,
            "metadata": augmented_input["metadata"],
            "confidence": self._estimate_confidence(response.content, augmented_input["metadata"])
        }
        return output

    def _estimate_confidence(self, response: str, metadata: Dict) -> float:
        """Simple confidence estimate based on source recency and coverage."""
        if not metadata["sources"]:
            return 0.5
        recency_bonus = 0.2 if metadata["last_updated"] and "2024" in metadata["last_updated"] else 0
        return min(1.0, 0.7 + recency_bonus)  # Baseline 0.7, +0.2 for recent data

    def combine_kb_and_rag(self, kb_response: str, rag_response: Dict) -> Dict:
        """Combine KB and RAG responses for hybrid queries."""
        combined_prompt = f"Integrate KB: {kb_response} with RAG: {rag_response['response']}. Provide a unified response."
        merged_response = self.llm.invoke(combined_prompt)
        return {
            "response": merged_response.content,
            "metadata": {**rag_response["metadata"], "kb_source": "Structured KB"},
            "confidence": self._estimate_confidence(merged_response.content, rag_response["metadata"])
        }

if __name__ == "__main__":
    from ..augmentation.augmenter import MedicalAugmenter
    from ..retrieval.retriever import MedicalRetriever
    retriever = MedicalRetriever()
    augmenter = MedicalAugmenter()
    generator = MedicalGenerator()
    docs = retriever.retrieve("diabetes management", strategy="mmr")
    augmented = augmenter.augment("diabetes management", docs)
    response = generator.generate(augmented)
    print(f"Response: {response['response']}\nConfidence: {response['confidence']}")