from typing import Dict
# CHANGED: Use ChatOpenAI instead of ChatDeepSeek (DeepSeek is OpenAI-compatible)
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

class MedicalGenerator:
    def __init__(self):
        # Configuration for DeepSeek API
        self.llm = ChatOpenAI(
            model="deepseek-reasoner", 
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

    def generate(self, augmented_input: Dict) -> Dict:
        """Generate response with safety checks."""
        # The prompt is passed as a string, so we invoke it directly
        response = self.llm.invoke(augmented_input["prompt"])
        
        output = {
            "response": response.content,
            "metadata": augmented_input["metadata"],
            "confidence": self._estimate_confidence(response.content, augmented_input["metadata"])
        }
        return output

    def _estimate_confidence(self, response: str, metadata: Dict) -> float:
        """Simple confidence estimate based on source recency and coverage."""
        if not metadata or "sources" not in metadata or not metadata["sources"]:
            return 0.5
        
        # Safety check for missing last_updated
        last_updated = metadata.get("last_updated", "")
        recency_bonus = 0.2 if last_updated and "2024" in str(last_updated) else 0
        
        return min(1.0, 0.7 + recency_bonus)

    def combine_kb_and_rag(self, kb_response: str, rag_response: Dict) -> Dict:
        """Combine KB and RAG responses for hybrid queries."""
        combined_prompt = f"Integrate KB: {kb_response} with RAG: {rag_response['response']}. Provide a unified response."
        merged_response = self.llm.invoke(combined_prompt)
        return {
            "response": merged_response.content,
            "metadata": {**rag_response.get("metadata", {}), "kb_source": "Structured KB"},
            "confidence": self._estimate_confidence(merged_response.content, rag_response.get("metadata", {}))
        }

if __name__ == "__main__":
    # Absolute imports for direct execution from root
    from rag.augmentation.augmenter import MedicalAugmenter
    from rag.retrieval.retriever import MedicalRetriever
    
    print("Initializing RAG Pipeline...")
    # Ensure you have .env variables set!
    if not DEEPSEEK_API_KEY:
        print("WARNING: DEEPSEEK_API_KEY not found in environment")

    try:
        retriever = MedicalRetriever()
        augmenter = MedicalAugmenter()
        generator = MedicalGenerator()
        
        query = "diabetes management"
        print(f"Testing query: {query}")
        
        docs = retriever.retrieve(query, strategy="mmr")
        augmented = augmenter.augment(query, docs)
        response = generator.generate(augmented)
        
        print(f"\nResponse: {response['response']}\nConfidence: {response['confidence']}")
    except Exception as e:
        print(f"Test failed: {e}")