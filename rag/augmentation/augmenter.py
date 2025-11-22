from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from rag.ingestion.document_schema import MedicalDocument

class MedicalAugmenter:
    def __init__(self):
        self.prompt_template = ChatPromptTemplate.from_template(
            """Based on the following context from medical sources:
            {context}
            Answer the query: {query}
            Sources and confidence: {metadata}
            Provide a simple, accurate response with source attribution."""
        )

    def augment(self, query: str, documents: List[MedicalDocument]) -> Dict:
        """Augment the query with retrieved contexts and metadata."""
        if not documents:
            return {"prompt": f"Answer: {query} (No relevant data found. Consult a doctor.)", "metadata": {}}

        context = "\n\n".join([f"- {doc.content}" for doc in documents])
        metadata = {
            "sources": [f"{doc.source} (Confidence: {doc.confidence})" for doc in documents],
            "last_updated": max((doc.last_updated for doc in documents if doc.last_updated), default=None)
        }
        prompt = self.prompt_template.format(context=context, query=query, metadata=metadata)
        return {"prompt": prompt, "metadata": metadata}

if __name__ == "__main__":
    from ..retrieval.retriever import MedicalRetriever
    retriever = MedicalRetriever()
    docs = retriever.retrieve("latest diabetes management", strategy="mmr")
    augmenter = MedicalAugmenter()
    augmented = augmenter.augment("latest diabetes management", docs)
    print(augmented["prompt"])