from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from .document_schema import MedicalDocument
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MedicalRetriever:
    def __init__(self, index_name: str = "medbot-rag"):
        """Initialize retriever with Pinecone and embeddings."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        self.vectorstore = PineconeVectorStore(
            embedding=self.embeddings,
            index_name=index_name,
            pinecone_api_key=PINECONE_API_KEY
        )
        self.base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})

    
    # Naive similarity-based retrieval
    def naive_retrieval(self, query: str) -> List[MedicalDocument]:
        """Basic similarity-based retrieval."""
        docs = self.base_retriever.get_relevant_documents(query)
        return [self._convert_to_medical_doc(doc) for doc in docs]

    
    # MMR (Maximal Marginal Relevance) for diversity-focused treatment
    def mmr_retrieval(self, query: str, lambda_mult: float = 0.7) -> List[MedicalDocument]:
        """Retrieval with Maximal Marginal Relevance for diversity."""
        mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "lambda_mult": lambda_mult}
        )
        docs = mmr_retriever.get_relevant_documents(query)
        return [self._convert_to_medical_doc(doc) for doc in docs]

    
    # precision-focused retrieval with reranking
    def rerank_retrieval(self, query: str, top_n: int = 5) -> List[MedicalDocument]:
        """Retrieval with reranking for precision."""
        compressor = CohereRerank(top_n=top_n)
        reranker = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
        docs = reranker.get_relevant_documents(query)
        return [self._convert_to_medical_doc(doc) for doc in docs]

    
    # Metadata-filtered retrieval
    def filtered_retrieval(self, query: str, metadata_filter: dict, strategy: str = "mmr") -> List[MedicalDocument]:
        """Retrieval with metadata filtering (e.g., recency, category)."""
        retriever = self.vectorstore.as_retriever(
            search_type=strategy if strategy == "mmr" else "similarity",
            search_kwargs={"k": 20, "filter": metadata_filter, "lambda_mult": 0.7 if strategy == "mmr" else None}
        )
        docs = retriever.get_relevant_documents(query)
        return [self._convert_to_medical_doc(doc) for doc in docs]


    # Maps LangChain’s Document to MedicalDocument for consistency with ingestion
    def _convert_to_medical_doc(self, doc) -> MedicalDocument:
        """Convert LangChain Document to MedicalDocument."""
        return MedicalDocument(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            category=doc.metadata.get("category", "general"),
            confidence=doc.metadata.get("confidence", 1.0),
            last_updated=doc.metadata.get("last_updated"),
            doc_id=doc.metadata.get("doc_id"),
            metadata=doc.metadata
        )

    def retrieve(self, query: str, strategy: str = "mmr", metadata_filter: Optional[dict] = None, top_n: int = 5) -> List[MedicalDocument]:
        """Unified retrieval method with strategy selection."""
        if metadata_filter:
            return self.filtered_retrieval(query, metadata_filter, strategy)
        elif strategy == "naive":
            return self.naive_retrieval(query)
        elif strategy == "mmr":
            return self.mmr_retrieval(query)
        elif strategy == "rerank":
            return self.rerank_retrieval(query)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

if __name__ == "__main__":
    retriever = MedicalRetriever()
    query = "latest treatment for long COVID"
    filter = {"category": "treatment", "last_updated": {"$gte": "2024-01-01"}}
    docs = retriever.retrieve(query, strategy="mmr", metadata_filter=filter)
    for doc in docs:
        print(f"Content: {doc.content[:100]}... Source: {doc.source}")