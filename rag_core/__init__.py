"""
RAG Core Module for Medical Chatbot
Provides enhanced retrieval-augmented generation capabilities with source attribution.
"""

from .vector_store import MedicalVectorStore, MedicalDocument, RetrievalResult
from .rag_router import MedicalRAGRouter, QueryType, RoutingDecision, RAGResponse

__version__ = "1.0.0"
__author__ = "Medical AI Team"

__all__ = [
    "MedicalVectorStore",
    "MedicalDocument", 
    "RetrievalResult",
    "MedicalRAGRouter",
    "QueryType",
    "RoutingDecision",
    "RAGResponse"
]

