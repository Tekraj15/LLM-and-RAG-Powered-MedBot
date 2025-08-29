"""
Enhanced RAG Vector Store for Medical Knowledge with Pinecone
Implements intelligent document retrieval with source attribution and confidence scoring.
"""

import json
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import time

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MedicalDocument:
    """Represents a medical knowledge document with metadata"""
    content: str
    source: str
    category: str
    confidence: float
    last_updated: str
    doc_id: str
    metadata: Dict[str, Any]

@dataclass
class RetrievalResult:
    """Result from vector search with confidence and attribution"""
    document: MedicalDocument
    relevance_score: float
    context_snippet: str

class MedicalVectorStore:
    """Advanced Pinecone vector store for medical knowledge with source tracking"""
    
    def __init__(self, index_name: str = "medical-knowledge", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the medical vector store with Pinecone
        
        Args:
            index_name: Name of the Pinecone index
            model_name: Sentence transformer model for embeddings
        """
        self.index_name = index_name
        self.model_name = model_name
        
        # Initialize embedding model
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=api_key)
        
        # Create or connect to index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        # Document metadata store (for full content retrieval)
        self.document_store = {}
        
        logger.info(f"Initialized MedicalVectorStore with Pinecone index: {self.index_name}")
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Change to your preferred region
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            logger.info(f"Index {self.index_name} created successfully")
        else:
            logger.info(f"Using existing Pinecone index: {self.index_name}")
    
    def add_knowledge_base(self, kb_path: str) -> None:
        """
        Add existing knowledge base to Pinecone vector store
        
        Args:
            kb_path: Path to the JSON knowledge base file
        """
        try:
            with open(kb_path, 'r') as f:
                kb_data = json.load(f)
            
            documents_to_upsert = []
            
            # Process symptoms
            for symptom, data in kb_data.get("symptoms", {}).items():
                doc_id = f"symptom_{symptom}_{self._generate_id()}"
                content = f"Symptom: {symptom}\n"
                content += f"Description: {data.get('description', '')}\n"
                content += f"Common causes: {', '.join(data.get('common_causes', []))}\n"
                
                if 'urgency' in data:
                    content += "Urgency guidelines:\n"
                    for duration, advice in data['urgency'].items():
                        content += f"- {duration}: {advice}\n"
                
                metadata = {
                    "category": "symptom",
                    "source": "internal_kb",
                    "confidence": 0.9,
                    "last_updated": kb_data.get("last_updated", "2023-10-01"),
                    "entity": symptom,
                    "content": content  # Store full content in metadata
                }
                
                # Generate embedding
                embedding = self.encoder.encode(content).tolist()
                
                documents_to_upsert.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                # Store in local document store
                self.document_store[doc_id] = MedicalDocument(
                    content=content,
                    source="internal_kb",
                    category="symptom",
                    confidence=0.9,
                    last_updated=kb_data.get("last_updated", "2023-10-01"),
                    doc_id=doc_id,
                    metadata={"entity": symptom}
                )
            
            # Process medications
            for med, data in kb_data.get("medications", {}).items():
                doc_id = f"medication_{med}_{self._generate_id()}"
                content = f"Medication: {med}\n"
                content += f"Uses: {data.get('uses', '')}\n"
                content += f"Side effects: {', '.join(data.get('side_effects', []))}\n"
                content += f"Precautions: {data.get('precautions', '')}\n"
                
                if 'max_daily_dose' in data:
                    content += f"Maximum daily dose: {data['max_daily_dose']}\n"
                
                metadata = {
                    "category": "medication",
                    "source": "internal_kb",
                    "confidence": 0.9,
                    "last_updated": kb_data.get("last_updated", "2023-10-01"),
                    "entity": med,
                    "content": content
                }
                
                embedding = self.encoder.encode(content).tolist()
                
                documents_to_upsert.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                self.document_store[doc_id] = MedicalDocument(
                    content=content,
                    source="internal_kb",
                    category="medication",
                    confidence=0.9,
                    last_updated=kb_data.get("last_updated", "2023-10-01"),
                    doc_id=doc_id,
                    metadata={"entity": med}
                )
            
            # Process interactions
            for interaction, description in kb_data.get("interactions", {}).items():
                doc_id = f"interaction_{interaction}_{self._generate_id()}"
                content = f"Drug interaction: {interaction}\n"
                content += f"Effect: {description}\n"
                
                metadata = {
                    "category": "interaction",
                    "source": "internal_kb",
                    "confidence": 0.95,
                    "last_updated": kb_data.get("last_updated", "2023-10-01"),
                    "entity": interaction,
                    "content": content
                }
                
                embedding = self.encoder.encode(content).tolist()
                
                documents_to_upsert.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                self.document_store[doc_id] = MedicalDocument(
                    content=content,
                    source="internal_kb",
                    category="interaction",
                    confidence=0.95,
                    last_updated=kb_data.get("last_updated", "2023-10-01"),
                    doc_id=doc_id,
                    metadata={"entity": interaction}
                )
            
            # Process chronic conditions
            for condition, data in kb_data.get("chronic_conditions", {}).items():
                doc_id = f"chronic_{condition}_{self._generate_id()}"
                content = f"Chronic condition: {condition}\n"
                
                if 'management' in data:
                    content += "Management guidelines:\n"
                    for key, value in data['management'].items():
                        if isinstance(value, list):
                            content += f"- {key}: {', '.join(value)}\n"
                        else:
                            content += f"- {key}: {value}\n"
                
                metadata = {
                    "category": "chronic_condition",
                    "source": "internal_kb",
                    "confidence": 0.9,
                    "last_updated": kb_data.get("last_updated", "2023-10-01"),
                    "entity": condition,
                    "content": content
                }
                
                embedding = self.encoder.encode(content).tolist()
                
                documents_to_upsert.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                self.document_store[doc_id] = MedicalDocument(
                    content=content,
                    source="internal_kb",
                    category="chronic_condition",
                    confidence=0.9,
                    last_updated=kb_data.get("last_updated", "2023-10-01"),
                    doc_id=doc_id,
                    metadata={"entity": condition}
                )
            
            # Upsert to Pinecone in batches
            if documents_to_upsert:
                batch_size = 100
                for i in range(0, len(documents_to_upsert), batch_size):
                    batch = documents_to_upsert[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                
                logger.info(f"Added {len(documents_to_upsert)} documents to Pinecone index")
            
        except Exception as e:
            logger.error(f"Error adding knowledge base: {str(e)}")
            raise
    
    def add_medical_document(self, document: MedicalDocument) -> None:
        """
        Add a single medical document to the Pinecone vector store
        
        Args:
            document: MedicalDocument instance to add
        """
        try:
            # Generate embedding
            embedding = self.encoder.encode(document.content).tolist()
            
            # Prepare metadata
            metadata = {
                "source": document.source,
                "category": document.category,
                "confidence": document.confidence,
                "last_updated": document.last_updated,
                "content": document.content,
                **document.metadata
            }
            
            # Upsert to Pinecone
            self.index.upsert(vectors=[{
                "id": document.doc_id,
                "values": embedding,
                "metadata": metadata
            }])
            
            # Store in local document store
            self.document_store[document.doc_id] = document
            
            logger.info(f"Added document {document.doc_id} to Pinecone vector store")
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query using Pinecone
        
        Args:
            query: User query text
            top_k: Number of top results to return
            category_filter: Filter by document category (optional)
            
        Returns:
            List of RetrievalResult objects with ranked relevance
        """
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode(query).tolist()
            
            # Prepare metadata filter
            filter_dict = {}
            if category_filter:
                filter_dict = {"category": {"$eq": category_filter}}
            
            # Perform vector search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Process results
            retrieval_results = []
            
            for match in search_results.matches:
                doc_id = match.id
                score = match.score  # Pinecone returns similarity score (higher is better)
                metadata = match.metadata
                
                # Get content from metadata or local store
                content = metadata.get('content', '')
                if not content and doc_id in self.document_store:
                    content = self.document_store[doc_id].content
                
                # Create MedicalDocument
                med_doc = MedicalDocument(
                    content=content,
                    source=metadata.get('source', 'unknown'),
                    category=metadata.get('category', 'general'),
                    confidence=metadata.get('confidence', 0.5),
                    last_updated=metadata.get('last_updated', datetime.now().isoformat()),
                    doc_id=doc_id,
                    metadata=metadata
                )
                
                # Generate context snippet (first 200 chars)
                context_snippet = content[:200] + "..." if len(content) > 200 else content
                
                retrieval_results.append(RetrievalResult(
                    document=med_doc,
                    relevance_score=score,  # Pinecone score is already 0-1 similarity
                    context_snippet=context_snippet
                ))
            
            # Results are already sorted by relevance score from Pinecone
            logger.info(f"Retrieved {len(retrieval_results)} documents for query: {query[:50]}...")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []
    
    def get_emergency_keywords(self) -> List[str]:
        """Return list of emergency keywords for immediate detection"""
        return [
            "chest pain", "heart attack", "stroke", "seizure", "overdose",
            "suicide", "emergency", "911", "unconscious", "bleeding",
            "severe allergic reaction", "anaphylaxis", "choking"
        ]
    
    def is_emergency_query(self, query: str) -> bool:
        """
        Detect if query indicates medical emergency
        
        Args:
            query: User query text
            
        Returns:
            True if emergency detected, False otherwise
        """
        query_lower = query.lower()
        emergency_keywords = self.get_emergency_keywords()
        
        return any(keyword in query_lower for keyword in emergency_keywords)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone vector store"""
        try:
            # Get index stats from Pinecone
            index_stats = self.index.describe_index_stats()
            total_count = index_stats.total_vector_count
            
            # Calculate category and source distribution from local document store
            categories = {}
            sources = {}
            
            for doc in self.document_store.values():
                cat = doc.category
                src = doc.source
                
                categories[cat] = categories.get(cat, 0) + 1
                sources[src] = sources.get(src, 0) + 1
            
            return {
                "total_documents": total_count,
                "local_documents": len(self.document_store),
                "categories": categories,
                "sources": sources,
                "index_stats": index_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
    
    def _generate_id(self) -> str:
        """Generate unique ID for documents"""
        return hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    
    def reset_store(self) -> None:
        """Reset the Pinecone vector store (use with caution!)"""
        try:
            # Delete all vectors from the index
            self.index.delete(delete_all=True)
            
            # Clear local document store
            self.document_store.clear()
            
            logger.info("Pinecone vector store reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting store: {str(e)}")
            raise
    
    def delete_index(self) -> None:
        """Delete the entire Pinecone index (use with extreme caution!)"""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Pinecone index {self.index_name} deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Note: Make sure to set PINECONE_API_KEY environment variable
    
    try:
        # Initialize vector store
        store = MedicalVectorStore()
        
        # Add knowledge base
        kb_path = "../Knowledge-base/med_knowledge.json"
        if os.path.exists(kb_path):
            store.add_knowledge_base(kb_path)
        
        # Test retrieval
        query = "What are the symptoms of diabetes?"
        results = store.retrieve(query, top_k=3)
        
        print(f"Query: {query}")
        print(f"Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Relevance: {result.relevance_score:.3f}")
            print(f"   Source: {result.document.source}")
            print(f"   Category: {result.document.category}")
            print(f"   Snippet: {result.context_snippet}")
        
        # Print statistics
        stats = store.get_collection_stats()
        print(f"\nVector Store Statistics: {stats}")
        
    except ValueError as e:
        print(f"Setup Error: {e}")
        print("Please set the PINECONE_API_KEY environment variable")
    except Exception as e:
        print(f"Error: {e}")
