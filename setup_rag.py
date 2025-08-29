#!/usr/bin/env python3
"""
RAG System Setup Script for Medical Chatbot
Initializes vector database, tests components, and provides setup verification.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_rag_system():
    """Initialize and setup the complete RAG system"""
    
    print("🚀 Starting RAG System Setup for Medical Chatbot")
    print("=" * 60)
    
    try:
        # Import RAG components
        from rag_core import MedicalVectorStore, MedicalRAGRouter
        from safety_layer.validator import MedicalResponseValidator
        
        print("✅ RAG modules imported successfully")
        
        # Step 1: Initialize Pinecone Vector Store
        print("\n📊 Step 1: Initializing Pinecone Vector Database...")
        
        # Check for Pinecone API key
        if not os.getenv("PINECONE_API_KEY"):
            print("❌ PINECONE_API_KEY environment variable not found")
            print("💡 Please set your Pinecone API key:")
            print("   export PINECONE_API_KEY='your-api-key-here'")
            return False
        
        vector_store = MedicalVectorStore(index_name="medical-knowledge-setup")
        
        # Step 2: Load Knowledge Base
        print("\n📚 Step 2: Loading Medical Knowledge Base...")
        kb_path = "Knowledge-base/med_knowledge.json"
        
        if not os.path.exists(kb_path):
            print(f"Knowledge base file not found: {kb_path}")
            return False
        
        # Check if vector store needs population
        stats = vector_store.get_collection_stats()
        current_docs = stats.get("total_documents", 0)
        
        if current_docs == 0:
            print(f"Populating vector store with knowledge base...")
            vector_store.add_knowledge_base(kb_path)
            print(f"Added documents to vector store")
        else:
            print(f"Vector store already contains {current_docs} documents")
        
        # Step 3: Initialize Router and Validator
        print("\nStep 3: Initializing RAG Router...")
        rag_router = MedicalRAGRouter(vector_store)
        
        print("\nStep 4: Initializing Safety Validator...")
        validator = MedicalResponseValidator()
        
        # Step 5: Test System Components
        print("\nStep 5: Testing System Components...")
        
        # Test vector search
        test_query = "What are the symptoms of fever?"
        results = vector_store.retrieve(test_query, top_k=3)
        print(f"   ✓ Vector search test: Found {len(results)} relevant documents")
        
        # Test query classification
        routing_decision = rag_router.classify_query(test_query)
        print(f"   ✓ Query classification test: {routing_decision.query_type.value} (confidence: {routing_decision.confidence:.2f})")
        
        # Test emergency detection
        emergency_query = "I'm having severe chest pain"
        emergency_decision = rag_router.classify_query(emergency_query)
        print(f"   ✓ Emergency detection test: Emergency flag = {emergency_decision.emergency_flag}")
        
        # Test safety validation
        test_response = "You might have a cold. Consider taking some rest."
        validation_result = validator.validate_response(test_response)
        print(f"   ✓ Safety validation test: {validation_result.level.value} (safe: {validation_result.is_safe})")
        
        # Step 6: Display System Statistics
        print("\nStep 6: System Statistics")
        stats = vector_store.get_collection_stats()
        print(f"   • Total documents: {stats.get('total_documents', 0)}")
        print(f"   • Categories: {list(stats.get('categories', {}).keys())}")
        print(f"   • Sources: {list(stats.get('sources', {}).keys())}")
        
        print("\nPinecone RAG System Setup Complete!")
        print("=" * 60)
        print("Key Features Enabled:")
        print("   • Pinecone cloud vector database")
        print("   • Intelligent query routing")
        print("   • Scalable vector-based document retrieval")
        print("   • Emergency detection and protocols")
        print("   • Multi-layer safety validation")
        print("   • Source attribution and traceability")
        print("   • Confidence scoring")
        
        print("\nNext Steps:")
        print("   1. Run 'python test_rag_system.py' to test the system")
        print("   2. Start RASA server with 'rasa run actions'")
        print("   3. Test the chatbot with 'rasa shell'")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {str(e)}")
        print("\nSolution: Install required dependencies with:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"Setup error: {str(e)}")
        logger.error(f"Setup failed: {str(e)}")
        return False

def verify_dependencies():
    """Verify all required dependencies are installed"""
    
    print("🔍 Verifying Dependencies...")
    
    required_packages = [
        'sentence_transformers',
        'pinecone', 
        'transformers',
        'torch',
        'rasa',
        'rasa_sdk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   {package}")
        except ImportError:
            print(f"   {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("All dependencies verified!")
    return True

if __name__ == "__main__":
    print("🔧 RAG System Setup for Medical Chatbot")
    print("This script will initialize the RAG components and verify the setup.\n")
    
    # Verify dependencies first
    if not verify_dependencies():
        sys.exit(1)
    
    # Setup RAG system
    success = setup_rag_system()
    
    if success:
        print("\nSetup completed successfully!")
        sys.exit(0)
    else:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)
