#!/usr/bin/env python3
"""
RAG System Test Suite for Medical Chatbot
Comprehensive testing of RAG components and end-to-end functionality.
"""

import os
import sys
import json
import logging
from typing import List, Dict
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Test case definition"""
    name: str
    query: str
    expected_type: str
    expected_emergency: bool = False
    description: str = ""

def run_comprehensive_tests():
    """Run comprehensive tests of the RAG system"""
    
    print("🧪 RAG System Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        # Import components
        from rag_core import MedicalVectorStore, MedicalRAGRouter, QueryType
        from safety_layer.validator import MedicalResponseValidator
        
        # Initialize components
        print("📊 Initializing RAG components with Pinecone...")
        
        # Check for Pinecone API key
        if not os.getenv("PINECONE_API_KEY"):
            print("❌ PINECONE_API_KEY environment variable not found")
            print("💡 Please set your Pinecone API key and run setup_rag.py first")
            return False
        
        vector_store = MedicalVectorStore(index_name="medical-knowledge-test")
        rag_router = MedicalRAGRouter(vector_store)
        validator = MedicalResponseValidator()
        
        # Define test cases
        test_cases = [
            TestCase(
                name="Emergency Detection",
                query="I'm having severe chest pain and can't breathe",
                expected_type="emergency",
                expected_emergency=True,
                description="Should detect medical emergency and provide immediate guidance"
            ),
            TestCase(
                name="Drug Interaction Query",
                query="Can I take ibuprofen with alcohol?",
                expected_type="drug_interaction",
                description="Should identify drug interaction concern and provide safety information"
            ),
            TestCase(
                name="Symptom Assessment",
                query="I have a headache that's lasted for 2 days",
                expected_type="symptom_check",
                description="Should assess symptoms and provide appropriate guidance"
            ),
            TestCase(
                name="Medication Information",
                query="What are the side effects of metformin?",
                expected_type="medication_info",
                description="Should provide medication information with proper disclaimers"
            ),
            TestCase(
                name="Chronic Care Management",
                query="How should I manage my diabetes?",
                expected_type="chronic_care",
                description="Should provide chronic condition management guidance"
            ),
            TestCase(
                name="Mental Health Support",
                query="I'm feeling anxious and having trouble sleeping",
                expected_type="mental_health",
                description="Should provide mental health support and resources"
            ),
            TestCase(
                name="General Health Query",
                query="What's a healthy diet for heart health?",
                expected_type="general_health",
                description="Should provide general health guidance"
            )
        ]
        
        print(f"🎯 Running {len(test_cases)} test cases...\n")
        
        passed_tests = 0
        failed_tests = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case.name}")
            print(f"Query: '{test_case.query}'")
            print(f"Description: {test_case.description}")
            
            try:
                # Test query classification
                routing_decision = rag_router.classify_query(test_case.query)
                
                # Test vector retrieval
                context_results = rag_router.retrieve_context(test_case.query, routing_decision)
                
                # Test response generation (mock LLM response)
                def mock_llm_response(prompt, context):
                    return f"Mock response based on: {context[:100]}..."
                
                rag_response = rag_router.generate_rag_response(
                    test_case.query, context_results, routing_decision.query_type, mock_llm_response
                )
                
                # Test safety validation
                validation_result = validator.validate_response(
                    rag_response.answer,
                    rag_response.sources,
                    routing_decision.query_type.value
                )
                
                # Verify results
                type_match = routing_decision.query_type.value == test_case.expected_type
                emergency_match = routing_decision.emergency_flag == test_case.expected_emergency
                
                if type_match and emergency_match:
                    print("✅ PASSED")
                    passed_tests += 1
                else:
                    print("❌ FAILED")
                    print(f"   Expected: {test_case.expected_type}, Emergency: {test_case.expected_emergency}")
                    print(f"   Got: {routing_decision.query_type.value}, Emergency: {routing_decision.emergency_flag}")
                    failed_tests += 1
                
                # Display detailed results
                print(f"   Query Type: {routing_decision.query_type.value}")
                print(f"   Confidence: {routing_decision.confidence:.2f}")
                print(f"   Emergency: {routing_decision.emergency_flag}")
                print(f"   Sources Found: {len(context_results)}")
                print(f"   Validation Level: {validation_result.level.value}")
                print(f"   Response Safe: {validation_result.is_safe}")
                
                if context_results:
                    print(f"   Top Source: {context_results[0].document.source}")
                
                print()
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                failed_tests += 1
                print()
        
        # Test Results Summary
        print("=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print(f"✅ Passed: {passed_tests}/{len(test_cases)}")
        print(f"❌ Failed: {failed_tests}/{len(test_cases)}")
        print(f"Success Rate: {(passed_tests/len(test_cases)*100):.1f}%")
        
        # Additional System Tests
        print("\n🔬 Additional System Tests")
        print("-" * 30)
        
        # Test vector store statistics
        stats = vector_store.get_collection_stats()
        print(f"Vector Store Documents: {stats.get('total_documents', 0)}")
        print(f"Categories: {len(stats.get('categories', {}))}")
        print(f"Sources: {len(stats.get('sources', {}))}")
        
        # Test emergency keyword detection
        emergency_keywords = vector_store.get_emergency_keywords()
        print(f"Emergency Keywords Loaded: {len(emergency_keywords)}")
        
        # Test safety patterns
        test_unsafe_response = "You should stop taking your medication immediately"
        unsafe_validation = validator.validate_response(test_unsafe_response)
        print(f"Unsafe Content Detection: {'✅' if not unsafe_validation.is_safe else '❌'}")
        
        print("\n🎯 Performance Insights")
        print("-" * 30)
        
        # Test retrieval performance
        import time
        start_time = time.time()
        test_results = vector_store.retrieve("diabetes management", top_k=5)
        retrieval_time = time.time() - start_time
        print(f"Vector Retrieval Time: {retrieval_time:.3f}s")
        print(f"Retrieved Documents: {len(test_results)}")
        
        if test_results:
            avg_relevance = sum(r.relevance_score for r in test_results) / len(test_results)
            print(f"Average Relevance Score: {avg_relevance:.3f}")
        
        print("\n🚀 RAG System Status: ", end="")
        if failed_tests == 0:
            print("✅ ALL SYSTEMS OPERATIONAL")
        elif failed_tests <= 2:
            print("⚠️ MOSTLY OPERATIONAL (Minor Issues)")
        else:
            print("❌ NEEDS ATTENTION (Multiple Failures)")
        
        return failed_tests == 0
        
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("💡 Run setup_rag.py first to initialize the system")
        return False
        
    except Exception as e:
        print(f"❌ Test Error: {str(e)}")
        logger.error(f"Test suite failed: {str(e)}")
        return False

def test_end_to_end_workflow():
    """Test complete end-to-end RAG workflow"""
    
    print("\n🔄 End-to-End Workflow Test")
    print("-" * 40)
    
    try:
        # Import the enhanced action function
        sys.path.append(os.path.join(os.path.dirname(__file__), 'rasa/actions'))
        
        # Test the integrated workflow (without RASA server)
        test_queries = [
            "What should I do for a persistent cough?",
            "Can I take aspirin with my blood pressure medication?",
            "I'm experiencing anxiety - what can help?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            # This would normally go through RASA, but we'll test the core function
            from rag_core import MedicalVectorStore, MedicalRAGRouter
            
            vector_store = MedicalVectorStore(index_name="medical-knowledge-test")
            rag_router = MedicalRAGRouter(vector_store)
            
            routing_decision = rag_router.classify_query(query)
            context_results = rag_router.retrieve_context(query, routing_decision)
            
            print(f"✓ Classified as: {routing_decision.query_type.value}")
            print(f"✓ Found {len(context_results)} relevant sources")
            
            if context_results:
                print(f"✓ Top source: {context_results[0].document.source}")
                print(f"✓ Relevance score: {context_results[0].relevance_score:.3f}")
        
        print("\n✅ End-to-end workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Starting RAG System Test Suite\n")
    
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Run end-to-end test
    e2e_success = test_end_to_end_workflow()
    
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST RESULTS")
    
    if success and e2e_success:
        print("🎉 All tests passed! RAG system is ready for deployment.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        sys.exit(1)
