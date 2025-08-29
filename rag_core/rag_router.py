"""
Intelligent RAG Router for Medical Queries
Routes queries to appropriate retrieval strategies based on intent and urgency.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import json

from .vector_store import MedicalVectorStore, RetrievalResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of medical queries for routing decisions"""
    EMERGENCY = "emergency"
    DRUG_INTERACTION = "drug_interaction"
    SYMPTOM_CHECK = "symptom_check"
    MEDICATION_INFO = "medication_info"
    CHRONIC_CARE = "chronic_care"
    MENTAL_HEALTH = "mental_health"
    GENERAL_HEALTH = "general_health"
    DIAGNOSTIC = "diagnostic"

@dataclass
class RoutingDecision:
    """Represents routing decision with confidence and strategy"""
    query_type: QueryType
    confidence: float
    retrieval_strategy: str
    emergency_flag: bool
    recommended_k: int
    category_filter: Optional[str] = None

@dataclass
class RAGResponse:
    """Complete RAG response with context and attribution"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_type: QueryType
    emergency_detected: bool
    context_used: str

class MedicalRAGRouter:
    """
    Intelligent router that determines the best RAG strategy for medical queries
    """
    
    def __init__(self, vector_store: MedicalVectorStore):
        """
        Initialize the RAG router
        
        Args:
            vector_store: Initialized MedicalVectorStore instance
        """
        self.vector_store = vector_store
        self.load_routing_patterns()
        logger.info("MedicalRAGRouter initialized")
    
    def load_routing_patterns(self) -> None:
        """Load routing patterns and keywords for query classification"""
        
        # Emergency keywords (highest priority)
        self.emergency_patterns = [
            r'\b(chest pain|heart attack|stroke|seizure|overdose)\b',
            r'\b(suicide|suicidal|kill myself)\b',
            r'\b(emergency|911|urgent|immediate)\b',
            r'\b(unconscious|bleeding heavily|choking)\b',
            r'\b(severe allergic reaction|anaphylaxis)\b',
            r'\b(can\'t breathe|difficulty breathing)\b'
        ]
        
        # Drug interaction patterns
        self.interaction_patterns = [
            r'\b(interaction|combine|mix|together)\b.*\b(medication|drug|pill)\b',
            r'\bcan i take\b.*\bwith\b',
            r'\b(safe to|danger|risk).*\b(combine|mix)\b',
            r'\bmedicine.*\btogether\b'
        ]
        
        # Symptom check patterns
        self.symptom_patterns = [
            r'\b(symptom|feel|pain|ache|hurt)\b',
            r'\b(headache|fever|nausea|dizzy)\b',
            r'\bhave.*\b(for|since)\b',
            r'\bwhat could cause\b',
            r'\bexperiencing\b'
        ]
        
        # Medication info patterns
        self.medication_patterns = [
            r'\b(side effects?|dosage|how much)\b',
            r'\bmedication.*\b(for|treat)\b',
            r'\btake.*\b(daily|times)\b',
            r'\bprescription\b'
        ]
        
        # Chronic care patterns
        self.chronic_patterns = [
            r'\b(diabetes|hypertension|asthma|arthritis)\b',
            r'\b(manage|management|control)\b',
            r'\b(diet|exercise|lifestyle)\b.*\b(chronic|condition)\b',
            r'\bblood (pressure|sugar|glucose)\b'
        ]
        
        # Mental health patterns
        self.mental_health_patterns = [
            r'\b(anxiety|depression|stress|mental)\b',
            r'\b(mood|feeling|emotional)\b',
            r'\b(therapy|counseling|support)\b',
            r'\b(sleep|insomnia|tired)\b'
        ]
        
        # Diagnostic patterns
        self.diagnostic_patterns = [
            r'\b(test|lab|blood work|scan)\b',
            r'\b(diagnosis|diagnose|what is)\b',
            r'\b(results|levels|values)\b',
            r'\b(normal range|abnormal)\b'
        ]
    
    def classify_query(self, query: str, rasa_intent: Optional[str] = None) -> RoutingDecision:
        """
        Classify query and determine routing strategy
        
        Args:
            query: User query text
            rasa_intent: Intent from RASA NLU (optional)
            
        Returns:
            RoutingDecision object with routing strategy
        """
        query_lower = query.lower()
        
        # Check for emergency first (highest priority)
        if self._matches_patterns(query_lower, self.emergency_patterns):
            return RoutingDecision(
                query_type=QueryType.EMERGENCY,
                confidence=0.95,
                retrieval_strategy="emergency_protocol",
                emergency_flag=True,
                recommended_k=1,
                category_filter=None
            )
        
        # Use RASA intent if available for better classification
        if rasa_intent:
            intent_mapping = {
                "ask_medication": QueryType.MEDICATION_INFO,
                "symptom_check": QueryType.SYMPTOM_CHECK,
                "chronic_care": QueryType.CHRONIC_CARE,
                "mental_health": QueryType.MENTAL_HEALTH
            }
            
            if rasa_intent in intent_mapping:
                query_type = intent_mapping[rasa_intent]
                return self._create_routing_decision(query_type, query_lower, 0.9)
        
        # Pattern-based classification
        if self._matches_patterns(query_lower, self.interaction_patterns):
            return self._create_routing_decision(QueryType.DRUG_INTERACTION, query_lower, 0.85)
        
        if self._matches_patterns(query_lower, self.symptom_patterns):
            return self._create_routing_decision(QueryType.SYMPTOM_CHECK, query_lower, 0.8)
        
        if self._matches_patterns(query_lower, self.medication_patterns):
            return self._create_routing_decision(QueryType.MEDICATION_INFO, query_lower, 0.8)
        
        if self._matches_patterns(query_lower, self.chronic_patterns):
            return self._create_routing_decision(QueryType.CHRONIC_CARE, query_lower, 0.8)
        
        if self._matches_patterns(query_lower, self.mental_health_patterns):
            return self._create_routing_decision(QueryType.MENTAL_HEALTH, query_lower, 0.8)
        
        if self._matches_patterns(query_lower, self.diagnostic_patterns):
            return self._create_routing_decision(QueryType.DIAGNOSTIC, query_lower, 0.75)
        
        # Default to general health
        return RoutingDecision(
            query_type=QueryType.GENERAL_HEALTH,
            confidence=0.6,
            retrieval_strategy="hybrid_search",
            emergency_flag=False,
            recommended_k=5,
            category_filter=None
        )
    
    def retrieve_context(self, query: str, routing_decision: RoutingDecision) -> List[RetrievalResult]:
        """
        Retrieve context based on routing decision
        
        Args:
            query: User query
            routing_decision: Routing decision from classify_query
            
        Returns:
            List of relevant documents
        """
        try:
            if routing_decision.emergency_flag:
                # For emergencies, return emergency protocol immediately
                return self._get_emergency_response()
            
            # Standard retrieval
            results = self.vector_store.retrieve(
                query=query,
                top_k=routing_decision.recommended_k,
                category_filter=routing_decision.category_filter
            )
            
            # Apply query-type specific filtering and ranking
            filtered_results = self._apply_type_specific_filtering(results, routing_decision.query_type)
            
            logger.info(f"Retrieved {len(filtered_results)} documents for {routing_decision.query_type.value}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in retrieve_context: {str(e)}")
            return []
    
    def generate_rag_response(self, query: str, context: List[RetrievalResult], 
                            query_type: QueryType, llm_generate_func) -> RAGResponse:
        """
        Generate final RAG response using LLM with retrieved context
        
        Args:
            query: Original user query
            context: Retrieved context documents
            query_type: Type of query
            llm_generate_func: Function to call LLM (from actions.py)
            
        Returns:
            Complete RAG response with sources
        """
        try:
            if not context:
                return RAGResponse(
                    answer="I couldn't find specific information about your query. Please consult a healthcare professional.",
                    sources=[],
                    confidence=0.1,
                    query_type=query_type,
                    emergency_detected=False,
                    context_used=""
                )
            
            # Handle emergency queries
            if query_type == QueryType.EMERGENCY:
                return RAGResponse(
                    answer="⚠️ MEDICAL EMERGENCY DETECTED ⚠️\n\nPlease call emergency services immediately (911 in US, 999 in UK, 112 in EU) or go to the nearest emergency room. Do not delay seeking immediate medical attention.",
                    sources=[{"source": "Emergency Protocol", "confidence": 1.0}],
                    confidence=1.0,
                    query_type=query_type,
                    emergency_detected=True,
                    context_used="Emergency protocol activated"
                )
            
            # Build context string from retrieved documents
            context_parts = []
            sources = []
            
            for i, result in enumerate(context[:3]):  # Use top 3 results
                context_parts.append(f"Source {i+1}: {result.context_snippet}")
                sources.append({
                    "source": result.document.source,
                    "category": result.document.category,
                    "confidence": result.document.confidence,
                    "relevance": result.relevance_score,
                    "last_updated": result.document.last_updated
                })
            
            context_text = "\n\n".join(context_parts)
            
            # Create specialized prompt based on query type
            prompt = self._create_specialized_prompt(query, query_type, context_text)
            
            # Generate response using provided LLM function
            llm_response = llm_generate_func(prompt, context_text)
            
            # Calculate overall confidence
            avg_relevance = sum(r.relevance_score for r in context) / len(context)
            avg_source_confidence = sum(r.document.confidence for r in context) / len(context)
            overall_confidence = (avg_relevance + avg_source_confidence) / 2
            
            return RAGResponse(
                answer=llm_response,
                sources=sources,
                confidence=overall_confidence,
                query_type=query_type,
                emergency_detected=False,
                context_used=context_text
            )
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return RAGResponse(
                answer="I'm unable to provide a response at this time. Please consult a healthcare professional.",
                sources=[],
                confidence=0.1,
                query_type=query_type,
                emergency_detected=False,
                context_used=""
            )
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given regex patterns"""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _create_routing_decision(self, query_type: QueryType, query: str, confidence: float) -> RoutingDecision:
        """Create routing decision based on query type"""
        
        strategy_map = {
            QueryType.DRUG_INTERACTION: ("interaction_focused", 3, "interaction"),
            QueryType.SYMPTOM_CHECK: ("symptom_focused", 4, "symptom"),
            QueryType.MEDICATION_INFO: ("medication_focused", 3, "medication"),
            QueryType.CHRONIC_CARE: ("chronic_focused", 4, "chronic_condition"),
            QueryType.MENTAL_HEALTH: ("mental_health_focused", 3, "mental_health"),
            QueryType.DIAGNOSTIC: ("diagnostic_focused", 4, None),
            QueryType.GENERAL_HEALTH: ("hybrid_search", 5, None)
        }
        
        strategy, k, category = strategy_map.get(query_type, ("hybrid_search", 5, None))
        
        return RoutingDecision(
            query_type=query_type,
            confidence=confidence,
            retrieval_strategy=strategy,
            emergency_flag=False,
            recommended_k=k,
            category_filter=category
        )
    
    def _get_emergency_response(self) -> List[RetrievalResult]:
        """Return emergency response protocol"""
        # This would typically retrieve from a dedicated emergency protocol database
        from .vector_store import MedicalDocument, RetrievalResult
        
        emergency_doc = MedicalDocument(
            content="EMERGENCY: Seek immediate medical attention. Call emergency services.",
            source="Emergency Protocol",
            category="emergency",
            confidence=1.0,
            last_updated="2024-01-01",
            doc_id="emergency_protocol",
            metadata={"priority": "critical"}
        )
        
        return [RetrievalResult(
            document=emergency_doc,
            relevance_score=1.0,
            context_snippet="Emergency protocol activated"
        )]
    
    def _apply_type_specific_filtering(self, results: List[RetrievalResult], 
                                     query_type: QueryType) -> List[RetrievalResult]:
        """Apply query-type specific filtering and boost relevant results"""
        
        # Boost scores based on query type and document category alignment
        boost_map = {
            QueryType.DRUG_INTERACTION: {"interaction": 1.2, "medication": 1.1},
            QueryType.SYMPTOM_CHECK: {"symptom": 1.2, "chronic_condition": 1.1},
            QueryType.MEDICATION_INFO: {"medication": 1.2, "interaction": 1.1},
            QueryType.CHRONIC_CARE: {"chronic_condition": 1.2, "symptom": 1.1},
            QueryType.MENTAL_HEALTH: {"mental_health": 1.2}
        }
        
        boosts = boost_map.get(query_type, {})
        
        for result in results:
            category = result.document.category
            if category in boosts:
                result.relevance_score *= boosts[category]
                result.relevance_score = min(1.0, result.relevance_score)  # Cap at 1.0
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def _create_specialized_prompt(self, query: str, query_type: QueryType, context: str) -> str:
        """Create specialized prompts based on query type"""
        
        base_prompt = f"""Based on the following medical information sources, please provide a helpful and accurate response to the user's question.

User Question: {query}

Medical Context:
{context}

"""
        
        type_specific_instructions = {
            QueryType.DRUG_INTERACTION: "Focus on drug interactions, safety concerns, and contraindications. Always recommend consulting a pharmacist or doctor for drug combinations.",
            
            QueryType.SYMPTOM_CHECK: "Provide general information about symptoms and when to seek medical care. Never provide definitive diagnoses. Include warning signs that require immediate medical attention.",
            
            QueryType.MEDICATION_INFO: "Provide factual information about medications including uses, side effects, and precautions. Always remind the user to follow their doctor's instructions.",
            
            QueryType.CHRONIC_CARE: "Focus on evidence-based management strategies, lifestyle modifications, and monitoring recommendations. Emphasize the importance of working with healthcare providers.",
            
            QueryType.MENTAL_HEALTH: "Provide supportive information and coping strategies. Include crisis resources when appropriate. Never provide therapy or replace professional mental health care.",
            
            QueryType.DIAGNOSTIC: "Explain what tests measure and normal ranges when available. Emphasize that interpretation should be done by healthcare professionals.",
            
            QueryType.GENERAL_HEALTH: "Provide general health information while emphasizing the importance of personalized medical advice."
        }
        
        instruction = type_specific_instructions.get(query_type, "Provide helpful medical information.")
        
        return base_prompt + f"""
Instructions: {instruction}

Remember to:
1. Base your response on the provided context
2. Include a disclaimer about consulting healthcare professionals
3. Be clear and easy to understand
4. Cite sources when possible
5. Never provide definitive medical diagnoses

Response:"""

# Example usage
if __name__ == "__main__":
    # This would typically be called from the RASA actions
    from .vector_store import MedicalVectorStore
    
    # Initialize components
    vector_store = MedicalVectorStore()
    router = MedicalRAGRouter(vector_store)
    
    # Test query classification
    test_queries = [
        "Can I take ibuprofen with alcohol?",
        "I have chest pain and shortness of breath",
        "What are the side effects of metformin?",
        "How to manage diabetes?",
        "I'm feeling anxious and can't sleep"
    ]
    
    for query in test_queries:
        decision = router.classify_query(query)
        print(f"Query: {query}")
        print(f"Type: {decision.query_type.value}")
        print(f"Confidence: {decision.confidence}")
        print(f"Emergency: {decision.emergency_flag}")
        print("---")

