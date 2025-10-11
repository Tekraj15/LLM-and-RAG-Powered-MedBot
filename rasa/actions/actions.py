from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from typing import Any, Text, Dict, List
import json
import requests  # For DeepSeek API calls
import os
import sys
import logging

# Add path for RAG modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# RAG imports
from rag_core import MedicalVectorStore, MedicalRAGRouter, QueryType
from safety_layer.validator import MedicalResponseValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DeepSeek-API-Key", "my-api-key")

# --- Knowledge Base ---
try:
    with open("../Knowledge-base/med_knowledge.json") as f:
        MEDICAL_KB = json.load(f)
except FileNotFoundError:
    # Fallback path for different execution contexts
    kb_path = os.path.join(os.path.dirname(__file__), "../../Knowledge-base/med_knowledge.json")
    with open(kb_path) as f:
        MEDICAL_KB = json.load(f)

# --- Initialize RAG Components ---
try:
    # Initialize Pinecone vector store and populate with knowledge base
    vector_store = MedicalVectorStore(index_name="medical-knowledge-chatbot")
    
    # Check if vector store is empty and populate it
    stats = vector_store.get_collection_stats()
    if stats.get("total_documents", 0) == 0:
        kb_path = "../Knowledge-base/med_knowledge.json"
        if not os.path.exists(kb_path):
            kb_path = os.path.join(os.path.dirname(__file__), "../../Knowledge-base/med_knowledge.json")
        vector_store.add_knowledge_base(kb_path)
        logger.info("Populated Pinecone vector store with knowledge base")
    
    # Initialize RAG router and validator
    rag_router = MedicalRAGRouter(vector_store)
    response_validator = MedicalResponseValidator()
    
    logger.info("RAG components initialized successfully with Pinecone")
    
except Exception as e:
    logger.error(f"Error initializing RAG components: {str(e)}")
    # Fallback to None - will use traditional KB approach
    vector_store = None
    rag_router = None
    response_validator = MedicalResponseValidator()

# --- Enhanced Helper Functions ---
def get_llm_response(prompt: str, medical_context: str = "") -> str:
    """Enhanced LLM call with RAG context and safety validation"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    system_message = f"""You are a medical assistant with access to verified medical sources. Follow these rules:
    1. Base ALL answers on the provided medical context: {medical_context}
    2. NEVER invent medical facts. Say "I don't have specific information" if unsure.
    3. Prioritize safety and clarity above all.
    4. Include source attribution when possible.
    5. Always recommend consulting healthcare professionals for personalized advice.
    6. Use clear, non-technical language that patients can understand."""
    
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # Even lower for medical accuracy with RAG
        "max_tokens": 300
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return "I recommend consulting a healthcare professional for accurate information."
    except Exception as e:
        logger.error(f"LLM API error: {str(e)}")
        return "I'm unable to respond at this time. Please consult a doctor."

def get_rag_enhanced_response(query: str, intent: str = None) -> Dict[str, Any]:
    """Get RAG-enhanced response with full pipeline"""
    try:
        # Use RAG pipeline if available
        if rag_router and vector_store:
            # Step 1: Classify query and determine routing
            routing_decision = rag_router.classify_query(query, intent)
            
            # Step 2: Handle emergency queries immediately
            if routing_decision.emergency_flag:
                emergency_response = "🚨 MEDICAL EMERGENCY DETECTED 🚨\n\nIf you're experiencing a medical emergency, please:\n• Call emergency services immediately (911/999/112)\n• Go to the nearest emergency room\n• Do not delay seeking professional medical help"
                
                validation_result = response_validator.validate_response(
                    emergency_response, 
                    [{"source": "Emergency Protocol", "confidence": 1.0}],
                    "emergency"
                )
                
                return {
                    "response": validation_result.modified_response,
                    "sources": [{"source": "Emergency Protocol", "confidence": 1.0}],
                    "query_type": routing_decision.query_type.value,
                    "confidence": validation_result.confidence_score,
                    "emergency": True
                }
            
            # Step 3: Retrieve relevant context
            context_results = rag_router.retrieve_context(query, routing_decision)
            
            # Step 4: Generate response using LLM with context
            rag_response = rag_router.generate_rag_response(
                query, context_results, routing_decision.query_type, get_llm_response
            )
            
            # Step 5: Validate response for safety
            validation_result = response_validator.validate_response(
                rag_response.answer,
                rag_response.sources,
                routing_decision.query_type.value
            )
            
            return {
                "response": validation_result.modified_response,
                "sources": rag_response.sources,
                "query_type": routing_decision.query_type.value,
                "confidence": validation_result.confidence_score,
                "emergency": routing_decision.emergency_flag,
                "warnings": validation_result.warnings
            }
        
        else:
            # Fallback to traditional approach
            llm_response = get_llm_response(query, json.dumps(MEDICAL_KB))
            validation_result = response_validator.validate_response(llm_response)
            
            return {
                "response": validation_result.modified_response,
                "sources": [{"source": "internal_kb", "confidence": 0.8}],
                "query_type": "general",
                "confidence": validation_result.confidence_score,
                "emergency": False
            }
            
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        fallback_response = "I'm unable to provide a specific response right now. For your safety, please consult a qualified healthcare professional who can properly assess your situation."
        
        validation_result = response_validator.validate_response(fallback_response)
        
        return {
            "response": validation_result.modified_response,
            "sources": [],
            "query_type": "error",
            "confidence": 0.1,
            "emergency": False
        }

# --- Enhanced RAG Action Classes ---
class ActionCheckSymptoms(Action):
    """RAG-enhanced symptom checking with intelligent routing"""
    def name(self) -> Text:
        return "action_check_symptoms"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            
        entities = {e["entity"]: e["value"] for e in tracker.latest_message.get("entities", [])}
        symptom = entities.get("symptom", "").lower()
        duration = entities.get("duration", "")
        user_query = tracker.latest_message["text"]
        
        # Build comprehensive query
        query_parts = [user_query]
        if symptom:
            query_parts.append(f"symptom: {symptom}")
        if duration:
            query_parts.append(f"duration: {duration}")
        
        full_query = " ".join(query_parts)
        
        # Get RAG-enhanced response
        rag_result = get_rag_enhanced_response(full_query, "symptom_check")
        
        # Format response with source attribution
        response_text = rag_result["response"]
        
        # Add source information if available
        if rag_result["sources"]:
            source_info = "\n\n📚 **Sources:** " + ", ".join([
                f"{src['source']} (confidence: {src.get('confidence', 0.8):.1f})"
                for src in rag_result["sources"][:3]
            ])
            response_text += source_info
        
        # Add confidence indicator
        confidence = rag_result.get("confidence", 0.8)
        if confidence < 0.7:
            response_text += f"\n\n🔍 **Note:** Response confidence is moderate ({confidence:.1f}). Consider consulting a healthcare professional for more specific guidance."
        
        dispatcher.utter_message(text=response_text)
        
        # Set slots with enhanced metadata
        return [
            SlotSet("symptom", symptom),
            SlotSet("duration", duration),
            SlotSet("query_type", rag_result["query_type"]),
            SlotSet("confidence_score", confidence)
        ]

class ActionCheckKnowledgeBase(Action):
    """Enhanced knowledge base check with RAG integration"""
    def name(self) -> Text:
        return "action_check_knowledge_base"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        intent = tracker.latest_message["intent"]["name"]
        user_query = tracker.latest_message["text"]
        
        # Get RAG-enhanced response
        rag_result = get_rag_enhanced_response(user_query, intent)
        
        # Handle emergency situations
        if rag_result.get("emergency"):
            dispatcher.utter_message(text=rag_result["response"])
            return [SlotSet("emergency_detected", True)]
        
        # Format response with enhanced information
        response_text = rag_result["response"]
        
        # Add metadata for medication queries
        if intent == "ask_medication":
            meds = [e["value"] for e in tracker.latest_message["entities"] 
                    if e["entity"] == "medication"]
            if len(meds) >= 2:
                response_text += f"\n\n💊 **Medications discussed:** {', '.join(meds)}"
        
        # Add source attribution
        if rag_result["sources"]:
            source_info = "\n\n📚 **Information sources:** " + ", ".join([
                f"{src['source']}" for src in rag_result["sources"][:2]
            ])
            response_text += source_info
        
        # Add confidence and warnings
        confidence = rag_result.get("confidence", 0.8)
        if confidence < 0.7:
            response_text += f"\n\n⚠️ **Please note:** Information confidence is moderate. Consult a healthcare provider for personalized advice."
        
        if rag_result.get("warnings"):
            response_text += f"\n\n🔔 **Important:** {'; '.join(rag_result['warnings'])}"
        
        dispatcher.utter_message(text=response_text)
        
        return [
            SlotSet("query_type", rag_result["query_type"]),
            SlotSet("confidence_score", confidence)
        ]

class ActionLLMFallback(Action):
    """RAG-enhanced fallback action for general queries"""
    def name(self) -> Text:
        return "action_llm_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_query = tracker.latest_message["text"]
        
        # Get RAG-enhanced response for general health queries
        rag_result = get_rag_enhanced_response(user_query, "general_health")
        
        # Format response
        response_text = rag_result["response"]
        
        # Add helpful context for general queries
        if rag_result["query_type"] in ["general_health", "mental_health"]:
            response_text += "\n\n💡 **Remember:** This information is for general guidance only. For personalized advice, always consult with a qualified healthcare professional."
        
        # Add source attribution for transparency
        if rag_result["sources"]:
            source_info = "\n\n📚 **Based on:** " + ", ".join([
                f"{src['source']}" for src in rag_result["sources"][:2]
            ])
            response_text += source_info
        
        dispatcher.utter_message(text=response_text)
        
        return [
            SlotSet("query_type", rag_result["query_type"]),
            SlotSet("confidence_score", rag_result.get("confidence", 0.8))
        ]

# New RAG-specific action for advanced queries
class ActionRAGQuery(Action):
    """Advanced RAG action for complex medical queries"""
    def name(self) -> Text:
        return "action_rag_query"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_query = tracker.latest_message["text"]
        
        # Get comprehensive RAG response
        rag_result = get_rag_enhanced_response(user_query)
        
        # Handle emergency situations with priority
        if rag_result.get("emergency"):
            dispatcher.utter_message(text=rag_result["response"])
            return [
                SlotSet("emergency_detected", True),
                SlotSet("query_type", "emergency")
            ]
        
        # Build comprehensive response
        response_parts = [rag_result["response"]]
        
        # Add detailed source information
        if rag_result["sources"]:
            source_details = []
            for src in rag_result["sources"][:3]:
                confidence = src.get("confidence", 0.8)
                last_updated = src.get("last_updated", "")
                source_line = f"• {src['source']} (confidence: {confidence:.1f})"
                if last_updated:
                    source_line += f" - Updated: {last_updated}"
                source_details.append(source_line)
            
            response_parts.append(f"\n\n📚 **Detailed Sources:**\n" + "\n".join(source_details))
        
        # Add confidence and reliability information
        confidence = rag_result.get("confidence", 0.8)
        if confidence >= 0.8:
            response_parts.append(f"\n\n✅ **High confidence response** ({confidence:.2f}) - Information is well-sourced and reliable.")
        elif confidence >= 0.6:
            response_parts.append(f"\n\n⚠️ **Moderate confidence** ({confidence:.2f}) - Consider seeking additional professional guidance.")
        else:
            response_parts.append(f"\n\n🔍 **Lower confidence** ({confidence:.2f}) - Recommend consulting a healthcare professional for verified information.")
        
        # Add warnings if present
        if rag_result.get("warnings"):
            warnings_text = "\n\n🚨 **Important Considerations:**\n• " + "\n• ".join(rag_result["warnings"])
            response_parts.append(warnings_text)
        
        final_response = "".join(response_parts)
        dispatcher.utter_message(text=final_response)
        
        return [
            SlotSet("query_type", rag_result["query_type"]),
            SlotSet("confidence_score", confidence),
            SlotSet("emergency_detected", rag_result.get("emergency", False)),
            SlotSet("sources_count", len(rag_result.get("sources", [])))
        ]
