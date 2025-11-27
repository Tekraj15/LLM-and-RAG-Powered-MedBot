from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from typing import Any, Text, Dict, List
import os
import sys
import json
import logging

# Add path for RAG modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# RAG imports
from rag.retrieval.retriever import MedicalRetriever
from rag.augmentation.augmenter import MedicalAugmenter
from rag.generation.generator import MedicalGenerator
from safety_layer.validator import MedicalResponseValidator, ValidationResult
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "my-api-key")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# --- Knowledge Base ---
kb_path = os.path.join(os.path.dirname(__file__), "../../Knowledge-base/med_knowledge.json")
try:
    with open(kb_path, "r", encoding="utf-8") as f:
        MEDICAL_KB = json.load(f)
    logger.info("Loaded medical knowledge base successfully.")
except FileNotFoundError as e:
    logger.error(f"Knowledge base file not found: {e}")
    MEDICAL_KB = {}

# --- Initialize RAG Components ---
retriever = None
augmenter = None
generator = None
validator = MedicalResponseValidator()

try:
    if PINECONE_API_KEY and OPENAI_API_KEY:
        logger.info("Attempting to connect to RAG pipeline...")
        # Only initialize if keys exist
        retriever = MedicalRetriever(index_name="medbot-rag")
        augmenter = MedicalAugmenter()
        generator = MedicalGenerator()
        logger.info("RAG Pipeline initialized successfully.")
    else:
        logger.warning("!! Missing API Keys. RAG features will be disabled.")
except Exception as e:
    logger.error(f"!! RAG Initialization Failed: {e}")
    logger.error("The bot will function in 'Fallback Mode' (KB only).")
    # We swallow the error so the Action Server doesn't crash!


# --- Helper Functions ---
def get_kb_response(query: str) -> Dict:
    """Fetch response from structured KB (Symptoms & Interactions)."""
    query_lower = query.lower()
    
    # 1. Check Symptoms
    symptoms = MEDICAL_KB.get("symptoms", {})
    for name, data in symptoms.items():
        # Match name or description
        if name.replace("_", " ") in query_lower or data.get("description", "").lower() in query_lower:
            response_text = (
                f"**Symptom:** {name.replace('_', ' ').title()}\n"
                f"**Description:** {data.get('description')}\n"
                f"**Common Causes:** {', '.join(data.get('common_causes', []))}\n"
                f"**Urgency:** {list(data.get('urgency', {}).values())[0]}"
            )
            return {"response": response_text, "source": "internal_kb_symptoms", "confidence": 0.95}

    # 2. Check Interactions
    interactions = MEDICAL_KB.get("interactions", {})
    for key, warning in interactions.items():
        # Key format: "drug1,drug2"
        drugs = key.split(",")
        if all(drug.strip().lower() in query_lower for drug in drugs):
             return {"response": f"**Interaction Warning:** {warning}", "source": "internal_kb_interactions", "confidence": 0.95}

    return {"response": "", "source": "internal_kb", "confidence": 0.0}

def combine_responses(kb_result: Dict, rag_result: Dict) -> Dict:
    """Merge KB and RAG responses for hybrid queries."""
    if not kb_result["response"] or not rag_result.get("response"):
        return rag_result if rag_result.get("response") else kb_result
    prompt = f"Integrate KB: {kb_result['response']} with RAG: {rag_result['response']}. Provide a unified response."
    merged = generator.llm.invoke(prompt) if generator else {"content": kb_result["response"]}
    return {
        "response": merged.content if generator else merged,
        "metadata": {**rag_result.get("metadata", {}), **{"kb_source": kb_result["source"]}},
        "confidence": min(rag_result.get("confidence", 0.5), kb_result["confidence"])
    }

# --- Enhanced RAG Action Classes ---
class ActionCheckSymptoms(Action):
    def name(self) -> Text:
        return "action_check_symptoms"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = {e["entity"]: e["value"] for e in tracker.latest_message.get("entities", [])}
        symptom = entities.get("symptom", "").lower()
        duration = entities.get("duration", "")
        user_query = tracker.latest_message["text"]
        
        query_parts = [user_query]
        if symptom: query_parts.append(f"symptom: {symptom}")
        if duration: query_parts.append(f"duration: {duration}")
        full_query = " ".join(query_parts)

        validation_result: ValidationResult
        if retriever and augmenter and generator:
            docs = retriever.retrieve(full_query, strategy="mmr", metadata_filter={"category": "symptom", "last_updated": {"$gte": "2023-01-01"}})
            augmented = augmenter.augment(full_query, docs)
            rag_result = generator.generate(augmented)
            validation_result = validator.validate_response(rag_result["response"], rag_result["metadata"]["sources"], "symptom")
        else:
            validation_result = validator.validate_response("No RAG data available. Consult a doctor.")

        response_text = validation_result.modified_response
        if validation_result.confidence_score < 0.7:
            response_text += "\n\nc**Note:** Moderate confidence. Consult a doctor."
        if validation_result.sources_verified and validation_result.sources:
            source_info = "\n\n **Sources:** " + ", ".join([f"{s['source']} (conf: {s.get('confidence', 0.8):.1f})" for s in validation_result.sources[:3]])
            response_text += source_info
        if validation_result.warnings:
            response_text += f"\n\n!! **Warnings !!:** {'; '.join(validation_result.warnings)}"

        dispatcher.utter_message(text=response_text)
        return [
            SlotSet("symptom", symptom),
            SlotSet("duration", duration),
            SlotSet("query_type", "symptom_check"),
            SlotSet("confidence_score", validation_result.confidence_score)
        ]

class ActionCheckKnowledgeBase(Action):
    def name(self) -> Text:
        return "action_check_knowledge_base"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        intent = tracker.latest_message["intent"]["name"]
        user_query = tracker.latest_message["text"]
        
        kb_result = get_kb_response(user_query)
        # RAG Retrieval with Error Handling
        rag_result = {"response": "", "metadata": {"sources": []}, "confidence": 0.5}
        try:
            if retriever and augmenter and generator:
                docs = retriever.retrieve(user_query, strategy="rerank")
                augmented = augmenter.augment(user_query, docs)
                rag_result = generator.generate(augmented)
        except Exception as e:
            logger.error(f"RAG Pipeline Failed: {e}")
        
        # Combine Results
        if kb_result["response"] and rag_result.get("response"):
            combined = combine_responses(kb_result, rag_result)
        else:
            combined = kb_result if kb_result["response"] else rag_result

        validation_result: ValidationResult = validator.validate_response(combined.get("response", "No data"), combined.get("metadata", {}).get("sources", []), intent)

        response_text = validation_result.modified_response
        if intent == "ask_medication" and tracker.latest_message.get("entities"):
            meds = [e["value"] for e in tracker.latest_message["entities"] if e["entity"] == "medication"]
            if meds: response_text += f"\n\n**Medications:** {', '.join(meds)}"

        if validation_result.sources_verified and validation_result.sources:
            source_info = "\n\n**Sources:** " + ", ".join([f"{s['source']}" for s in validation_result.sources[:2]])
            response_text += source_info

        if validation_result.confidence_score < 0.7:
            response_text += "\n\n**Note!:** Moderate confidence. Consult a healthcare provider."
        if validation_result.warnings:
            response_text += f"\n\n!! **Warnings:** {'; '.join(validation_result.warnings)}"

        dispatcher.utter_message(text=response_text)
        return [
            SlotSet("query_type", intent),
            SlotSet("confidence_score", validation_result.confidence_score)
        ]

class ActionLLMFallback(Action):
    def name(self) -> Text:
        return "action_llm_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_query = tracker.latest_message["text"]
        validation_result: ValidationResult
        if retriever and augmenter and generator:
            docs = retriever.retrieve(user_query, strategy="mmr")
            augmented = augmenter.augment(user_query, docs)
            rag_result = generator.generate(augmented)
            validation_result = validator.validate_response(rag_result["response"], rag_result["metadata"]["sources"], "general_health")
        else:
            validation_result = validator.validate_response("No data available. Consult a doctor.")

        response_text = validation_result.modified_response
        response_text += "\n\n**Note:** General guidance only. Consult a healthcare professional."
        if validation_result.sources_verified and validation_result.sources:
            source_info = "\n\n**Based on:** " + ", ".join([f"{s['source']}" for s in validation_result.sources[:2]])
            response_text += source_info
        if validation_result.warnings:
            response_text += f"\n\n!! **Warnings !!:** {'; '.join(validation_result.warnings)}"

        dispatcher.utter_message(text=response_text)
        return [
            SlotSet("query_type", "general_health"),
            SlotSet("confidence_score", validation_result.confidence_score)
        ]

class ActionRAGQuery(Action):
    def name(self) -> Text:
        return "action_rag_query"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_query = tracker.latest_message["text"]
        validation_result: ValidationResult
        if retriever and augmenter and generator:
            docs = retriever.retrieve(user_query, strategy="mmr", metadata_filter={"last_updated": {"$gte": "2024-01-01"}})
            rag_result = generator.generate(augmenter.augment(user_query, docs))
            validation_result = validator.validate_response(rag_result["response"], rag_result["metadata"]["sources"], "complex")
        else:
            validation_result = validator.validate_response("No RAG data available. Consult a doctor.", query_type="complex")

        response_parts = [validation_result.modified_response]
        if validation_result.sources_verified and validation_result.sources:
            source_details = [f"• {s['source']} (conf: {s.get('confidence', 0.8):.1f})" + (f" - Updated: {s.get('last_updated', '')}" if s.get('last_updated') else "") for s in validation_result.sources[:3]]
            response_parts.append(f"\n\n📚 **Detailed Sources:**\n" + "\n".join(source_details))

        confidence = validation_result.confidence_score
        if confidence >= 0.8:
            response_parts.append(f"\n\n**High confidence** ({confidence:.2f}) - Reliable.")
        elif confidence >= 0.6:
            response_parts.append(f"\n\n**Moderate confidence** ({confidence:.2f}) - Seek guidance.")
        else:
            response_parts.append(f"\n\n**Lower confidence** ({confidence:.2f}) - Consult a doctor.")

        if validation_result.warnings:
            response_parts.append(f"\n\n!! **Warnings !!:** {'; '.join(validation_result.warnings)}")

        dispatcher.utter_message(text="".join(response_parts))
        return [
            SlotSet("query_type", "complex"),
            SlotSet("confidence_score", confidence),
            SlotSet("emergency_detected", False),
            SlotSet("sources_count", len(validation_result.sources) if validation_result.sources_verified else 0)
        ]

class ActionEmergencyResponse(Action):
    def name(self) -> Text:
        return "action_emergency_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        emergency_response = "!! MEDICAL EMERGENCY DETECTED !!\n\nIf you're experiencing a medical emergency, please:\n• Call emergency services immediately (911/999/112)\n• Go to the nearest emergency room\n• Do not delay seeking professional medical help"
        validation_result: ValidationResult = validator.validate_response(emergency_response, [{"source": "Emergency Protocol", "confidence": 1.0}], "emergency")
        dispatcher.utter_message(text=validation_result.modified_response)
        return [SlotSet("emergency_detected", True), SlotSet("query_type", "emergency")]