from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from typing import Any, Text, Dict, List
import json
import requests  # For DeepSeek API calls
import os

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DeepSeek-API-Key", "my-api-key")

# --- Knowledge Base ---
with open("../Knowledge-base/med_knowledge.json") as f:
    MEDICAL_KB = json.load(f)

# --- Helper Functions ---
def get_llm_response(prompt: str, medical_context: str = "") -> str:
    """Call DeepSeek-R1 API with proper configuration"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    system_message = f"""You are a medical assistant. Follow these rules:
    1. Base answers on: {medical_context}
    2. Never invent facts. Say "I don't know" if unsure.
    3. Prioritize safety and clarity."""
    
    payload = {
        "model": "deepseek-reasoner",  # Pointing to DeepSeek-R1-0528 - the latest model by DeepSeek with improved benchmarks and reduced hallucinations
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,  # Lower for medical accuracy
        "max_tokens": 200
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",  # Verified endpoint
            headers=headers,
            json=payload,
            timeout=15
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return "I recommend consulting a healthcare professional."
    except Exception as e:
        return "I'm unable to respond. Please consult a doctor."

# --- Action Classes ---
class ActionCheckSymptoms(Action):
    """Handles symptom_check intent (matches stories.yml)"""
    def name(self) -> Text:
        return "action_check_symptoms"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            
        entities = {e["entity"]: e["value"] for e in tracker.latest_message.get("entities", [])}
        symptom = entities.get("symptom", "").lower()
        duration = entities.get("duration", "")
        
        # 1. Check Knowledge Base
        kb_response = None
        if symptom in MEDICAL_KB["symptoms"]:
            kb_data = MEDICAL_KB["symptoms"][symptom]
            if duration and "urgency" in kb_data:
                kb_response = kb_data["urgency"].get(duration)
        
        # 2. If not found in KB, use LLM to explain
        if kb_response:
            prompt = f"Explain this medical advice in simple terms: {kb_response}"
            llm_response = get_llm_response(prompt, str(kb_data))
            dispatcher.utter_message(text=llm_response)
        else:
            prompt = f"Assess {symptom} lasting {duration} with possible causes"
            llm_response = get_llm_response(prompt, json.dumps(MEDICAL_KB["symptoms"]))
            dispatcher.utter_message(text=llm_response)
        
        return [SlotSet("symptom", symptom), SlotSet("duration", duration)]

class ActionCheckKnowledgeBase(Action):
    def name(self) -> Text:
        return "action_check_knowledge_base"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        intent = tracker.latest_message["intent"]["name"]
        user_query = tracker.latest_message["text"]
        
        # Medication/Interaction Check
        if intent == "ask_medication":
            meds = [e["value"] for e in tracker.latest_message["entities"] 
                    if e["entity"] == "medication"]
            if len(meds) >= 2:
                interaction = MEDICAL_KB["interactions"].get(",".join(sorted(meds)))
                if interaction:
                    prompt = f"Explain drug interaction: {interaction}"
                    response = get_llm_response(prompt, interaction)
                else:
                    prompt = f"Potential interaction between {meds[0]} and {meds[1]}"
                    response = get_llm_response(prompt, json.dumps(MEDICAL_KB["medications"]))
                dispatcher.utter_message(text=response)
                return []

        # For other intents, use the default implementation
        dispatcher.utter_message(text=get_llm_response(user_query))
        return []

class ActionLLMFallback(Action):
    def name(self) -> Text:
        return "action_llm_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_query = tracker.latest_message["text"]
        response = get_llm_response(
            prompt=user_query,
            medical_context=json.dumps(MEDICAL_KB["mental_health"])
        )
        dispatcher.utter_message(text=response)
        return []
