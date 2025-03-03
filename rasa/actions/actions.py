# Core Logic
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from typing import Any, Text, Dict, List
import json

# Load knowledge base (simplified example)
with open("knowledge-base/medical_kb.json") as f:
    MEDICAL_KB = json.load(f)

class ActionCheckKnowledgeBase(Action):
    def name(self) -> Text:
        return "action_check_knowledge_base"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Extract entities from the latest message
        entities = {e["entity"]: e["value"] for e in tracker.latest_message.get("entities", [])}
        intent = tracker.latest_message.get("intent", {}).get("name")

        response = None

        # Symptom + Duration Check (matches stories.yml "symptom check path")
        if intent == "symptom_check":
            symptom = entities.get("symptom")
            duration = entities.get("duration")
            if symptom and duration:
                response = MEDICAL_KB["symptoms"].get(symptom, {}).get("urgency", {}).get(duration)
                if not response:
                    response = self.get_llm_response(f"Explain {symptom} lasting {duration}")

        # Medication Interaction Check (matches "medication interaction check" story)
        elif intent == "ask_medication":
            meds = [e for e in entities.values() if e in MEDICAL_KB["medications"]]
            if len(meds) >= 2:
                interaction = MEDICAL_KB["interactions"].get(tuple(sorted(meds)))
                response = interaction or self.get_llm_response(f"Interaction between {meds[0]} and {meds[1]}")

        # Chronic Care (matches "diabetes diet advice" story)
        elif intent == "chronic_care":
            condition = entities.get("condition")
            response = MEDICAL_KB["chronic_conditions"].get(condition, {}).get("advice")

        # Fallback if no KB match
        if not response:
            response = self.get_llm_response(tracker.latest_message.get('text'))

        dispatcher.utter_message(text=response)
        return [SlotSet(key, value) for key, value in entities.items()]  # Save entities to slots

    def get_llm_response(self, query: Text) -> Text:
        # Simplified - replace with actual LLM API call
        return "I recommend consulting a doctor for personalized advice."

class ActionLLMFallback(Action):
    """Handles mental_health intents (matches 'anxiety support flow' story)"""
    def name(self) -> Text:
        return "action_llm_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        prompt = f"""You are a mental health assistant. Respond to:
        {tracker.latest_message.get('text')}
        """
        response = self.get_llm_response(prompt)
        dispatcher.utter_message(text=response)
        return []

    def get_llm_response(self, query: Text) -> Text:
        # Add OpenAI/other LLM integration here
        return "Consider mindfulness exercises. Breathe deeply for 5 minutes."
