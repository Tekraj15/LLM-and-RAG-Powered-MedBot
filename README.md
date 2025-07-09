# LLM-integrated-Med-ChatBot

A RASA chatbot that can act as a doctor's assistant, providing symptom assessment and triage, Medical information, mental health support, and Chronic Disease Management, etc., with LLM integration for handling ambiguous queries(out of the scope of Knowledge Base) and generating dynamic responses.. 


## Integration Architecture:

<img width="1113" alt="Health-bot Integration Architecture" src="https://github.com/user-attachments/assets/4a885364-ae5a-438b-9fb3-b2c79f8985cc" />



## Key Workflows:

1. Precision First:
    Critical queries (e.g., drug interactions) are answered directly from the KB.

    Example:
    In `actions.py`:  
    if user_query == "aspirin and ibuprofen interaction":  
        return KB["aspirin"]["interactions"]["ibuprofen"]


2. LLM Augmentation:
    For complex or open-ended questions (e.g., "How to manage diabetes?"), LLM elaborates using KB data.

    Example prompt:
    "Based on [KB_Diabetes_Guide], list 3 diet tips for diabetes. Use simple language."  

3. Validation Layer:
    Cross-check LLM outputs against the KB to filter out hallucinations.
    
    Example:
    
    if llm_response not in KB["allowed_advice"]:  
        return "Consult a doctor."
