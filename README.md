# RASA-Health-bot

Medical Bot which can act as an doctor assistant - Symptom Assessment & Triage, Medical information , mental health support, and Chronic Disease Management. 


## Integration Architecture:

User Query
    │
    ↓
[RASA NLU] ───→ [Intent Detection & Entity Extraction]
    │
    └→ [Knowledge Base] → Check KB → If Found → Return KB Answer ────────────────────────────────────→↓
                                   │                                                                  ├──→ Combine KB fact + LLM explanation ─→ Return Response to User
                                   └→ If Not Found → [LLM Integration] → Generates Dynamic Response──→↑
                                                    └─→ Safety Check


## Key Workflows:

1. Precision First:
Critical queries (e.g., drug interactions) are answered directly from the KB.

Example:
# In `actions.py`:  
if user_query == "aspirin and ibuprofen interaction":  
    return KB["aspirin"]["interactions"]["ibuprofen"]  


2. LLM Augmentation:
For complex or open-ended questions (e.g., "How to manage diabetes?"), the LLM elaborates using KB data.

Example prompt:
"Based on [KB_Diabetes_Guide], list 3 diet tips for diabetes. Use simple language."  

3. Validation Layer:
Cross-check LLM outputs against the KB to filter out hallucinations.

Example:

if llm_response not in KB["allowed_advice"]:  
    return "Consult a doctor."  
