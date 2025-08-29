## LLM-and-RAG-Powered-MedBot

A RASA chatbot that can act as a doctor's assistant, providing symptom assessment and triage, Medical information, mental health support, and Chronic Disease Management, etc., with integration of LLM  for handling ambiguous queries(out of the scope of the Knowledge Base) and generating dynamic responses, and RAG to solve the limitations of LLM.

## Data Sources for RAG
1. Structured Medical Knowledge
- DrugBank API: Medication interactions & side effects
- CDC Guidelines: Prevention & treatment protocols
- PubMed Abstracts: Latest research findings
- WHO Disease Guidelines: International standards

2. Unstructured Data Processing
- PDF Parsing and Extraction

3. Real-time API Integration with Medical Knowledge Bases
   
MEDICAL_APIS = {
    "drug_interactions": "https://api.drugbank.com/v1/interactions",
    "symptom_checker": "https://api.infermedica.com/v3/diagnosis",
    "clinical_guidelines": "https://clinicaltrials.gov/api/v2/studies"
}

## Integration Architecture:

<img width="1113" alt="Health-bot Integration Architecture" src="https://github.com/user-attachments/assets/4a885364-ae5a-438b-9fb3-b2c79f8985cc" />

## RAG Supported Architecture(WIP)

RAG is being built with the following capabilities to solve the Key LLM limitations:

- Eliminates Hallucinations: Grounds responses in verified medical sources.

- Up-to-Date Information: Dynamic retrieval from the latest medical databases.

- Traceability: Every response can reference its source (e.g., CDC, DrugBank).

- Customization: Blend general medical knowledge with your proprietary guidelines.
<img width="6323" height="522" alt="RAG Integration Architecture" src="https://github.com/user-attachments/assets/c90d5e5c-6377-4f83-96ae-440a188794d7" />


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
