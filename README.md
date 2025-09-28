## LLM-and-RAG-Powered-MedBot

An advanced RASA-based medical chatbot with **Retrieval-Augmented Generation (RAG)** capabilities, providing intelligent symptom assessment, medication guidance, mental health support, and chronic disease management with **source attribution and safety validation**, supported by LLM  for handling ambiguous queries(out of the scope of the Knowledge Base) and generating dynamic responses, and RAG for solving the limitations of LLM.



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



## RASA Chatbot with LLM Integration: Architecture

<img width="1113" height="702" alt="Health-bot Integration Architecture" src="https://github.com/user-attachments/assets/4a885364-ae5a-438b-9fb3-b2c79f8985cc" />


## 🚀 New RAG Features

### ✅ **Enhanced Capabilities**
- **Intelligent Query Routing**: Automatically classifies queries and routes to appropriate retrieval strategies
- **Vector-Based Retrieval**: Semantic search through medical knowledge with relevance scoring
- **Emergency Detection**: Immediate identification and response to medical emergencies
- **Source Attribution**: Every response includes verifiable medical sources with confidence scores
- **Multi-Layer Safety Validation**: Advanced safety checking with medical disclaimer injection
- **Confidence Scoring**: Quantified trust levels for all medical information provided


## RAG Supported Architecture(WIP)

RAG is built with the following capabilities to solve the Key LLM limitations:

- ✅ **Eliminates Hallucinations**: Grounds responses in verified medical sources
- ✅ **Up-to-Date Information**: Dynamic retrieval from latest medical databases  
- ✅ **Traceability**: Every response references its source (CDC, DrugBank, internal KB)
- ✅ **Customization**: Blend general medical knowledge with proprietary guidelines
<img width="6323" height="702" alt="RAG Integration Architecture" src="https://github.com/user-attachments/assets/c90d5e5c-6377-4f83-96ae-440a188794d7" />


## Final Workflow:
<img width="1532" height="1245" alt="LLM-RAG-Final Workflow" src="https://github.com/user-attachments/assets/a22545c4-1798-473c-8da4-67a2dbc7b50a" />


## Key Workflows:

1. Precision First:
    Critical queries (e.g., drug interactions) are answered directly from the KB.

    Example:
    In `actions.py`:  
    if user_query == "aspirin and ibuprofen interaction":  
        return KB["aspirin"]["interactions"]["ibuprofen"]

2. RAG Retrieval Process
   
4. LLM Augmentation with RAG
    For complex or open-ended questions (e.g., "How to manage diabetes?"), LLM elaborates using KB data.

    Example prompt:
    "Based on [KB_Diabetes_Guide], list 3 diet tips for diabetes. Use simple language."  

3. Enhanced Validation Layer:
    Cross-check LLM outputs against the KB to filter out hallucinations.
    
    Example:
    
    if llm_response not in KB["allowed_advice"]:  
        return "Consult a doctor."


## Example Workflows:

Case 1: Medication Interaction (Structured + RAG)

```python
# User: "Can I take aspirin with warfarin?"
# Step 1: Structured KB lookup
kb_result = MEDICAL_KB["interactions"].get(("aspirin", "warfarin"))

# Step 2: If not in KB, RAG retrieval
if not kb_result:
    context = retrieve_medical_context(
        "aspirin warfarin interaction", 
        {"medication": ["aspirin", "warfarin"]}
    )
    
# Step 3: LLM synthesis
response = generate_rag_response(
    "aspirin and warfarin interaction", 
    context
)

# Response: "According to DrugBank: Aspirin may increase the anticoagulant effect of warfarin, increasing bleeding risk. Sources: DrugBank 2024, PubMed Study 2023"
```

Case 2: Emerging Health Topic (RAG-Primary)

```python
# User: "Latest treatment for long COVID?"
# Step 1: RAG retrieval from latest sources
context = retrieve_medical_context(
    "long COVID treatment guidelines 2024",
    {"condition": "long COVID", "category": "treatment"}
)

# Step 2: LLM synthesis with recency filter
response = generate_rag_response(
    "current long COVID treatment protocols",
    filter_by_recency(context, max_days=180)  # Last 6 months
)

# Response: "According to WHO 2024 guidelines: Graduated exercise therapy and cognitive behavioral therapy show efficacy. Sources: WHO Guidelines 2024, NIH Bulletin 2025"
```

Case 3: Multi-faceted Query (Hybrid Approach)
```python
# User: "Diabetes management with kidney complications"
# Step 1: Structured KB for basics
kb_diabetes = MEDICAL_KB["chronic_conditions"]["diabetes"]["management"]
kb_kidney = MEDICAL_KB["complications"]["kidney"]

# Step 2: RAG for a specific combination
context = retrieve_medical_context(
    "diabetes nephropathy management",
    {"condition": ["diabetes", "kidney disease"]}
)

# Step 3: Combined response
response = combine_kb_and_rag(kb_diabetes, kb_kidney, context)
```



## 🔧 **Quick Setup**

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Pinecone API Key
```bash
export PINECONE_API_KEY="your-pinecone-api-key-here"
```

### 3. Initialize RAG System
```bash
python setup_rag.py
```

### 4. Test RAG Components
```bash
python test_rag_system.py
```

### 5. Start the Chatbot
```bash
# Terminal 1: Start RASA Action Server
rasa run actions

# Terminal 2: Start Chatbot Interface  
rasa shell
```
