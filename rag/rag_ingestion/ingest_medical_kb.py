import json
from pathlib import Path
from .document_schema import MedicalDocument

def load_medical_kb(json_path: str):
    with open(json_path, 'r') as f:
        kb = json.load(f)
    return kb

def preprocess_documents(kb_data):
    # Example: convert KB entries to MedicalDocument objects
    documents = []
    for entry in kb_data:
        doc = MedicalDocument(
            content=entry.get('content', ''),
            source=entry.get('source', 'internal_kb'),
            category=entry.get('category', 'general'),
            confidence=entry.get('confidence', 1.0),
            last_updated=entry.get('last_updated'),
            doc_id=entry.get('doc_id'),
            metadata=entry.get('metadata', {})
        )
        documents.append(doc)
    return documents

if __name__ == "__main__":
    kb_path = "../../Knowledge-base/med_knowledge.json"
    kb_data = load_medical_kb(kb_path)
    documents = preprocess_documents(kb_data)
    print(f"Loaded {len(documents)} medical documents.")
