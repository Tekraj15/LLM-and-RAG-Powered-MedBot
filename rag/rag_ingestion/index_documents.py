from .ingest_medical_kb import preprocess_documents, load_medical_kb
from .document_schema import MedicalDocument
# from rag_core.vector_store import MedicalVectorStore  # Uncomment when integrating

# Placeholder for embedding and indexing logic
def embed_and_index_documents(documents):
    # TODO: Integrate with vector store and embedding model
    for doc in documents:
        # Example: embed doc.content and index
        pass
    print(f"Indexed {len(documents)} documents.")

if __name__ == "__main__":
    kb_path = "../../Knowledge-base/med_knowledge.json"
    kb_data = load_medical_kb(kb_path)
    documents = preprocess_documents(kb_data)
    embed_and_index_documents(documents)
