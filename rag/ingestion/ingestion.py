from .loader import load_json, load_txt, load_pdf
from .chunker import chunk_document
from .document_schema import MedicalDocument
from typing import List
import os

def ingest_documents(paths: List[str], doc_type: str, source: str, category: str) -> List[MedicalDocument]:
    docs = []
    for path in paths:
        if doc_type == 'json':
            entries = load_json(path)
            for entry in entries:
                doc = MedicalDocument(
                    content=entry.get('content', ''),
                    source=source,
                    category=category,
                    confidence=entry.get('confidence', 1.0),
                    last_updated=entry.get('last_updated'),
                    doc_id=entry.get('doc_id'),
                    metadata=entry.get('metadata', {})
                )
                docs.append(doc)
        elif doc_type == 'txt':
            content = load_txt(path)
            doc = MedicalDocument(content=content, source=source, category=category)
            docs.append(doc)
        elif doc_type == 'pdf':
            content = load_pdf(path)
            doc = MedicalDocument(content=content, source=source, category=category)
            docs.append(doc)
    return docs

def ingest_and_chunk_all():
    # Example usage: ingest KB (json), guidelines (pdf), notes (txt)
    kb_docs = ingest_documents(["../../Knowledge-base/med_knowledge.json"], 'json', 'internal_kb', 'general')
    # Add more sources as needed
    all_docs = kb_docs # + guideline_docs + txt_docs
    chunked_docs = []
    for doc in all_docs:
        chunked_docs.extend(chunk_document(doc))
    print(f"Ingested and chunked {len(chunked_docs)} documents.")
    return chunked_docs

if __name__ == "__main__":
    ingest_and_chunk_all()
