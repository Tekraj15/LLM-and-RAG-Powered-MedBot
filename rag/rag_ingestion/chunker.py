from .document_schema import MedicalDocument
from typing import List

# Simple chunker: splits content into chunks of N characters
CHUNK_SIZE = 500

def chunk_document(doc: MedicalDocument, chunk_size: int = CHUNK_SIZE) -> List[MedicalDocument]:
    content = doc.content
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    chunked_docs = []
    for idx, chunk in enumerate(chunks):
        chunked_doc = MedicalDocument(
            content=chunk,
            source=doc.source,
            category=doc.category,
            confidence=doc.confidence,
            last_updated=doc.last_updated,
            doc_id=f"{doc.doc_id}_chunk{idx}" if doc.doc_id else None,
            metadata=doc.metadata
        )
        chunked_docs.append(chunked_doc)
    return chunked_docs
