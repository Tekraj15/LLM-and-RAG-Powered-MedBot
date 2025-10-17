from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .document_schema import MedicalDocument

# Configurable defaults
CHUNK_SIZE = 500  # Tokens/characters
CHUNK_OVERLAP = 100

def chunk_document(doc: MedicalDocument, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[MedicalDocument]:
    """
    Chunks the document content semantically while preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # Semantic breaks (paragraphs, sentences)
        keep_separator=True
    )
    chunks = splitter.split_text(doc.content)
    
    chunked_docs = []
    for idx, chunk in enumerate(chunks):
        chunked_doc = MedicalDocument(
            content=chunk,
            source=doc.source,
            category=doc.category,
            confidence=doc.confidence,
            last_updated=doc.last_updated,
            doc_id=f"{doc.doc_id}_chunk_{idx}" if doc.doc_id else f"chunk_{idx}",
            metadata={**doc.metadata, "chunk_index": idx}  # Add chunk metadata
        )
        chunked_docs.append(chunked_doc)
    return chunked_docs