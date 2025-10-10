import json
from pathlib import Path
from typing import List
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

def load_json(json_path: str) -> List[dict]:
    with open(json_path, 'r') as f:
        return json.load(f)

def load_txt(txt_path: str) -> str:
    with open(txt_path, 'r') as f:
        return f.read()

def load_pdf(pdf_path: str) -> str:
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required for PDF loading.")
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text
