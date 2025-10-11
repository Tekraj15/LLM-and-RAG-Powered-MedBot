from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class MedicalDocument:
    content: str
    source: str
    category: str
    confidence: float = 1.0
    last_updated: Optional[str] = None
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
