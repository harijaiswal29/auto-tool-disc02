"""Standalone intent models to avoid circular imports."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

@dataclass
class Intent:
    """Represents a classified intent from user input."""
    type: str
    confidence: float
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'type': self.type,
            'confidence': self.confidence,
            'keywords': self.keywords,
            'entities': self.entities,
            'metadata': self.metadata
        }
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """Create from dictionary."""
        embedding = None
        if 'embedding' in data:
            embedding = np.array(data['embedding'])
        
        return cls(
            type=data['type'],
            confidence=data['confidence'],
            keywords=data.get('keywords', []),
            entities=data.get('entities', []),
            embedding=embedding,
            metadata=data.get('metadata', {})
        )

@dataclass
class IntentResult:
    """Result from intent recognition including confidence and metadata."""
    intent: Intent
    raw_query: str
    processed_query: str
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'intent': self.intent.to_dict(),
            'raw_query': self.raw_query,
            'processed_query': self.processed_query,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }