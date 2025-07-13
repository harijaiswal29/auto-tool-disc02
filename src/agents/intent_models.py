"""
Shared models and components for intent recognition.

This module contains common data structures and utilities used by both
the legacy and pipeline-based intent recognition implementations.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class Intent:
    """Represents a classified intent with confidence score."""
    type: str  # e.g., "query.search", "action.create"
    confidence: float
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentResult:
    """Result of intent recognition processing."""
    query: str
    normalized_query: str
    primary_intent: Intent
    all_intents: List[Intent]
    features: Dict[str, Any]
    processing_time_ms: float
    confidence_passed: bool


class TextPreprocessor:
    """Handles text normalization and preprocessing."""
    
    def __init__(self):
        self.contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "doesn't": "does not",
            "didn't": "did not",
            "let's": "let us",
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will"
        }
    
    async def process(self, text: str) -> str:
        """Process text through all normalization steps."""
        text = self.lowercase(text)
        text = self.expand_contractions(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        return text
    
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions to full forms."""
        for contraction, expansion in self.contractions.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters while preserving meaningful punctuation."""
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!\-]', ' ', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class MultiIntentHandler:
    """Handles queries with multiple intents."""
    
    def __init__(self):
        self.separators = [' and ', ' then ', ' also ', ' plus ', '. ', '; ']
        self.dependency_patterns = {
            'sequential': ['then', 'after', 'next'],
            'parallel': ['and', 'also', 'plus'],
            'conditional': ['if', 'when', 'unless']
        }
    
    async def detect_multi_intent(self, query: str) -> bool:
        """Detect if query contains multiple intents."""
        for separator in self.separators:
            if separator in query.lower():
                return True
        return False
    
    async def split_intents(self, query: str) -> List[str]:
        """Split query into separate intent queries."""
        segments = [query]
        
        for separator in self.separators:
            new_segments = []
            for segment in segments:
                parts = segment.split(separator)
                new_segments.extend(parts)
            segments = new_segments
        
        # Clean up segments
        segments = [s.strip() for s in segments if s.strip()]
        
        return segments
    
    def determine_execution_order(self, segments: List[str]) -> List[Tuple[str, str]]:
        """Determine execution order and dependencies."""
        execution_plan = []
        
        for i, segment in enumerate(segments):
            execution_type = 'parallel'  # Default
            
            # Check for dependency indicators
            for dep_type, patterns in self.dependency_patterns.items():
                if any(pattern in segment.lower() for pattern in patterns):
                    execution_type = dep_type
                    break
            
            execution_plan.append((segment, execution_type))
        
        return execution_plan