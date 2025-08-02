"""
Shared models and components for intent recognition.

This module contains common data structures and utilities used by both
the legacy and pipeline-based intent recognition implementations.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# Import from the new location to avoid circular imports
from src.models.intent import Intent, IntentResult

# Re-export for backward compatibility
__all__ = ['Intent', 'IntentResult', 'TextPreprocessor', 'MultiIntentHandler']


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
        """Detect if query contains multiple intents.
        
        Enhanced to avoid false positives for common phrases.
        """
        query_lower = query.lower()
        
        # Common phrases that should NOT be split
        false_positive_phrases = [
            'search and process',  # Common combined action
            'read and write',      # Common file operation
            'get and set',         # Common accessor pattern
            'create and configure', # Common setup pattern
            'track and monitor',   # Common monitoring pattern
            'backup and restore',  # Common data operation
            'import and export',   # Common data operation
            'login and logout',    # Common auth operation
            'start and stop',      # Common control operation
            'open and close',      # Common file operation
            'find and analyze',    # Common workflow
            'fetch and display',   # Common data operation
            'load and save',       # Common file operation
            'copy and paste',      # Common edit operation
        ]
        
        # First check if query contains false positive phrases
        for phrase in false_positive_phrases:
            if phrase in query_lower:
                return False
        
        # Check for verb proximity - if two verbs are connected by 'and' without 
        # significant words between them, it's likely a compound action
        import re
        verb_and_pattern = r'\b(\w+)\s+and\s+(\w+)\b'
        matches = re.findall(verb_and_pattern, query_lower)
        
        common_verbs = {'search', 'find', 'get', 'fetch', 'create', 'make', 'update', 
                       'modify', 'delete', 'remove', 'analyze', 'process', 'export',
                       'import', 'read', 'write', 'view', 'display', 'check', 'monitor'}
        
        for match in matches:
            verb1, verb2 = match
            # If both words are common verbs and close together, it's likely a compound action
            if verb1 in common_verbs and verb2 in common_verbs:
                # Check if there are significant words between them
                pattern = rf'\b{verb1}\s+and\s+{verb2}\b'
                if re.search(pattern, query_lower):
                    return False  # It's a compound action, not multi-intent
        
        # Check for actual multi-intent separators
        # But require them to be between substantial parts
        for separator in self.separators:
            if separator in query_lower:
                # Skip 'and' if it's between two verbs (handled above)
                if separator == ' and ':
                    # Check for pattern with multiple 'and' separators (3+ actions)
                    and_count = query_lower.count(' and ')
                    if and_count >= 2:
                        # Extract all verb phrases separated by 'and'
                        parts = query_lower.split(' and ')
                        verb_phrases = []
                        for part in parts:
                            words = part.strip().split()
                            for word in words:
                                if word in common_verbs:
                                    verb_phrases.append((word, part.strip()))
                                    break
                        
                        # If we have 3+ different verbs, it's likely multi-intent
                        if len(verb_phrases) >= 3:
                            unique_verbs = set(vp[0] for vp in verb_phrases)
                            if len(unique_verbs) >= 3:
                                return True
                    
                    # Also check for comma-separated lists with 'and'
                    if ',' in query_lower and ' and ' in query_lower:
                        # This is likely a list of actions
                        return True
                    continue
                    
                parts = query_lower.split(separator)
                # Only consider it multi-intent if both parts are substantial
                # and contain different conceptual actions
                if len(parts) >= 2 and all(len(part.strip()) > 5 for part in parts):
                    # Check if parts have different action verbs
                    parts_verbs = []
                    for part in parts:
                        part_verbs = [w for w in part.split() if w in common_verbs]
                        if part_verbs:
                            parts_verbs.append(set(part_verbs))
                    
                    # If we have verbs in multiple parts and they're different, it's multi-intent
                    if len(parts_verbs) >= 2:
                        # Check if verb sets are different
                        if not all(parts_verbs[0] == vset for vset in parts_verbs[1:]):
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