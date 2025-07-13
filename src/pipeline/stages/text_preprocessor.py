"""
Text Preprocessing Pipeline Stage.

This module handles text normalization and preprocessing as part of the
intent recognition pipeline.
"""

import re
from typing import Dict, Any, Optional

from src.pipeline.base import PipelineStage, PipelineData


class TextPreprocessorStage(PipelineStage):
    """
    Pipeline stage for text preprocessing and normalization.
    
    This stage:
    - Converts text to lowercase
    - Expands contractions
    - Removes special characters
    - Normalizes whitespace
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text preprocessor stage."""
        super().__init__(name="TextPreprocessor", config=config)
        
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
        
        # Allow custom contractions from config
        if config and 'contractions' in config:
            self.contractions.update(config['contractions'])
    
    async def _initialize(self):
        """No special initialization needed for text preprocessing."""
        self.logger.debug("TextPreprocessor stage initialized")
    
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Process the text through all normalization steps.
        
        Args:
            data: Pipeline data containing the query text
            
        Returns:
            Pipeline data with normalized text
        """
        # Get the query text
        if isinstance(data.raw_input, str):
            text = data.raw_input
        elif isinstance(data.raw_input, dict) and 'query' in data.raw_input:
            text = data.raw_input['query']
        else:
            raise ValueError("No text found in pipeline data")
        
        self.logger.debug(f"Processing text: {text}")
        
        # Apply preprocessing steps
        text = self.lowercase(text)
        text = self.expand_contractions(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        
        # Store results
        data.add_stage_result(self.name, 'normalized_text', text)
        data.add_stage_result(self.name, 'original_text', data.raw_input)
        
        # Also add to metadata for easy access
        data.add_metadata('normalized_query', text)
        
        self.logger.debug(f"Normalized text: {text}")
        
        return data
    
    async def validate_input(self, data: PipelineData) -> bool:
        """Validate that input contains text to process."""
        if isinstance(data.raw_input, str):
            return True
        elif isinstance(data.raw_input, dict) and 'query' in data.raw_input:
            return True
        
        self.logger.error("No valid text input found in pipeline data")
        return False
    
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions to full forms."""
        for contraction, expansion in self.contractions.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters while preserving meaningful punctuation."""
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!\-]', ' ', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing whitespace
        return text.strip()