"""
Tokenizer Pipeline Stage.

This module handles text tokenization and linguistic analysis as part of the
intent recognition pipeline.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

from src.pipeline.base import PipelineStage, PipelineData


class TokenizerModule(PipelineStage):
    """
    Pipeline stage for text tokenization and linguistic analysis.
    
    This stage:
    - Tokenizes text into words
    - Identifies question words
    - Extracts n-grams
    - Performs basic linguistic analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the tokenizer stage."""
        super().__init__(name="Tokenizer", config=config)
        
        # Question words for detecting interrogative sentences
        self.question_words = {'what', 'where', 'when', 'how', 'why', 'who', 
                              'which', 'whose', 'whom'}
        
        # Common stop words (minimal set for intent recognition)
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                          'at', 'to', 'for', 'of', 'with', 'by', 'from',
                          'as', 'is', 'was', 'are', 'were', 'been', 'be'}
        
        # Configure n-gram settings
        self.max_ngram = config.get('max_ngram', 3) if config else 3
        self.min_word_length = config.get('min_word_length', 2) if config else 2
    
    async def _initialize(self):
        """No special initialization needed for tokenizer."""
        self.logger.debug("Tokenizer stage initialized")
    
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Tokenize and analyze the text.
        
        Args:
            data: Pipeline data containing normalized text
            
        Returns:
            Pipeline data with tokenization results
        """
        # Get normalized text from previous stage
        normalized_text = data.get_stage_result('TextPreprocessor', 'normalized_text')
        
        if not normalized_text:
            # Fallback to raw input if preprocessing stage was skipped
            normalized_text = data.raw_input if isinstance(data.raw_input, str) else str(data.raw_input)
        
        self.logger.debug(f"Tokenizing text: {normalized_text}")
        
        # Perform tokenization
        tokens = self.tokenize(normalized_text)
        
        # Extract linguistic features
        word_count = len(tokens)
        unique_tokens = list(set(tokens))
        token_frequencies = Counter(tokens)
        
        # Identify question words
        has_question = self.detect_question(tokens)
        question_words_found = [w for w in tokens if w in self.question_words]
        
        # Filter stop words for content tokens
        content_tokens = [t for t in tokens if t not in self.stop_words and len(t) >= self.min_word_length]
        
        # Generate n-grams
        bigrams = self.get_ngrams(tokens, 2)
        trigrams = self.get_ngrams(tokens, 3) if self.max_ngram >= 3 else []
        
        # Detect sentence type
        sentence_type = self.detect_sentence_type(normalized_text, has_question)
        
        # Store results
        data.add_stage_result(self.name, 'tokens', tokens)
        data.add_stage_result(self.name, 'word_count', word_count)
        data.add_stage_result(self.name, 'unique_tokens', unique_tokens)
        data.add_stage_result(self.name, 'content_tokens', content_tokens)
        data.add_stage_result(self.name, 'token_frequencies', dict(token_frequencies))
        data.add_stage_result(self.name, 'has_question', has_question)
        data.add_stage_result(self.name, 'question_words', question_words_found)
        data.add_stage_result(self.name, 'bigrams', bigrams)
        data.add_stage_result(self.name, 'trigrams', trigrams)
        data.add_stage_result(self.name, 'sentence_type', sentence_type)
        
        # Add key features to metadata for easy access
        data.add_metadata('tokens', tokens)
        data.add_metadata('word_count', word_count)
        data.add_metadata('has_question', has_question)
        
        self.logger.debug(f"Tokenization complete: {word_count} tokens")
        
        return data
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation, but keep meaningful punctuation
        # This regex splits on spaces and punctuation but preserves the punctuation
        tokens = re.findall(r'\b\w+\b|[.!?]', text.lower())
        
        # Filter out single characters except meaningful punctuation
        tokens = [t for t in tokens if len(t) > 1 or t in '.!?']
        
        return tokens
    
    def detect_question(self, tokens: List[str]) -> bool:
        """
        Detect if the text is a question.
        
        Args:
            tokens: List of tokens
            
        Returns:
            True if question detected
        """
        # Check for question marks
        if '?' in tokens:
            return True
        
        # Check for question words at the beginning
        if tokens and tokens[0] in self.question_words:
            return True
        
        # Check for question words anywhere
        return any(token in self.question_words for token in tokens)
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from tokens.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            List of n-grams
        """
        if n > len(tokens):
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            # Skip n-grams that are all stop words
            if not all(token in self.stop_words for token in ngram):
                ngrams.append(ngram)
        
        return ngrams
    
    def detect_sentence_type(self, text: str, has_question: bool) -> str:
        """
        Detect the type of sentence.
        
        Args:
            text: Original text
            has_question: Whether question was detected
            
        Returns:
            Sentence type (question, command, statement)
        """
        if has_question:
            return "question"
        elif text.strip().endswith('!'):
            return "exclamation"
        elif any(text.lower().startswith(cmd) for cmd in ['please', 'could you', 'can you', 'would you']):
            return "polite_request"
        elif any(text.lower().startswith(cmd) for cmd in ['create', 'delete', 'update', 'find', 'show', 'get']):
            return "command"
        else:
            return "statement"
    
    async def validate_input(self, data: PipelineData) -> bool:
        """Validate that input contains text to tokenize."""
        # Check if we have normalized text from preprocessor
        if data.get_stage_result('TextPreprocessor', 'normalized_text'):
            return True
        
        # Otherwise check for raw text input
        if isinstance(data.raw_input, str):
            return True
        
        self.logger.error("No valid text input found for tokenization")
        return False