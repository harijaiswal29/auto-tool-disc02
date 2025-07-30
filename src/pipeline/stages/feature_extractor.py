"""
Feature Extraction Pipeline Stage.

This module handles semantic embedding generation and feature extraction
as part of the intent recognition pipeline.
"""

import sys
import os
from typing import Dict, Any, Optional, List
from collections import OrderedDict

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline.base import PipelineStage, PipelineData
from src.utils.model_manager import get_model_manager


class FeatureExtractorStage(PipelineStage):
    """
    Pipeline stage for extracting semantic and linguistic features.
    
    This stage:
    - Generates semantic embeddings using sentence transformers
    - Calculates similarity scores with intent patterns
    - Extracts linguistic features
    - Manages embedding cache for performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature extractor stage."""
        super().__init__(name="FeatureExtractor", config=config)
        
        # Model configuration
        self.model_name = config.get('model', 'all-MiniLM-L6-v2') if config else 'all-MiniLM-L6-v2'
        self.cache_size = config.get('cache_size', 1000) if config else 1000
        self.similarity_threshold = config.get('similarity_threshold', 0.7) if config else 0.7
        
        # Model will be loaded during initialization
        self.model = None
        
        # Cache for embeddings
        self.embedding_cache = OrderedDict()
        
        # Intent patterns for semantic matching
        self.intent_patterns = {
            'query.search': [
                "find files",
                "search for documents",
                "locate resources",
                "where can I find"
            ],
            'query.retrieve': [
                "get data",
                "fetch information",
                "show me the results",
                "display content"
            ],
            'query.analyze': [
                "analyze the data",
                "examine the code",
                "investigate the issue",
                "evaluate performance"
            ],
            'action.create': [
                "create a new file",
                "generate a report",
                "build a project",
                "make a directory"
            ],
            'action.modify': [
                "update the configuration",
                "change the settings",
                "edit the file",
                "modify the code"
            ],
            'action.delete': [
                "remove the file",
                "delete the directory",
                "clear the cache",
                "drop the table"
            ],
            'system.configure': [
                "setup the environment",
                "configure the system",
                "change settings",
                "initialize the project"
            ],
            'system.monitor': [
                "check the status",
                "monitor performance",
                "track progress",
                "watch for changes"
            ]
        }
        
        # Pattern embeddings will be computed during initialization
        self.pattern_embeddings = {}
    
    async def _initialize(self):
        """Initialize the sentence transformer model and precompute embeddings."""
        self.logger.info(f"Getting sentence transformer model: {self.model_name}")
        
        # Get shared model instance from model manager
        model_manager = get_model_manager()
        self.model = model_manager.get_sentence_transformer(self.model_name, device='cpu')
        
        # Warmup the model if not already done
        model_manager.warmup_model(self.model_name)
        
        # Precompute embeddings for intent patterns
        self.logger.debug("Precomputing pattern embeddings")
        for intent_type, patterns in self.intent_patterns.items():
            self.pattern_embeddings[intent_type] = self.model.encode(patterns)
        
        self.logger.info("Feature extractor initialized successfully")
    
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Extract semantic and linguistic features from the text.
        
        Args:
            data: Pipeline data containing normalized text and tokens
            
        Returns:
            Pipeline data with extracted features
        """
        # Get normalized text
        normalized_text = data.get_stage_result('TextPreprocessor', 'normalized_text')
        if not normalized_text:
            normalized_text = data.raw_input if isinstance(data.raw_input, str) else str(data.raw_input)
        
        # Get tokens from tokenizer
        tokens = data.get_stage_result('Tokenizer', 'tokens', [])
        has_question = data.get_stage_result('Tokenizer', 'has_question', False)
        word_count = data.get_stage_result('Tokenizer', 'word_count', len(normalized_text.split()))
        
        self.logger.debug(f"Extracting features for: {normalized_text}")
        
        # Generate semantic embedding
        embedding = await self._get_embedding(normalized_text)
        
        # Calculate semantic similarity scores with intent patterns
        semantic_scores = self._calculate_semantic_scores(embedding)
        
        # Extract additional linguistic features
        linguistic_features = self._extract_linguistic_features(
            normalized_text, tokens, has_question, word_count
        )
        
        # Extract keyword features
        keyword_features = self._extract_keyword_features(tokens)
        
        # Combine all features
        features = {
            'embedding': embedding,
            'semantic_scores': semantic_scores,
            'linguistic_features': linguistic_features,
            'keyword_features': keyword_features,
            'has_question': has_question,
            'word_count': word_count
        }
        
        # Store results
        data.add_stage_result(self.name, 'features', features)
        data.add_stage_result(self.name, 'embedding', embedding)
        data.add_stage_result(self.name, 'semantic_scores', semantic_scores)
        
        # Add to metadata for easy access
        data.add_metadata('embedding', embedding.tolist())  # Convert to list for JSON serialization
        data.add_metadata('semantic_scores', semantic_scores)
        
        self.logger.debug(f"Feature extraction complete")
        
        return data
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get semantic embedding for text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache
        if text in self.embedding_cache:
            # Move to end (LRU behavior)
            self.embedding_cache.move_to_end(text)
            return self.embedding_cache[text]
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Update cache
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry
            self.embedding_cache.popitem(last=False)
        
        self.embedding_cache[text] = embedding
        
        return embedding
    
    def _calculate_semantic_scores(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Calculate semantic similarity scores with intent patterns.
        
        Args:
            embedding: Query embedding
            
        Returns:
            Similarity scores for each intent type
        """
        semantic_scores = {}
        
        for intent_type, pattern_embeddings in self.pattern_embeddings.items():
            # Calculate cosine similarity with all patterns
            similarities = cosine_similarity([embedding], pattern_embeddings)[0]
            
            # Take the maximum similarity as the score for this intent
            semantic_scores[intent_type] = float(np.max(similarities))
        
        return semantic_scores
    
    def _extract_linguistic_features(self, text: str, tokens: List[str], 
                                   has_question: bool, word_count: int) -> Dict[str, Any]:
        """
        Extract linguistic features from the text.
        
        Args:
            text: Normalized text
            tokens: List of tokens
            has_question: Whether text contains a question
            word_count: Number of words
            
        Returns:
            Dictionary of linguistic features
        """
        features = {
            'length': len(text),
            'word_count': word_count,
            'avg_word_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            'has_question': has_question,
            'ends_with_question': text.strip().endswith('?'),
            'starts_with_verb': self._starts_with_verb(tokens),
            'imperative_mood': self._is_imperative(tokens),
            'polite_request': self._is_polite_request(text.lower())
        }
        
        return features
    
    def _extract_keyword_features(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Extract keyword-based features.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary of keyword features
        """
        # Action keywords
        action_keywords = {
            'create': ['create', 'make', 'generate', 'build', 'add', 'new'],
            'read': ['find', 'search', 'get', 'fetch', 'show', 'display', 'list'],
            'update': ['update', 'change', 'edit', 'modify', 'alter', 'revise'],
            'delete': ['remove', 'delete', 'drop', 'clear', 'erase', 'destroy'],
            'analyze': ['analyze', 'examine', 'investigate', 'inspect', 'evaluate']
        }
        
        features = {}
        for action, keywords in action_keywords.items():
            features[f'has_{action}_keyword'] = any(token in keywords for token in tokens)
        
        return features
    
    def _starts_with_verb(self, tokens: List[str]) -> bool:
        """Check if text starts with a verb (simple heuristic)."""
        if not tokens:
            return False
        
        # Common command verbs
        command_verbs = {
            'find', 'search', 'get', 'create', 'update', 'delete', 'show',
            'list', 'analyze', 'check', 'monitor', 'configure', 'setup'
        }
        
        return tokens[0] in command_verbs
    
    def _is_imperative(self, tokens: List[str]) -> bool:
        """Check if sentence is in imperative mood."""
        return self._starts_with_verb(tokens) and '?' not in tokens
    
    def _is_polite_request(self, text: str) -> bool:
        """Check if text is a polite request."""
        polite_phrases = [
            'please', 'could you', 'can you', 'would you',
            'i would like', "i'd like", 'may i'
        ]
        
        return any(phrase in text for phrase in polite_phrases)
    
    async def validate_input(self, data: PipelineData) -> bool:
        """Validate that input contains text to process."""
        # We need either normalized text or raw input
        if data.get_stage_result('TextPreprocessor', 'normalized_text'):
            return True
        
        if isinstance(data.raw_input, str):
            return True
        
        self.logger.error("No valid text input found for feature extraction")
        return False