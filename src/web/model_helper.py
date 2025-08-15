"""
Model helper for safe ML model loading in the demo.
Ensures models load properly with fallbacks.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SafeModelLoader:
    """Safely load ML models with fallbacks."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.fallback_mode = False
        
    async def load_sentence_transformer(self, model_name: str = 'all-MiniLM-L6-v2') -> bool:
        """
        Try to load sentence transformer model with fallback.
        
        Returns:
            True if model loaded (or fallback enabled), False otherwise
        """
        try:
            # First try to import sentence_transformers
            from sentence_transformers import SentenceTransformer
            
            # Create cache directory
            cache_dir = Path.home() / '.cache' / 'sentence_transformers'
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to load model
            logger.info(f"Loading sentence transformer model: {model_name}")
            
            # Force CPU to avoid CUDA issues
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            # Load with specific cache dir
            self.model = SentenceTransformer(
                model_name,
                device='cpu',
                cache_folder=str(cache_dir)
            )
            
            # Test the model
            test_embedding = self.model.encode("test sentence")
            
            self.model_loaded = True
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except ImportError as e:
            logger.warning(f"sentence_transformers not installed: {e}")
            self.fallback_mode = True
            return True  # Use fallback
            
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            logger.info("Will use fallback embedding method")
            self.fallback_mode = True
            return True  # Use fallback
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embeddings, using model or fallback.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        if self.model_loaded and self.model:
            # Use real model
            return self.model.encode(text)
        else:
            # Use simple fallback embedding
            return self._fallback_encode(text)
    
    def _fallback_encode(self, text: str) -> np.ndarray:
        """
        Simple fallback encoding based on text features.
        
        Args:
            text: Text to encode
            
        Returns:
            Fixed-size embedding vector
        """
        # Create a simple 384-dimensional embedding (same as MiniLM)
        embedding = np.zeros(384)
        
        # Simple feature extraction
        words = text.lower().split()
        
        # Word count feature
        embedding[0] = len(words) / 10.0
        
        # Character count feature
        embedding[1] = len(text) / 100.0
        
        # Keyword presence features
        keywords = {
            'search': 2, 'find': 3, 'create': 4, 'update': 5,
            'delete': 6, 'list': 7, 'get': 8, 'analyze': 9,
            'weather': 10, 'file': 11, 'database': 12, 'table': 13
        }
        
        for word in words:
            if word in keywords:
                embedding[keywords[word]] = 1.0
        
        # Hash-based features for unknown words
        for i, word in enumerate(words[:20]):  # Use first 20 words
            hash_val = hash(word) % 300
            embedding[50 + hash_val] = 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def is_ready(self) -> bool:
        """Check if model is ready (either loaded or in fallback mode)."""
        return self.model_loaded or self.fallback_mode


class SimplifiedIntentRecognizer:
    """Simplified intent recognizer that doesn't require ML models."""
    
    def __init__(self):
        self.intent_patterns = {
            'query.search': ['find', 'search', 'look', 'locate', 'discover'],
            'query.retrieve': ['get', 'fetch', 'retrieve', 'show', 'display', 'list'],
            'query.analyze': ['analyze', 'examine', 'inspect', 'evaluate', 'assess'],
            'action.create': ['create', 'make', 'generate', 'add', 'new'],
            'action.modify': ['update', 'edit', 'modify', 'change', 'alter'],
            'action.delete': ['delete', 'remove', 'clear', 'drop', 'erase'],
            'query.weather': ['weather', 'temperature', 'forecast', 'climate'],
            'query.database': ['database', 'table', 'sql', 'query', 'schema'],
            'query.filesystem': ['file', 'directory', 'folder', 'path', 'python']
        }
    
    def recognize_intent(self, query: str) -> Dict[str, Any]:
        """
        Recognize intent from query using pattern matching.
        
        Args:
            query: User query
            
        Returns:
            Intent information
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Score each intent type
        intent_scores = {}
        for intent_type, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent_type] = score
        
        # Get best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(0.9, 0.5 + (intent_scores[best_intent] * 0.2))
        else:
            best_intent = 'query.general'
            confidence = 0.5
        
        # Extract keywords
        keywords = [w for w in words if len(w) > 3][:5]
        
        return {
            'type': best_intent,
            'confidence': confidence,
            'keywords': keywords
        }


# Global instance
model_loader = SafeModelLoader()
intent_recognizer = SimplifiedIntentRecognizer()