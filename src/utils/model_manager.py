"""
Model Manager for shared model instances.

This module provides a singleton pattern for managing expensive ML models
like sentence transformers to avoid redundant loading and reduce memory usage.
"""

import threading
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
import os

from src.utils.logger import get_logger


class ModelManager:
    """
    Singleton manager for ML models.
    
    This class ensures that expensive models are loaded only once
    and shared across all components that need them.
    """
    
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _logger = None
    
    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._logger = get_logger("ModelManager")
        return cls._instance
    
    def get_sentence_transformer(self, model_name: str = 'all-MiniLM-L6-v2', 
                               device: Optional[str] = None) -> SentenceTransformer:
        """
        Get a sentence transformer model instance.
        
        Args:
            model_name: Name of the model to load
            device: Device to load model on ('cpu', 'cuda', or None for auto)
            
        Returns:
            SentenceTransformer instance
        """
        model_key = f"sentence_transformer_{model_name}"
        
        if model_key not in self._models:
            with self._lock:
                # Double-check pattern
                if model_key not in self._models:
                    self._logger.info(f"Loading sentence transformer model: {model_name}")
                    
                    # Determine device
                    if device is None:
                        # Force CPU to avoid CUDA issues
                        device = 'cpu'
                    
                    # Load model
                    model = SentenceTransformer(model_name, device=device)
                    
                    # Put model in eval mode
                    model.eval()
                    
                    # Store in cache
                    self._models[model_key] = model
                    
                    self._logger.info(f"Model {model_name} loaded successfully on {device}")
        
        return self._models[model_key]
    
    def preload_models(self, models_config: Optional[Dict[str, Any]] = None):
        """
        Preload models based on configuration.
        
        Args:
            models_config: Dictionary with model configurations
        """
        if models_config is None:
            # Default models to preload
            models_config = {
                'sentence_transformer': {
                    'model_name': 'all-MiniLM-L6-v2',
                    'device': 'cpu'
                }
            }
        
        self._logger.info("Preloading models...")
        
        for model_type, config in models_config.items():
            if model_type == 'sentence_transformer':
                self.get_sentence_transformer(
                    model_name=config.get('model_name', 'all-MiniLM-L6-v2'),
                    device=config.get('device', 'cpu')
                )
        
        self._logger.info(f"Preloaded {len(self._models)} models")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {}
        
        for key, model in self._models.items():
            model_info = {
                'loaded': True,
                'type': type(model).__name__
            }
            
            # Add model-specific info
            if isinstance(model, SentenceTransformer):
                model_info['device'] = str(model.device)
                model_info['max_seq_length'] = model.max_seq_length
                
                # Get model size estimate
                param_count = sum(p.numel() for p in model.parameters())
                model_info['parameters'] = param_count
                model_info['size_mb'] = param_count * 4 / 1024 / 1024  # Assuming float32
            
            info[key] = model_info
        
        return info
    
    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear cached models.
        
        Args:
            model_type: Specific model type to clear, or None for all
        """
        with self._lock:
            if model_type:
                keys_to_remove = [k for k in self._models.keys() if k.startswith(model_type)]
                for key in keys_to_remove:
                    del self._models[key]
                    self._logger.info(f"Cleared model: {key}")
            else:
                self._models.clear()
                self._logger.info("Cleared all cached models")
    
    def warmup_model(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Warmup a model by running a test inference.
        
        Args:
            model_name: Name of the model to warmup
        """
        self._logger.debug(f"Warming up model: {model_name}")
        
        model = self.get_sentence_transformer(model_name)
        
        # Run a test inference to ensure model is ready
        test_sentences = ["This is a warmup sentence.", "Another test sentence."]
        _ = model.encode(test_sentences)
        
        self._logger.debug(f"Model {model_name} warmed up successfully")


# Global instance getter
_manager_instance = None


def get_model_manager() -> ModelManager:
    """
    Get the global ModelManager instance.
    
    Returns:
        ModelManager singleton instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance


# Convenience functions
def get_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Convenience function to get a sentence transformer model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        SentenceTransformer instance
    """
    return get_model_manager().get_sentence_transformer(model_name)


def preload_models(models_config: Optional[Dict[str, Any]] = None):
    """
    Convenience function to preload models.
    
    Args:
        models_config: Model configuration dictionary
    """
    get_model_manager().preload_models(models_config)