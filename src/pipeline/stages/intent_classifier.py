"""
Intent Classification Pipeline Stage.

This module handles intent classification based on taxonomy and patterns
as part of the intent recognition pipeline.
"""

from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple

from src.pipeline.base import PipelineStage, PipelineData
from src.agents.intent_models import Intent


class IntentClassifierStage(PipelineStage):
    """
    Pipeline stage for classifying intents based on taxonomy and patterns.
    
    This stage:
    - Performs keyword-based intent classification
    - Combines semantic scores with keyword matching
    - Generates ranked intent candidates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intent classifier stage."""
        super().__init__(name="IntentClassifier", config=config)
        
        # Intent taxonomy mapping keywords to intent types
        self.intent_taxonomy = {
            'query': {
                'search': ['find', 'search', 'look for', 'where is', 'locate', 'discover'],
                'retrieve': ['get', 'fetch', 'show', 'display', 'list', 'view'],
                'analyze': ['analyze', 'examine', 'investigate', 'inspect', 'evaluate', 'assess']
            },
            'action': {
                'create': ['create', 'make', 'generate', 'build', 'add', 'new'],
                'modify': ['update', 'change', 'edit', 'modify', 'alter', 'revise'],
                'delete': ['remove', 'delete', 'drop', 'clear', 'erase', 'destroy']
            },
            'system': {
                'configure': ['setup', 'configure', 'settings', 'config', 'initialize'],
                'monitor': ['check', 'monitor', 'status', 'health', 'track', 'watch']
            }
        }
        
        # Allow custom taxonomy from config
        if config and 'intent_taxonomy' in config:
            self.intent_taxonomy.update(config['intent_taxonomy'])
        
        # Create reverse mapping for quick lookup
        self.keyword_to_intent = {}
        self._build_keyword_mapping()
        
        # Classification parameters
        self.keyword_weight = config.get('keyword_weight', 0.5) if config else 0.5
        self.semantic_weight = config.get('semantic_weight', 0.5) if config else 0.5
    
    async def _initialize(self):
        """No special initialization needed for classifier."""
        self.logger.debug("Intent classifier stage initialized")
    
    def _build_keyword_mapping(self):
        """Build reverse mapping from keywords to intent types."""
        self.keyword_to_intent = {}
        
        for category, subcategories in self.intent_taxonomy.items():
            for subcat, keywords in subcategories.items():
                intent_type = f"{category}.{subcat}"
                for keyword in keywords:
                    if keyword not in self.keyword_to_intent:
                        self.keyword_to_intent[keyword] = []
                    self.keyword_to_intent[keyword].append(intent_type)
    
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Classify intents based on features and patterns.
        
        Args:
            data: Pipeline data containing tokens and features
            
        Returns:
            Pipeline data with classified intents
        """
        # Get tokens from tokenizer
        tokens = data.get_stage_result('Tokenizer', 'tokens', [])
        content_tokens = data.get_stage_result('Tokenizer', 'content_tokens', tokens)
        
        # Get semantic scores from feature extractor
        semantic_scores = data.get_stage_result('FeatureExtractor', 'semantic_scores', {})
        
        # Get normalized text
        normalized_text = data.get_stage_result('TextPreprocessor', 'normalized_text', '')
        
        self.logger.debug(f"Classifying intent for tokens: {tokens}")
        
        # Extract keywords from tokens
        keywords = self.extract_keywords(content_tokens)
        
        # Get keyword-based classification scores
        keyword_scores = self.classify_by_keywords(content_tokens)
        
        # Combine semantic and keyword scores
        combined_scores = self.combine_scores(semantic_scores, keyword_scores)
        
        # Create intent objects
        intents = self.create_intents(combined_scores, keywords)
        
        # Store classification results
        data.add_stage_result(self.name, 'keywords', keywords)
        data.add_stage_result(self.name, 'keyword_scores', keyword_scores)
        data.add_stage_result(self.name, 'combined_scores', combined_scores)
        data.add_stage_result(self.name, 'classified_intents', intents)
        
        # Add keywords to metadata
        data.add_metadata('keywords', keywords)
        
        self.logger.debug(f"Classified {len(intents)} intent candidates")
        
        return data
    
    def extract_keywords(self, tokens: List[str]) -> List[str]:
        """
        Extract relevant keywords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of identified keywords
        """
        keywords = []
        
        # Check each token against our keyword mapping
        for token in tokens:
            if token in self.keyword_to_intent:
                keywords.append(token)
            
            # Also check multi-word keywords
            for keyword in self.keyword_to_intent:
                if ' ' in keyword and keyword in ' '.join(tokens):
                    keywords.append(keyword)
        
        return list(set(keywords))  # Remove duplicates
    
    def classify_by_keywords(self, tokens: List[str]) -> Dict[str, float]:
        """
        Classify intents based on keyword matching.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary of intent types to scores
        """
        intent_scores = defaultdict(float)
        
        # Score based on keyword matches
        for token in tokens:
            if token in self.keyword_to_intent:
                for intent_type in self.keyword_to_intent[token]:
                    intent_scores[intent_type] += 1.0
        
        # Check multi-word keywords
        text = ' '.join(tokens)
        for keyword, intent_types in self.keyword_to_intent.items():
            if ' ' in keyword and keyword in text:
                for intent_type in intent_types:
                    intent_scores[intent_type] += 1.5  # Higher weight for multi-word matches
        
        # Normalize scores
        if intent_scores:
            max_score = max(intent_scores.values())
            for intent_type in intent_scores:
                intent_scores[intent_type] /= max_score
        
        return dict(intent_scores)
    
    def combine_scores(self, semantic_scores: Dict[str, float], 
                      keyword_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Combine semantic and keyword scores.
        
        Args:
            semantic_scores: Scores from semantic similarity
            keyword_scores: Scores from keyword matching
            
        Returns:
            Combined scores for each intent type
        """
        combined_scores = {}
        
        # Get all possible intent types
        all_intent_types = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        for intent_type in all_intent_types:
            # Get individual scores (default to 0 if not present)
            semantic_score = semantic_scores.get(intent_type, 0.0)
            keyword_score = keyword_scores.get(intent_type, 0.0)
            
            # Weighted combination
            combined_score = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )
            
            # Boost score if both methods agree
            if semantic_score > 0.5 and keyword_score > 0.5:
                combined_score *= 1.2  # 20% boost
            
            combined_scores[intent_type] = min(combined_score, 1.0)  # Cap at 1.0
        
        return combined_scores
    
    def create_intents(self, scores: Dict[str, float], keywords: List[str]) -> List[Intent]:
        """
        Create Intent objects from scores.
        
        Args:
            scores: Intent type scores
            keywords: Extracted keywords
            
        Returns:
            List of Intent objects
        """
        intents = []
        
        for intent_type, score in scores.items():
            intent = Intent(
                type=intent_type,
                confidence=score,
                keywords=keywords,
                entities=[],  # Entity extraction could be added later
                parameters={}
            )
            intents.append(intent)
        
        # Sort by confidence descending
        intents.sort(key=lambda x: x.confidence, reverse=True)
        
        return intents
    
    async def validate_input(self, data: PipelineData) -> bool:
        """Validate that input contains required data."""
        # We need tokens from tokenizer
        if not data.get_stage_result('Tokenizer', 'tokens'):
            self.logger.error("No tokens found from tokenizer stage")
            return False
        
        return True