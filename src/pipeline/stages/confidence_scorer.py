"""
Confidence Scoring Pipeline Stage.

This module handles confidence score calculation for intent classifications
as part of the intent recognition pipeline.
"""

from typing import Dict, Any, Optional, List

from src.pipeline.base import PipelineStage, PipelineData
from src.agents.intent_models import Intent


class ConfidenceScorerStage(PipelineStage):
    """
    Pipeline stage for calculating confidence scores for intent classifications.
    
    This stage:
    - Combines multiple scoring factors
    - Applies confidence thresholds
    - Ranks and filters intents
    - Determines primary intent
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the confidence scorer stage."""
        super().__init__(name="ConfidenceScorer", config=config)
        
        # Feature weights for confidence calculation
        self.feature_weights = config.get('feature_weights', {
            'semantic_similarity': 0.3,
            'keyword_match': 0.5,
            'context_relevance': 0.1,
            'pattern_match': 0.1
        }) if config else {
            'semantic_similarity': 0.3,
            'keyword_match': 0.5,
            'context_relevance': 0.1,
            'pattern_match': 0.1
        }
        
        # Confidence thresholds
        self.confidence_threshold = config.get('confidence_threshold', 0.7) if config else 0.7
        self.similarity_threshold = config.get('similarity_threshold', 0.7) if config else 0.7
        
        # Scoring parameters
        self.keyword_boost_threshold = config.get('keyword_boost_threshold', 0.8) if config else 0.8
        self.agreement_boost = config.get('agreement_boost', 0.2) if config else 0.2
    
    async def _initialize(self):
        """No special initialization needed for confidence scorer."""
        self.logger.debug("Confidence scorer stage initialized")
    
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Calculate confidence scores for classified intents.
        
        Args:
            data: Pipeline data containing classified intents and features
            
        Returns:
            Pipeline data with scored and ranked intents
        """
        # Get intents from classifier
        intents = data.get_stage_result('IntentClassifier', 'classified_intents', [])
        
        # Get scoring components
        semantic_scores = data.get_stage_result('FeatureExtractor', 'semantic_scores', {})
        keyword_scores = data.get_stage_result('IntentClassifier', 'keyword_scores', {})
        context_score = data.get_stage_result('ContextEnricher', 'context_score', 0.5)
        features = data.get_stage_result('FeatureExtractor', 'features', {})
        
        self.logger.debug(f"Scoring {len(intents)} intent candidates")
        
        # Calculate confidence scores for each intent
        scored_intents = []
        for intent in intents:
            confidence = await self._calculate_confidence(
                intent.type,
                semantic_scores,
                keyword_scores,
                context_score,
                features
            )
            
            # Update intent confidence
            intent.confidence = confidence
            scored_intents.append(intent)
        
        # Sort by confidence
        scored_intents.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply thresholds and filtering
        filtered_intents = self._filter_intents(scored_intents)
        
        # Determine primary intent
        primary_intent = self._select_primary_intent(filtered_intents)
        
        # Calculate confidence statistics
        confidence_stats = self._calculate_confidence_stats(scored_intents)
        
        # Store results
        data.add_stage_result(self.name, 'scored_intents', scored_intents)
        data.add_stage_result(self.name, 'filtered_intents', filtered_intents)
        data.add_stage_result(self.name, 'primary_intent', primary_intent)
        data.add_stage_result(self.name, 'confidence_stats', confidence_stats)
        data.add_stage_result(self.name, 'confidence_passed', 
                            primary_intent.confidence >= self.confidence_threshold)
        
        # Add to metadata
        data.add_metadata('primary_intent_type', primary_intent.type)
        data.add_metadata('primary_intent_confidence', primary_intent.confidence)
        data.add_metadata('confidence_passed', primary_intent.confidence >= self.confidence_threshold)
        
        self.logger.debug(f"Primary intent: {primary_intent.type} "
                         f"(confidence: {primary_intent.confidence:.2f})")
        
        return data
    
    async def _calculate_confidence(self, intent_type: str,
                                  semantic_scores: Dict[str, float],
                                  keyword_scores: Dict[str, float],
                                  context_score: float,
                                  features: Dict[str, Any]) -> float:
        """
        Calculate confidence score for an intent.
        
        Args:
            intent_type: Type of intent
            semantic_scores: Semantic similarity scores
            keyword_scores: Keyword matching scores
            context_score: Context relevance score
            features: Additional features
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get individual scores
        scores = {
            'semantic_similarity': semantic_scores.get(intent_type, 0.0),
            'keyword_match': keyword_scores.get(intent_type, 0.0),
            'context_relevance': context_score,
            'pattern_match': 0.0  # Could be extended with pattern matching
        }
        
        # Apply keyword boost if strong match
        if scores['keyword_match'] >= self.keyword_boost_threshold:
            scores['keyword_match'] = 1.0
        
        # Check for agreement between methods
        if scores['semantic_similarity'] > 0.5 and scores['keyword_match'] > 0.5:
            # Both methods agree - apply boost
            agreement_factor = 1.0 + self.agreement_boost
        else:
            agreement_factor = 1.0
        
        # Calculate weighted average
        confidence = sum(
            self.feature_weights.get(key, 0.25) * value
            for key, value in scores.items()
        )
        
        # Apply agreement factor
        confidence *= agreement_factor
        
        # Apply intent-specific adjustments
        confidence = self._apply_intent_adjustments(intent_type, confidence, features)
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _apply_intent_adjustments(self, intent_type: str, base_confidence: float,
                                 features: Dict[str, Any]) -> float:
        """
        Apply intent-specific confidence adjustments.
        
        Args:
            intent_type: Type of intent
            base_confidence: Base confidence score
            features: Additional features
            
        Returns:
            Adjusted confidence score
        """
        adjusted_confidence = base_confidence
        
        # Boost query intents if question detected
        if intent_type.startswith('query.') and features.get('has_question', False):
            adjusted_confidence *= 1.1
        
        # Boost action intents if imperative mood detected
        linguistic_features = features.get('linguistic_features', {})
        if intent_type.startswith('action.') and linguistic_features.get('imperative_mood', False):
            adjusted_confidence *= 1.1
        
        # Boost system intents for specific keywords
        keyword_features = features.get('keyword_features', {})
        if intent_type.startswith('system.'):
            if intent_type == 'system.configure' and keyword_features.get('has_create_keyword', False):
                adjusted_confidence *= 1.05
            elif intent_type == 'system.monitor' and keyword_features.get('has_analyze_keyword', False):
                adjusted_confidence *= 1.05
        
        return min(adjusted_confidence, 1.0)
    
    def _filter_intents(self, intents: List[Intent]) -> List[Intent]:
        """
        Filter intents based on thresholds.
        
        Args:
            intents: List of scored intents
            
        Returns:
            Filtered list of intents
        """
        if not intents:
            return []
        
        # If no intents meet the threshold, keep at least the top one
        if all(intent.confidence < self.similarity_threshold for intent in intents):
            return [intents[0]]  # Keep the best one
        
        # Otherwise, filter by threshold
        filtered = [intent for intent in intents 
                   if intent.confidence >= self.similarity_threshold]
        
        # Limit to top 5 to avoid too many options
        return filtered[:5]
    
    def _select_primary_intent(self, intents: List[Intent]) -> Intent:
        """
        Select the primary intent from filtered candidates.
        
        Args:
            intents: List of filtered intents
            
        Returns:
            Primary intent
        """
        if not intents:
            # Create a default fallback intent
            return Intent(
                type="query.search",
                confidence=0.5,
                keywords=[],
                entities=[]
            )
        
        # Return the highest confidence intent
        return intents[0]
    
    def _calculate_confidence_stats(self, intents: List[Intent]) -> Dict[str, float]:
        """
        Calculate statistics about confidence scores.
        
        Args:
            intents: List of scored intents
            
        Returns:
            Dictionary of statistics
        """
        if not intents:
            return {
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'avg_confidence': 0.0,
                'confidence_spread': 0.0
            }
        
        confidences = [intent.confidence for intent in intents]
        
        return {
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'avg_confidence': sum(confidences) / len(confidences),
            'confidence_spread': max(confidences) - min(confidences)
        }
    
    async def validate_input(self, data: PipelineData) -> bool:
        """Validate that input contains required data."""
        # We need classified intents
        if not data.get_stage_result('IntentClassifier', 'classified_intents'):
            self.logger.error("No classified intents found")
            return False
        
        return True