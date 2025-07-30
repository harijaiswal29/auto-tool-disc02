"""
Intent Recognition Agent - Pipeline Architecture.

This module implements the intent recognition functionality using a modular
pipeline architecture with separate, reusable stages.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pipeline import Pipeline, PipelineData
from src.pipeline.stages import (
    TextPreprocessorStage,
    TokenizerModule,
    FeatureExtractorStage,
    IntentClassifierStage,
    ContextEnricherStage,
    ConfidenceScorerStage
)
from src.pipeline.stages.state_manager import StateManagerStage
from src.agents.intent_models import IntentResult, Intent, MultiIntentHandler
from src.utils.logger import get_logger
from src.services.context_persistence_service import ContextPersistenceService
from src.monitoring.intent_recognition_metrics import get_metrics


class IntentRecognitionAgent:
    """
    Refactored Intent Recognition Agent using pipeline architecture.
    
    This agent processes natural language queries through a modular pipeline
    to determine user intents, enabling appropriate tool selection.
    """
    
    # Class-level pipeline cache to share pipelines across instances
    _pipeline_cache = {}
    _cache_lock = asyncio.Lock()
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Intent Recognition Agent with pipeline architecture."""
        self.logger = get_logger(__name__)
        
        # Load configuration
        if config is None:
            config = self._load_default_config()
        
        self.config = config
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Initialize multi-intent handler (still needed for complex queries)
        self.multi_intent_handler = MultiIntentHandler()
        
        # State management configuration
        self.enable_state_tracking = config.get('enable_state_tracking', True)
        
        # Context persistence configuration
        self.enable_persistence = config.get('enable_persistence', True)
        self.persistence_service = None
        
        # Create or get cached pipeline
        self.logger.info("Setting up pipeline...")
        self.pipeline = None  # Will be set asynchronously
        self._pipeline_key = self._generate_pipeline_key(config)
        
        # Get reference to state manager if enabled
        self.state_manager = None
        
        # Initialize persistence service if enabled
        if self.enable_persistence:
            self._init_persistence_service()
        
        # Initialize metrics collector
        self.metrics = get_metrics()
        self.collect_metrics = config.get('collect_metrics', True)
        
        self.logger.info("Intent Recognition Agent initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from config file."""
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                return full_config.get('intent_recognition', {})
        
        # Fallback configuration
        return {
            'model': 'all-MiniLM-L6-v2',
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.7,
            'cache_size': 1000
        }
    
    def _generate_pipeline_key(self, config: Dict[str, Any]) -> str:
        """Generate a unique key for pipeline caching based on config."""
        # Create a key based on important config parameters
        key_parts = [
            config.get('model', 'all-MiniLM-L6-v2'),
            str(config.get('similarity_threshold', 0.7)),
            str(config.get('confidence_threshold', 0.7)),
            str(config.get('enable_state_tracking', True))
        ]
        return "_".join(key_parts)
    
    async def _get_or_create_pipeline(self) -> Pipeline:
        """Get pipeline from cache or create a new one."""
        async with self._cache_lock:
            if self._pipeline_key not in self._pipeline_cache:
                self.logger.info(f"Creating new pipeline for key: {self._pipeline_key}")
                
                # Create pipeline stages
                stages = self._create_pipeline_stages(self.config)
                
                # Create the pipeline
                pipeline = Pipeline(stages, name="IntentRecognitionPipeline")
                
                # Initialize the pipeline
                await pipeline.initialize()
                
                # Cache the pipeline
                self._pipeline_cache[self._pipeline_key] = pipeline
                
                # Set state manager reference if needed
                if self.enable_state_tracking:
                    for stage in stages:
                        if isinstance(stage, StateManagerStage):
                            self.state_manager = stage
                            break
            else:
                self.logger.info(f"Using cached pipeline for key: {self._pipeline_key}")
                pipeline = self._pipeline_cache[self._pipeline_key]
                
                # Update state manager reference
                if self.enable_state_tracking:
                    for stage in pipeline.stages:
                        if isinstance(stage, StateManagerStage):
                            self.state_manager = stage
                            break
            
            return pipeline
    
    def _create_pipeline_stages(self, config: Dict[str, Any]) -> List:
        """Create and configure pipeline stages."""
        stages = []
        
        # Add state manager as first stage if enabled
        if self.enable_state_tracking:
            stages.append(StateManagerStage(config.get('state_manager', {})))
        
        # Core pipeline stages
        stages.extend([
            # Text preprocessing
            TextPreprocessorStage(config.get('text_preprocessor', {})),
            
            # Tokenization
            TokenizerModule(config.get('tokenizer', {})),
            
            # Feature extraction with semantic embeddings
            FeatureExtractorStage({
                'model': config.get('model', 'all-MiniLM-L6-v2'),
                'cache_size': config.get('cache_size', 1000),
                'similarity_threshold': self.similarity_threshold
            }),
            
            # Intent classification
            IntentClassifierStage(config.get('intent_classifier', {})),
            
            # Context enrichment
            ContextEnricherStage(config.get('context_enricher', {})),
            
            # Confidence scoring
            ConfidenceScorerStage({
                'confidence_threshold': self.confidence_threshold,
                'similarity_threshold': self.similarity_threshold
            })
        ])
        
        return stages
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntentResult:
        """
        Process a user query to extract intent using the pipeline.
        
        Args:
            query: The user's natural language query
            context: Optional context information
            
        Returns:
            IntentResult with classified intent and metadata
        """
        start_time = time.time()
        
        if context is None:
            context = {}
        
        # Ensure pipeline is initialized
        if self.pipeline is None:
            self.pipeline = await self._get_or_create_pipeline()
        
        self.logger.info(f"Processing query: {query}")
        
        # Check for multi-intent query
        is_multi_intent = await self.multi_intent_handler.detect_multi_intent(query)
        
        if is_multi_intent:
            self.logger.info("Multi-intent query detected")
            result = await self._handle_multi_intent_query(query, context)
        else:
            result = await self._process_single_intent(query, context)
        
        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000
        result.processing_time_ms = processing_time_ms
        
        self.logger.info(f"Intent recognized: {result.primary_intent.type} "
                        f"(confidence: {result.primary_intent.confidence:.2f}) "
                        f"in {processing_time_ms:.2f}ms")
        
        # Record metrics
        if self.collect_metrics:
            try:
                self.metrics.record_query_processing(result)
                
                # Record cache hit/miss if available
                if hasattr(self.pipeline, 'cache_hit'):
                    self.metrics.record_cache_access(self.pipeline.cache_hit)
            except Exception as e:
                self.logger.warning(f"Failed to record metrics: {e}")
        
        return result
    
    async def _process_single_intent(self, query: str, context: Dict[str, Any]) -> IntentResult:
        """Process a single intent query through the pipeline."""
        # Track stage timings for metrics
        stage_timings = {}
        stage_start = time.time()
        
        # Process through pipeline
        pipeline_result = await self.pipeline.process(query, context)
        
        # Collect stage timings if available
        if self.collect_metrics and hasattr(pipeline_result, 'stage_timings'):
            stage_timings = pipeline_result.stage_timings
        
        # Extract results from pipeline
        normalized_query = pipeline_result.get_stage_result('TextPreprocessor', 'normalized_text', query)
        primary_intent = pipeline_result.get_stage_result('ConfidenceScorer', 'primary_intent')
        all_intents = pipeline_result.get_stage_result('ConfidenceScorer', 'filtered_intents', [])
        confidence_passed = pipeline_result.get_stage_result('ConfidenceScorer', 'confidence_passed', False)
        
        # Build features dictionary for compatibility
        features = {
            'tokens': pipeline_result.get_stage_result('Tokenizer', 'tokens', []),
            'keywords': pipeline_result.get_stage_result('IntentClassifier', 'keywords', []),
            'semantic_scores': pipeline_result.get_stage_result('FeatureExtractor', 'semantic_scores', {}),
            'keyword_scores': pipeline_result.get_stage_result('IntentClassifier', 'keyword_scores', {}),
            'context_score': pipeline_result.get_stage_result('ContextEnricher', 'context_score', 0.5),
            'word_count': pipeline_result.get_stage_result('Tokenizer', 'word_count', 0),
            'has_question': pipeline_result.get_stage_result('Tokenizer', 'has_question', False)
        }
        
        # Create result
        return IntentResult(
            primary_intent=primary_intent,
            all_intents=all_intents,
            raw_query=query,
            processed_query=normalized_query,
            confidence_threshold_met=confidence_passed,
            metadata={
                'features': features,
                'processing_time_ms': 0  # Will be set by caller
            }
        )
    
    async def _handle_multi_intent_query(self, query: str, context: Dict[str, Any]) -> IntentResult:
        """Handle queries with multiple intents."""
        # Split query into segments
        segments = await self.multi_intent_handler.split_intents(query)
        
        all_intents = []
        all_features = {}
        
        # Process each segment through the pipeline
        for i, segment in enumerate(segments):
            # Process segment
            segment_result = await self._process_single_intent(segment, context)
            
            # Collect intents
            if segment_result.primary_intent:
                all_intents.append(segment_result.primary_intent)
            
            # Merge features
            for key, value in segment_result.features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(value)
        
        # Select primary intent (highest confidence)
        primary_intent = max(all_intents, key=lambda x: x.confidence) if all_intents else None
        
        if primary_intent is None:
            # Fallback intent
            primary_intent = Intent(
                type="query.search",
                confidence=0.5,
                keywords=[],
                entities=[]
            )
            all_intents = [primary_intent]
        
        # Create combined result
        return IntentResult(
            primary_intent=primary_intent,
            all_intents=all_intents,
            raw_query=query,
            processed_query=query.lower(),  # Simple normalization for multi-intent
            confidence_threshold_met=primary_intent.confidence >= self.confidence_threshold,
            metadata={
                'features': all_features,
                'processing_time_ms': 0  # Will be set by caller
            }
        )
    
    async def get_intent_details(self, intent_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific intent type.
        
        Args:
            intent_type: The intent type to get details for
            
        Returns:
            Dictionary with intent details
        """
        # Parse intent type
        parts = intent_type.split('.')
        if len(parts) != 2:
            return {}
        
        category, subcategory = parts
        
        # Get classifier stage to access taxonomy
        classifier_stage = next(
            (stage for stage in self.stages if isinstance(stage, IntentClassifierStage)),
            None
        )
        
        if not classifier_stage:
            return {}
        
        # Get keywords from taxonomy
        keywords = classifier_stage.intent_taxonomy.get(category, {}).get(subcategory, [])
        
        # Get example patterns from feature extractor
        extractor_stage = next(
            (stage for stage in self.stages if isinstance(stage, FeatureExtractorStage)),
            None
        )
        
        patterns = []
        if extractor_stage:
            patterns = extractor_stage.intent_patterns.get(intent_type, [])
        
        return {
            'type': intent_type,
            'category': category,
            'subcategory': subcategory,
            'keywords': keywords,
            'example_patterns': patterns
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        if self.pipeline is None:
            # Return basic info if pipeline not yet initialized
            return {
                'pipeline_name': 'IntentRecognitionPipeline',
                'stages': ['Not yet initialized'],
                'config': {
                    'similarity_threshold': self.similarity_threshold,
                    'confidence_threshold': self.confidence_threshold,
                    'state_tracking_enabled': self.enable_state_tracking
                },
                'cache_key': self._pipeline_key,
                'cached_pipelines': len(self._pipeline_cache)
            }
        
        info = {
            'pipeline_name': self.pipeline.name,
            'stages': self.pipeline.get_stage_names(),
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'confidence_threshold': self.confidence_threshold,
                'state_tracking_enabled': self.enable_state_tracking
            },
            'cache_key': self._pipeline_key,
            'cached_pipelines': len(self._pipeline_cache)
        }
        
        # Add state info if available
        if self.state_manager:
            info['current_state'] = self.state_manager.get_current_state_name()
            info['state_summary'] = self.state_manager.state_machine.get_conversation_summary()
        
        return info
    
    # State management methods
    
    def get_current_state(self) -> Optional[str]:
        """Get the current conversation state."""
        if not self.state_manager:
            return None
        return self.state_manager.get_current_state_name()
    
    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the state transition history."""
        if not self.state_manager:
            return []
        
        history = self.state_manager.get_state_history(limit)
        return [
            {
                'from_state': t.from_state,
                'to_state': t.to_state,
                'timestamp': t.timestamp.isoformat(),
                'trigger': t.trigger
            }
            for t in history
        ]
    
    def is_in_error_state(self) -> bool:
        """Check if the conversation is in an error state."""
        if not self.state_manager:
            return False
        return self.state_manager.is_in_error_state()
    
    def needs_user_input(self) -> bool:
        """Check if the system is waiting for user input."""
        if not self.state_manager:
            return False
        return self.state_manager.needs_user_input()
    
    async def handle_clarification(self, clarification: str) -> bool:
        """
        Handle user clarification when in CLARIFICATION_NEEDED state.
        
        Args:
            clarification: User's clarification text
            
        Returns:
            True if clarification was handled successfully
        """
        if not self.state_manager:
            self.logger.warning("State tracking not enabled")
            return False
        
        return await self.state_manager.handle_clarification(clarification)
    
    async def request_retry(self) -> bool:
        """
        Request to retry the current operation.
        
        Returns:
            True if retry was accepted
        """
        if not self.state_manager:
            self.logger.warning("State tracking not enabled")
            return False
        
        return await self.state_manager.request_retry()
    
    async def cancel_operation(self) -> bool:
        """
        Cancel the current operation.
        
        Returns:
            True if cancellation was successful
        """
        if not self.state_manager:
            self.logger.warning("State tracking not enabled")
            return False
        
        return await self.state_manager.cancel_operation()
    
    async def reset_conversation(self) -> bool:
        """
        Reset the conversation to idle state.
        
        Returns:
            True if reset was successful
        """
        if not self.state_manager:
            self.logger.warning("State tracking not enabled")
            return False
        
        return await self.state_manager.reset_to_idle()
    
    def _init_persistence_service(self):
        """Initialize persistence service asynchronously."""
        async def init():
            try:
                self.persistence_service = ContextPersistenceService()
                await self.persistence_service.initialize()
                self.logger.info("Persistence service initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize persistence service: {e}")
                self.persistence_service = None
                self.enable_persistence = False
        
        # Schedule initialization
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(init())
        except RuntimeError:
            # No event loop running yet
            pass
    
    # User Profile and Session Management Methods
    
    async def get_or_create_user(self, user_id: Optional[str] = None, 
                               username: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get or create a user profile.
        
        Args:
            user_id: Optional user ID
            username: Optional username
            
        Returns:
            User profile dictionary or None if persistence is disabled
        """
        if not self.persistence_service:
            return None
        
        return await self.persistence_service.get_or_create_user(user_id, username)
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: User ID
            preferences: Preferences to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.persistence_service:
            return False
        
        return await self.persistence_service.update_user_preferences(user_id, preferences)
    
    async def create_session(self, user_id: Optional[str] = None, 
                           domain: str = 'general') -> Optional[str]:
        """
        Create a new session.
        
        Args:
            user_id: Optional user ID to associate with session
            domain: Domain context (default: 'general')
            
        Returns:
            Session ID or None if persistence is disabled
        """
        if not self.persistence_service:
            return None
        
        return await self.persistence_service.create_session(user_id, domain)
    
    async def process_query_with_persistence(self, query: str, 
                                           session_id: Optional[str] = None,
                                           user_id: Optional[str] = None,
                                           domain: str = 'general') -> IntentResult:
        """
        Process query with full persistence support.
        
        This method handles session creation, user association, and 
        conversation history persistence automatically.
        
        Args:
            query: The user's natural language query
            session_id: Optional existing session ID
            user_id: Optional user ID
            domain: Domain context
            
        Returns:
            IntentResult with classified intent and metadata
        """
        context = {
            'domain': domain
        }
        
        if self.persistence_service:
            # Get or create session
            session = await self.persistence_service.get_or_create_session(
                session_id=session_id,
                user_id=user_id,
                domain=domain
            )
            
            context['session_id'] = session['session_id']
            if session.get('user_id'):
                context['user_id'] = session['user_id']
        else:
            # Fallback for non-persistent mode
            if session_id:
                context['session_id'] = session_id
            if user_id:
                context['user_id'] = user_id
        
        # Process query with enriched context
        result = await self.process_query(query, context)
        
        # Save conversation turn if persistence is enabled
        if self.persistence_service and context.get('session_id'):
            try:
                await self.persistence_service.save_conversation_turn(
                    session_id=context['session_id'],
                    query=query,
                    intent_result=result,
                    context={
                        'domain': domain,
                        'processing_time_ms': result.processing_time_ms
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to save conversation turn: {e}")
        
        return result
    
    async def get_user_statistics(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get usage statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary or None if persistence is disabled
        """
        if not self.persistence_service:
            return None
        
        db = self.persistence_service.db
        return await db.get_user_statistics(user_id)
    
    async def learn_from_feedback(self, session_id: str, feedback: Dict[str, Any]) -> bool:
        """
        Process user feedback to improve future interactions.
        
        Args:
            session_id: Current session ID
            feedback: Feedback data
            
        Returns:
            True if successful
        """
        if not self.persistence_service:
            return False
        
        # Record feedback in metrics
        if self.collect_metrics and 'intent' in feedback and 'correct' in feedback:
            self.metrics.record_feedback(feedback['intent'], feedback['correct'])
        
        return await self.persistence_service.learn_from_feedback(session_id, feedback)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of performance metrics.
        
        Returns:
            Dictionary containing comprehensive metrics
        """
        if not self.collect_metrics:
            return {"metrics_collection": "disabled"}
        
        return self.metrics.get_summary_metrics()
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to a file.
        
        Args:
            filepath: Path to export metrics to
        """
        if self.collect_metrics:
            self.metrics.export_metrics(filepath)
        else:
            self.logger.warning("Metrics collection is disabled")


# Example usage and testing
if __name__ == "__main__":
    async def test_intent_recognition():
        """Test the refactored Intent Recognition Agent."""
        # Create agent
        agent = IntentRecognitionAgent()
        
        # Get pipeline info
        print("Pipeline Configuration:")
        print(json.dumps(agent.get_pipeline_info(), indent=2))
        print("\n" + "="*50 + "\n")
        
        # Test queries
        test_queries = [
            "Find all Python files in the project",
            "Create a new configuration file and then update the settings",
            "Analyze the performance of the system",
            "Delete old log files",
            "Show me the current status and monitor for changes",
            "Can you help me search for documentation?",
            "Update the database schema"
        ]
        
        for query in test_queries:
            print(f"Query: {query}")
            result = await agent.process_query(query)
            print(f"Primary Intent: {result.primary_intent.type} (confidence: {result.primary_intent.confidence:.2f})")
            print(f"All Intents: {[(i.type, f'{i.confidence:.2f}') for i in result.all_intents]}")
            print(f"Keywords: {result.features.get('keywords', [])}")
            print(f"Processing Time: {result.processing_time_ms:.2f}ms")
            print(f"Confidence Passed: {result.confidence_passed}")
            print("-" * 40)
    
    # Run the test
    asyncio.run(test_intent_recognition())