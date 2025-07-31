"""
Context Enrichment Pipeline Stage.

This module handles context integration and enrichment as part of the
intent recognition pipeline.
"""

from typing import Dict, Any, Optional, List
from collections import deque
from datetime import datetime

from src.pipeline.base import PipelineStage, PipelineData
from src.services.context_persistence_service import ContextPersistenceService


class ContextEnricherStage(PipelineStage):
    """
    Pipeline stage for enriching data with contextual information.
    
    This stage:
    - Integrates conversation history
    - Adds user profile information
    - Considers domain context
    - Calculates context relevance scores
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the context enricher stage."""
        super().__init__(name="ContextEnricher", config=config)
        
        # Configuration
        self.max_history_length = config.get('max_history_length', 10) if config else 10
        self.context_weights = config.get('context_weights', {
            'history': 0.4,
            'domain': 0.3,
            'user_profile': 0.2,
            'session': 0.1
        }) if config else {
            'history': 0.4,
            'domain': 0.3,
            'user_profile': 0.2,
            'session': 0.1
        }
        
        # Enable persistence
        self.enable_persistence = config.get('enable_persistence', True) if config else True
        self.persistence_service = None
        
        # Internal conversation history (per session) - fallback for when persistence is disabled
        self.conversation_history = deque(maxlen=self.max_history_length)
    
    async def _initialize(self):
        """Initialize context enricher with optional persistence."""
        if self.enable_persistence:
            try:
                self.persistence_service = ContextPersistenceService()
                await self.persistence_service.initialize()
                self.logger.info("Context enricher initialized with persistence")
            except Exception as e:
                self.logger.warning(f"Failed to initialize persistence service: {e}")
                self.logger.warning("Falling back to in-memory context management")
                self.persistence_service = None
                self.enable_persistence = False
        else:
            self.logger.debug("Context enricher initialized without persistence")
    
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Enrich the pipeline data with contextual information.
        
        Args:
            data: Pipeline data containing processed information
            
        Returns:
            Pipeline data enriched with context
        """
        try:
            # Get the original context passed with the query
            original_context = data.context or {}
            
            # Get classified intents from previous stage
            intents = data.get_stage_result('IntentClassifier', 'classified_intents', [])
            
            # Get the normalized query
            normalized_query = data.get_stage_result('TextPreprocessor', 'normalized_text', '')
            
            self.logger.debug(f"Enriching context for query: {normalized_query}")
            
            # If persistence is enabled, get enriched context from service
            if self.persistence_service and self.enable_persistence:
                # Get or create session and user
                session_id = original_context.get('session_id')
                user_id = original_context.get('user_id')
                domain = original_context.get('domain', 'general')
                
                # Get or create session
                session = await self.persistence_service.get_or_create_session(
                    session_id=session_id,
                    user_id=user_id,
                    domain=domain
                )
                
                # Update context with session info
                original_context['session_id'] = session['session_id']
                if session.get('user_id'):
                    original_context['user_id'] = session['user_id']
                
                # Get enriched context from persistence service
                enriched_from_db = await self.persistence_service.get_enriched_context(
                    session_id=session['session_id'],
                    user_id=session.get('user_id')
                )
                
                # Extract context components from persistent data (handle None case)
                if enriched_from_db:
                    history_context = enriched_from_db.get('history', [])
                    domain_context = {'name': enriched_from_db.get('domain', 'general')}
                    user_context = enriched_from_db.get('user_profile', {})
                    session_context = enriched_from_db.get('session', {})
                else:
                    # Fallback if persistence returns None
                    history_context = []
                    domain_context = {'name': 'general'}
                    user_context = {}
                    session_context = {}
            else:
                # Fallback to original extraction methods
                history_context = self._extract_history_context(original_context)
                domain_context = self._extract_domain_context(original_context)
                user_context = self._extract_user_context(original_context)
                session_context = self._extract_session_context(original_context)
            
            # Calculate context relevance score
            context_score = self._calculate_context_relevance(
                history_context, domain_context, user_context, session_context
            )
            
            # Enrich intents with context
            enriched_intents = self._enrich_intents_with_context(
                intents, history_context, domain_context
            )
            
            # Find related previous queries
            related_queries = self._find_related_queries(normalized_query, history_context)
            
            # Build enriched context
            enriched_context = {
                'original_context': original_context,
                'history': history_context,
                'domain': domain_context,
                'user': user_context,
                'session': session_context,
                'context_score': context_score,
                'related_queries': related_queries,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update conversation history
            await self._update_conversation_history(normalized_query, intents, enriched_context)
            
            # Store results
            data.add_stage_result(self.name, 'enriched_context', enriched_context)
            data.add_stage_result(self.name, 'context_score', context_score)
            data.add_stage_result(self.name, 'enriched_intents', enriched_intents)
            
            # Update pipeline context
            data.context.update(enriched_context)
            
            # Add to metadata
            data.add_metadata('context_score', context_score)
            data.add_metadata('has_history', bool(history_context))
            
            self.logger.debug(f"Context enrichment complete. Score: {context_score:.2f}")
            
            return data
            
        except Exception as e:
            # Log the error with full traceback for debugging
            import traceback
            self.logger.error(f"Error in ContextEnricher: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Provide minimal fallback data so pipeline can continue
            data.add_stage_result(self.name, 'enriched_context', {'original_context': data.context or {}})
            data.add_stage_result(self.name, 'context_score', 0.5)
            data.add_stage_result(self.name, 'enriched_intents', intents if 'intents' in locals() else [])
            
            return data
    
    def _extract_history_context(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conversation history from context."""
        # If we're using persistence service, history is already loaded
        if self.persistence_service and self.enable_persistence:
            # History should already be in the context from persistence service
            return context.get('history', [])
            
        # Fallback to original method
        history = context.get('history', [])
        
        # Add internal history if available
        if self.conversation_history:
            internal_history = list(self.conversation_history)
            # Merge with provided history, avoiding duplicates
            for item in internal_history:
                if item not in history:
                    history.append(item)
        
        return history[-self.max_history_length:]  # Limit history length
    
    def _extract_domain_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract domain-specific context."""
        domain = context.get('domain', {})
        
        # If domain is a string, convert to dict
        if isinstance(domain, str):
            domain = {'name': domain}
        
        # Add default domain info if not present
        if 'name' not in domain:
            domain['name'] = 'general'
        
        return domain
    
    def _extract_user_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user profile information."""
        user_profile = context.get('user_profile', {})
        
        # Add default user info if not present
        if 'preferences' not in user_profile:
            user_profile['preferences'] = {}
        
        if 'expertise_level' not in user_profile:
            user_profile['expertise_level'] = 'intermediate'
        
        return user_profile
    
    def _extract_session_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract session-specific context."""
        session = context.get('session', {})
        
        # Add session metadata
        if 'id' not in session:
            session['id'] = context.get('session_id', 'default')
        
        if 'start_time' not in session:
            session['start_time'] = datetime.now().isoformat()
        
        return session
    
    def _calculate_context_relevance(self, history: List[Dict], domain: Dict,
                                   user: Dict, session: Dict) -> float:
        """
        Calculate overall context relevance score.
        
        Args:
            history: Conversation history
            domain: Domain context
            user: User profile
            session: Session context
            
        Returns:
            Relevance score between 0 and 1
        """
        scores = {
            'history': 1.0 if history else 0.0,
            'domain': 1.0 if domain and domain.get('name') != 'general' else 0.5,
            'user_profile': 1.0 if user and user.get('preferences') else 0.5,
            'session': 1.0 if session and session.get('id') != 'default' else 0.5
        }
        
        # Weighted average
        relevance_score = sum(
            self.context_weights.get(key, 0.25) * value
            for key, value in scores.items()
        )
        
        return min(relevance_score, 1.0)
    
    def _enrich_intents_with_context(self, intents: List[Any], 
                                    history: List[Dict], domain: Dict) -> List[Any]:
        """
        Enrich intents with contextual information.
        
        Args:
            intents: List of classified intents
            history: Conversation history
            domain: Domain context
            
        Returns:
            Enriched intents
        """
        enriched_intents = []
        
        for intent in intents:
            # Safely get intent type (handles both Intent objects and dicts)
            intent_type = None
            if hasattr(intent, 'type'):
                intent_type = intent.type
            elif isinstance(intent, dict) and 'type' in intent:
                intent_type = intent['type']
            
            if not intent_type:
                continue
            
            # Check if intent type matches recent history
            history_boost = 0.0
            for hist_item in history[-3:]:  # Look at last 3 items
                if hist_item.get('intent_type') == intent_type:
                    history_boost = 0.1  # 10% boost for repeated intent type
                    break
            
            # Check if intent matches domain
            domain_boost = 0.0
            if domain.get('name') and domain.get('name') in intent_type:
                domain_boost = 0.1  # 10% boost for domain match
            
            # Apply boosts safely
            if hasattr(intent, 'confidence'):
                intent.confidence = min(intent.confidence + history_boost + domain_boost, 1.0)
            elif isinstance(intent, dict) and 'confidence' in intent:
                intent['confidence'] = min(intent['confidence'] + history_boost + domain_boost, 1.0)
            
            enriched_intents.append(intent)
        
        return enriched_intents
    
    def _find_related_queries(self, current_query: str, 
                            history: List[Dict]) -> List[Dict[str, Any]]:
        """
        Find related queries from history.
        
        Args:
            current_query: Current normalized query
            history: Conversation history
            
        Returns:
            List of related queries with similarity scores
        """
        related = []
        current_words = set(current_query.lower().split())
        
        for hist_item in history:
            hist_query = hist_item.get('query', '').lower()
            hist_words = set(hist_query.split())
            
            # Calculate simple word overlap
            overlap = len(current_words & hist_words)
            total = len(current_words | hist_words)
            
            if total > 0:
                similarity = overlap / total
                if similarity > 0.3:  # Threshold for relatedness
                    related.append({
                        'query': hist_item.get('query'),
                        'intent': hist_item.get('intent_type'),
                        'similarity': similarity,
                        'timestamp': hist_item.get('timestamp')
                    })
        
        # Sort by similarity
        related.sort(key=lambda x: x['similarity'], reverse=True)
        
        return related[:3]  # Return top 3 related queries
    
    async def _update_conversation_history(self, query: str, intents: List[Any], 
                                          context: Dict[str, Any]):
        """
        Update conversation history (both in-memory and persistent).
        
        Args:
            query: The processed query
            intents: Classified intents
            context: Enriched context
        """
        history_item = {
            'query': query,
            'intent_type': self._safe_get_intent_attribute(intents[0] if intents else None, 'type', 'unknown'),
            'confidence': self._safe_get_intent_attribute(intents[0] if intents else None, 'confidence', 0.0),
            'timestamp': datetime.now().isoformat(),
            'domain': context.get('domain', {}).get('name', 'general') if context else 'general'
        }
        
        # Always update in-memory history
        self.conversation_history.append(history_item)
        
        # If persistence is enabled, save to database
        if self.persistence_service and self.enable_persistence:
            session_id = context.get('original_context', {}).get('session_id')
            if session_id:
                # Create a simple intent result object for persistence
                class SimpleIntentResult:
                    def __init__(self, intent):
                        self.primary_intent = intent
                        self.normalized_query = query.lower()
                
                intent_result = SimpleIntentResult(intents[0] if intents else None)
                
                # Save conversation turn asynchronously
                try:
                    await self.persistence_service.save_conversation_turn(
                        session_id=session_id,
                        query=query,
                        intent_result=intent_result,
                        context=context
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to persist conversation turn: {e}")
    
    async def validate_input(self, data: PipelineData) -> bool:
        """Validate that input contains required data."""
        # Context enricher can work with minimal data
        return True
    
    def _safe_get_intent_attribute(self, intent, attr_name, default_value):
        """
        Safely get attribute from intent (handles both objects and dicts).
        
        Args:
            intent: Intent object or dictionary
            attr_name: Attribute/key name to get
            default_value: Default value if attribute not found
            
        Returns:
            Attribute value or default
        """
        if intent is None:
            return default_value
        
        # Try object attribute
        if hasattr(intent, attr_name):
            return getattr(intent, attr_name)
        
        # Try dictionary key
        if isinstance(intent, dict) and attr_name in intent:
            return intent[attr_name]
        
        return default_value