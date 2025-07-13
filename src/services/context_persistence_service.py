"""
Context Persistence Service for managing all context-related data.

This service provides a centralized interface for persisting and retrieving
user profiles, sessions, and conversation history.
"""

import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from src.database.context_models import ContextDatabase
from src.utils.logger import get_logger


class ContextPersistenceService:
    """
    Service for managing persistent context data.
    
    This service handles:
    - User profile management
    - Session lifecycle management
    - Conversation history persistence
    - Context retrieval and aggregation
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the context persistence service."""
        self.logger = get_logger(__name__)
        self.db = ContextDatabase(db_path)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the service and database."""
        if not self._initialized:
            await self.db.initialize()
            self._initialized = True
            self.logger.info("Context persistence service initialized")
    
    # User Management
    
    async def get_or_create_user(self, user_id: Optional[str] = None, 
                                username: Optional[str] = None,
                                email: Optional[str] = None) -> Dict[str, Any]:
        """Get existing user or create new one."""
        if not user_id:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # Try to get existing user
        user = await self.db.get_user(user_id)
        if user:
            # Update last active time
            await self.db.update_user(user_id, last_active=datetime.now().isoformat())
            return user
        
        # Create new user
        await self.db.create_user(user_id, username, email)
        user = await self.db.get_user(user_id)
        self.logger.info(f"Created new user: {user_id}")
        
        return user
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        user = await self.db.get_user(user_id)
        if not user:
            self.logger.warning(f"User not found: {user_id}")
            return False
        
        # Merge with existing preferences
        current_prefs = user.get('preferences', {})
        current_prefs.update(preferences)
        
        return await self.db.update_user(user_id, preferences=current_prefs)
    
    async def update_user_expertise(self, user_id: str, expertise_level: str) -> bool:
        """Update user expertise level."""
        valid_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        if expertise_level not in valid_levels:
            self.logger.warning(f"Invalid expertise level: {expertise_level}")
            return False
        
        return await self.db.update_user(user_id, expertise_level=expertise_level)
    
    # Session Management
    
    async def create_session(self, user_id: Optional[str] = None, 
                           domain: str = 'general') -> str:
        """Create a new session and return session ID."""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        await self.db.create_session(session_id, user_id, domain)
        self.logger.info(f"Created session: {session_id} for user: {user_id}")
        
        return session_id
    
    async def get_or_create_session(self, session_id: Optional[str] = None,
                                  user_id: Optional[str] = None,
                                  domain: str = 'general') -> Dict[str, Any]:
        """Get existing session or create new one."""
        if session_id:
            session = await self.db.get_session(session_id)
            if session and session.get('is_active'):
                return session
        
        # Create new session
        session_id = await self.create_session(user_id, domain)
        return await self.db.get_session(session_id)
    
    async def update_session_context(self, session_id: str, context_updates: Dict[str, Any]) -> bool:
        """Update session context with new information."""
        session = await self.db.get_session(session_id)
        if not session:
            self.logger.warning(f"Session not found: {session_id}")
            return False
        
        # Merge with existing context
        current_context = session.get('context', {})
        current_context.update(context_updates)
        
        return await self.db.update_session_context(session_id, current_context)
    
    async def end_session(self, session_id: str) -> bool:
        """End a session."""
        return await self.db.end_session(session_id)
    
    # Conversation History
    
    async def save_conversation_turn(self, 
                                   session_id: str,
                                   query: str,
                                   intent_result: Optional[Any] = None,
                                   execution_result: Optional[Any] = None,
                                   context: Optional[Dict[str, Any]] = None) -> int:
        """Save a complete conversation turn."""
        # Extract user_id from session
        session = await self.db.get_session(session_id)
        user_id = session.get('user_id') if session else None
        
        # Extract intent information
        intent_type = None
        intent_confidence = None
        normalized_query = query.lower()  # Default normalization
        
        if intent_result:
            intent_type = getattr(intent_result.primary_intent, 'type', None)
            intent_confidence = getattr(intent_result.primary_intent, 'confidence', None)
            normalized_query = getattr(intent_result, 'normalized_query', normalized_query)
        
        # Extract execution information
        tools_discovered = []
        tools_selected = []
        execution_success = None
        execution_time_ms = None
        
        if execution_result:
            tools_discovered = getattr(execution_result, 'discovered_tools', [])
            tools_selected = getattr(execution_result, 'selected_tools', [])
            execution_success = getattr(execution_result, 'success', None)
            execution_time_ms = getattr(execution_result, 'execution_time_ms', None)
        
        # Add to conversation history
        entry_id = await self.db.add_conversation_entry(
            session_id=session_id,
            user_id=user_id,
            query=query,
            normalized_query=normalized_query,
            intent_type=intent_type,
            intent_confidence=intent_confidence,
            tools_discovered=tools_discovered,
            tools_selected=tools_selected,
            execution_success=execution_success,
            execution_time_ms=execution_time_ms,
            context=context
        )
        
        self.logger.debug(f"Saved conversation turn {entry_id} for session {session_id}")
        
        return entry_id
    
    async def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's conversation history."""
        return await self.db.get_conversation_history(user_id=user_id, limit=limit)
    
    async def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get session's conversation history."""
        return await self.db.get_conversation_history(session_id=session_id, limit=limit)
    
    # Context Aggregation
    
    async def get_enriched_context(self, 
                                 session_id: Optional[str] = None,
                                 user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get enriched context combining user profile, session, and history.
        
        Returns:
            Dictionary containing:
            - user_profile: User preferences and expertise
            - session: Current session information
            - history: Recent conversation history
            - domain: Current domain context
            - statistics: User statistics
        """
        context = {
            'user_profile': None,
            'session': None,
            'history': [],
            'domain': 'general',
            'statistics': None
        }
        
        # Get user profile
        if user_id:
            user = await self.db.get_user(user_id)
            if user:
                context['user_profile'] = {
                    'user_id': user['user_id'],
                    'preferences': user['preferences'],
                    'expertise_level': user['expertise_level']
                }
                
                # Get user statistics
                context['statistics'] = await self.db.get_user_statistics(user_id)
        
        # Get session info
        if session_id:
            session = await self.db.get_session(session_id)
            if session:
                context['session'] = {
                    'session_id': session['session_id'],
                    'domain': session['domain'],
                    'start_time': session['start_time'],
                    'context': session['context']
                }
                context['domain'] = session['domain']
                
                # Use user_id from session if not provided
                if not user_id and session.get('user_id'):
                    user_id = session['user_id']
                    user = await self.db.get_user(user_id)
                    if user:
                        context['user_profile'] = {
                            'user_id': user['user_id'],
                            'preferences': user['preferences'],
                            'expertise_level': user['expertise_level']
                        }
        
        # Get conversation history
        if session_id:
            context['history'] = await self.get_session_history(session_id, limit=10)
        elif user_id:
            context['history'] = await self.get_user_history(user_id, limit=10)
        
        return context
    
    # Learning and Adaptation
    
    async def learn_from_feedback(self, 
                                session_id: str,
                                feedback: Dict[str, Any]) -> bool:
        """
        Process user feedback to improve future interactions.
        
        Args:
            session_id: Current session ID
            feedback: Feedback data including rating, helpful flags, etc.
        """
        session = await self.db.get_session(session_id)
        if not session:
            self.logger.warning(f"Session not found for feedback: {session_id}")
            return False
        
        user_id = session.get('user_id')
        if not user_id:
            self.logger.warning("No user associated with session for feedback learning")
            return True
        
        # Get recent conversation entry
        history = await self.get_session_history(session_id, limit=1)
        if not history:
            self.logger.warning("No conversation history found for feedback")
            return True
        
        recent_entry = history[0]
        
        # Update user preferences based on feedback
        if feedback.get('helpful', False) and recent_entry.get('intent_type'):
            # Track successful intent types
            user = await self.db.get_user(user_id)
            if user:
                prefs = user.get('preferences', {})
                successful_intents = prefs.get('successful_intents', {})
                intent_type = recent_entry['intent_type']
                
                successful_intents[intent_type] = successful_intents.get(intent_type, 0) + 1
                prefs['successful_intents'] = successful_intents
                
                await self.db.update_user(user_id, preferences=prefs)
                self.logger.info(f"Updated user preferences based on positive feedback")
        
        return True
    
    # Cleanup and Maintenance
    
    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Clean up inactive sessions older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # This would need to be implemented in the database layer
        # For now, return 0
        self.logger.info(f"Session cleanup not yet implemented")
        return 0