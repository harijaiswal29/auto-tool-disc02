"""
Tests for context persistence functionality.

This module tests the database models, persistence service, and
integration with the Intent Recognition Agent.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from src.database.context_models import ContextDatabase
from src.services.context_persistence_service import ContextPersistenceService
from src.agents.intent_recognition_agent import IntentRecognitionAgent


class TestContextDatabase:
    """Test database operations."""
    
    @pytest.fixture
    async def db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db = ContextDatabase(db_path)
        await db.initialize()
        
        yield db
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_user_crud(self, db):
        """Test user creation and retrieval."""
        # Create user
        success = await db.create_user(
            user_id="test_user_1",
            username="testuser",
            email="test@example.com",
            preferences={"theme": "dark"}
        )
        assert success is True
        
        # Retrieve user
        user = await db.get_user("test_user_1")
        assert user is not None
        assert user['username'] == "testuser"
        assert user['email'] == "test@example.com"
        assert user['preferences']['theme'] == "dark"
        
        # Update user
        success = await db.update_user(
            "test_user_1",
            expertise_level="advanced",
            preferences={"theme": "light", "lang": "en"}
        )
        assert success is True
        
        # Verify update
        user = await db.get_user("test_user_1")
        assert user['expertise_level'] == "advanced"
        assert user['preferences']['theme'] == "light"
        assert user['preferences']['lang'] == "en"
    
    @pytest.mark.asyncio
    async def test_session_crud(self, db):
        """Test session creation and management."""
        # Create session
        success = await db.create_session(
            session_id="test_session_1",
            user_id="test_user_1",
            domain="engineering"
        )
        assert success is True
        
        # Retrieve session
        session = await db.get_session("test_session_1")
        assert session is not None
        assert session['user_id'] == "test_user_1"
        assert session['domain'] == "engineering"
        assert session['is_active'] is True
        
        # Update session context
        context = {"current_task": "code_review"}
        success = await db.update_session_context("test_session_1", context)
        assert success is True
        
        # Verify context update
        session = await db.get_session("test_session_1")
        assert session['context']['current_task'] == "code_review"
        
        # End session
        success = await db.end_session("test_session_1")
        assert success is True
        
        # Verify session ended
        session = await db.get_session("test_session_1")
        assert session['is_active'] is False
        assert session['end_time'] is not None
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, db):
        """Test conversation history storage."""
        # Add conversation entry
        entry_id = await db.add_conversation_entry(
            session_id="test_session_1",
            user_id="test_user_1",
            query="Find all Python files",
            normalized_query="find all python files",
            intent_type="query.search",
            intent_confidence=0.85,
            tools_discovered=["filesystem_mcp", "git_mcp"],
            tools_selected=["filesystem_mcp"],
            execution_success=True,
            execution_time_ms=250.5
        )
        assert entry_id > 0
        
        # Retrieve history by session
        history = await db.get_conversation_history(session_id="test_session_1")
        assert len(history) == 1
        assert history[0]['query'] == "Find all Python files"
        assert history[0]['intent_type'] == "query.search"
        assert history[0]['tools_selected'] == ["filesystem_mcp"]
        
        # Retrieve history by user
        history = await db.get_conversation_history(user_id="test_user_1")
        assert len(history) == 1


class TestContextPersistenceService:
    """Test the persistence service."""
    
    @pytest.fixture
    async def service(self):
        """Create a temporary persistence service."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        service = ContextPersistenceService(db_path)
        await service.initialize()
        
        yield service
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_user_management(self, service):
        """Test user management through service."""
        # Get or create user
        user = await service.get_or_create_user(username="testuser")
        assert user is not None
        assert user['username'] == "testuser"
        user_id = user['user_id']
        
        # Update preferences
        success = await service.update_user_preferences(
            user_id,
            {"notifications": True, "language": "en"}
        )
        assert success is True
        
        # Update expertise
        success = await service.update_user_expertise(user_id, "expert")
        assert success is True
        
        # Verify updates
        user = await service.db.get_user(user_id)
        assert user['preferences']['notifications'] is True
        assert user['expertise_level'] == "expert"
    
    @pytest.mark.asyncio
    async def test_session_management(self, service):
        """Test session management through service."""
        # Create user
        user = await service.get_or_create_user(username="testuser")
        user_id = user['user_id']
        
        # Create session
        session_id = await service.create_session(user_id, domain="science")
        assert session_id is not None
        assert session_id.startswith("session_")
        
        # Get or create session (should return existing)
        session = await service.get_or_create_session(
            session_id=session_id,
            user_id=user_id
        )
        assert session['session_id'] == session_id
        assert session['domain'] == "science"
        
        # Update session context
        success = await service.update_session_context(
            session_id,
            {"experiment": "quantum_computing"}
        )
        assert success is True
    
    @pytest.mark.asyncio
    async def test_conversation_persistence(self, service):
        """Test conversation turn persistence."""
        # Create session
        session_id = await service.create_session(domain="general")
        
        # Create mock intent result
        class MockIntent:
            def __init__(self):
                self.type = "query.search"
                self.confidence = 0.9
        
        class MockIntentResult:
            def __init__(self):
                self.primary_intent = MockIntent()
                self.normalized_query = "find python files"
        
        # Save conversation turn
        entry_id = await service.save_conversation_turn(
            session_id=session_id,
            query="Find Python files",
            intent_result=MockIntentResult()
        )
        assert entry_id > 0
        
        # Retrieve history
        history = await service.get_session_history(session_id)
        assert len(history) == 1
        assert history[0]['query'] == "Find Python files"
        assert history[0]['intent_type'] == "query.search"
    
    @pytest.mark.asyncio
    async def test_enriched_context(self, service):
        """Test enriched context retrieval."""
        # Create user and session
        user = await service.get_or_create_user(username="testuser")
        user_id = user['user_id']
        session_id = await service.create_session(user_id, domain="engineering")
        
        # Add some history
        class MockIntent:
            def __init__(self, intent_type):
                self.type = intent_type
                self.confidence = 0.8
        
        class MockIntentResult:
            def __init__(self, intent_type):
                self.primary_intent = MockIntent(intent_type)
                self.normalized_query = "test query"
        
        await service.save_conversation_turn(
            session_id, "Test query 1", MockIntentResult("query.search")
        )
        await service.save_conversation_turn(
            session_id, "Test query 2", MockIntentResult("action.create")
        )
        
        # Get enriched context
        context = await service.get_enriched_context(session_id, user_id)
        
        assert context['user_profile'] is not None
        assert context['user_profile']['user_id'] == user_id
        assert context['session'] is not None
        assert context['session']['domain'] == "engineering"
        assert len(context['history']) == 2
        assert context['domain'] == "engineering"


class TestIntentRecognitionWithPersistence:
    """Test Intent Recognition Agent with persistence enabled."""
    
    @pytest.fixture
    async def agent_with_persistence(self):
        """Create agent with temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Create agent with persistence enabled
        config = {
            'enable_persistence': True,
            'enable_state_tracking': True
        }
        
        agent = IntentRecognitionAgent(config)
        
        # Initialize persistence service manually for testing
        agent.persistence_service = ContextPersistenceService(db_path)
        await agent.persistence_service.initialize()
        
        yield agent
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_process_query_with_persistence(self, agent_with_persistence):
        """Test query processing with full persistence."""
        agent = agent_with_persistence
        
        # Create user
        user = await agent.get_or_create_user(username="test_developer")
        assert user is not None
        user_id = user['user_id']
        
        # Process query with persistence
        result = await agent.process_query_with_persistence(
            query="Find all Python files in the project",
            user_id=user_id,
            domain="development"
        )
        
        assert result is not None
        assert result.primary_intent is not None
        assert result.confidence_passed is True
        
        # Verify persistence
        if agent.persistence_service:
            # Get user statistics
            stats = await agent.get_user_statistics(user_id)
            assert stats is not None
            assert stats['total_queries'] == 1
    
    @pytest.mark.asyncio
    async def test_session_continuity(self, agent_with_persistence):
        """Test session continuity across multiple queries."""
        agent = agent_with_persistence
        
        # Create session
        session_id = await agent.create_session(domain="engineering")
        assert session_id is not None
        
        # Process multiple queries in same session
        queries = [
            "Find Python files",
            "Create a new test file",
            "Analyze code quality"
        ]
        
        for query in queries:
            result = await agent.process_query_with_persistence(
                query=query,
                session_id=session_id
            )
            assert result is not None
        
        # Verify session history
        if agent.persistence_service:
            history = await agent.persistence_service.get_session_history(session_id)
            assert len(history) == 3
            
            # Verify queries are in reverse chronological order
            assert history[0]['query'] == "Analyze code quality"
            assert history[2]['query'] == "Find Python files"
    
    @pytest.mark.asyncio 
    async def test_feedback_learning(self, agent_with_persistence):
        """Test learning from user feedback."""
        agent = agent_with_persistence
        
        # Create user and session
        user = await agent.get_or_create_user(username="learner")
        session_id = await agent.create_session(user_id=user['user_id'])
        
        # Process a query
        result = await agent.process_query_with_persistence(
            query="Search for documentation",
            session_id=session_id
        )
        
        # Provide positive feedback
        feedback = {
            'helpful': True,
            'rating': 5,
            'comment': "Found exactly what I needed"
        }
        
        success = await agent.learn_from_feedback(session_id, feedback)
        assert success is True
        
        # Verify preferences were updated
        if agent.persistence_service:
            updated_user = await agent.persistence_service.db.get_user(user['user_id'])
            prefs = updated_user.get('preferences', {})
            successful_intents = prefs.get('successful_intents', {})
            
            # Should have recorded the successful intent type
            assert len(successful_intents) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])