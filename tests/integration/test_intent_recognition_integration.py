"""
Integration tests for the Intent Recognition Agent.

This module tests the complete Intent Recognition Agent functionality
including multi-intent handling, state management, and persistence.
"""

import pytest
import asyncio
import sys
import os
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.agents.intent_models import Intent, IntentResult


class TestIntentRecognitionIntegration:
    """Integration tests for Intent Recognition Agent."""
    
    @pytest.fixture
    async def agent(self):
        """Create an intent recognition agent for testing."""
        config = {
            'model': 'all-MiniLM-L6-v2',
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.7,
            'enable_state_tracking': True,
            'enable_persistence': False  # Disable for testing
        }
        agent = IntentRecognitionAgent(config)
        yield agent
    
    @pytest.fixture
    async def agent_with_persistence(self, tmp_path):
        """Create an agent with persistence enabled."""
        config = {
            'model': 'all-MiniLM-L6-v2',
            'similarity_threshold': 0.7,
            'confidence_threshold': 0.7,
            'enable_state_tracking': True,
            'enable_persistence': True
        }
        
        # Use temporary database
        os.environ['CONTEXT_DB_PATH'] = str(tmp_path / "test_context.db")
        
        agent = IntentRecognitionAgent(config)
        # Wait for persistence service to initialize
        await asyncio.sleep(0.1)
        yield agent
    
    @pytest.mark.asyncio
    async def test_single_intent_recognition(self, agent):
        """Test recognition of single intents."""
        test_cases = [
            {
                'query': "Find all Python files in the src directory",
                'expected_intent': "query.search",
                'min_confidence': 0.7
            },
            {
                'query': "Create a new configuration file for the project",
                'expected_intent': "action.create",
                'min_confidence': 0.7
            },
            {
                'query': "Delete all temporary log files",
                'expected_intent': "action.delete",
                'min_confidence': 0.7
            },
            {
                'query': "Show me the current system status",
                'expected_intent': "system.monitor",
                'min_confidence': 0.6
            }
        ]
        
        for test_case in test_cases:
            result = await agent.process_query(test_case['query'])
            
            assert isinstance(result, IntentResult)
            assert result.primary_intent.type == test_case['expected_intent']
            assert result.primary_intent.confidence >= test_case['min_confidence']
            assert result.confidence_passed is True
            assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_multi_intent_recognition(self, agent):
        """Test recognition of multi-intent queries."""
        query = "Search for Python files and then analyze their complexity"
        
        result = await agent.process_query(query)
        
        assert isinstance(result, IntentResult)
        assert len(result.all_intents) >= 2
        
        # Should contain both search and analyze intents
        intent_types = [intent.type for intent in result.all_intents]
        assert any('search' in t for t in intent_types)
        assert any('analyze' in t for t in intent_types)
    
    @pytest.mark.asyncio
    async def test_context_enrichment(self, agent):
        """Test context enrichment functionality."""
        context = {
            'session_id': 'test_session_123',
            'domain': 'software_development',
            'user_id': 'test_user'
        }
        
        result = await agent.process_query("Find configuration files", context)
        
        assert result.features is not None
        assert 'context_score' in result.features
        assert result.features['context_score'] >= 0.5
    
    @pytest.mark.asyncio
    async def test_state_management(self, agent):
        """Test conversation state management."""
        # Initial state should be IDLE
        assert agent.get_current_state() == 'IDLE'
        
        # Process a query
        await agent.process_query("Find Python files")
        assert agent.get_current_state() == 'QUERY_RECEIVED'
        
        # Get state history
        history = agent.get_state_history(limit=5)
        assert len(history) >= 1
        assert history[0]['from_state'] == 'IDLE'
        assert history[0]['to_state'] == 'QUERY_RECEIVED'
    
    @pytest.mark.asyncio
    async def test_low_confidence_handling(self, agent):
        """Test handling of low confidence queries."""
        # Very ambiguous query
        result = await agent.process_query("thing stuff")
        
        assert result.confidence_passed is False
        assert result.primary_intent.confidence < 0.7
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, agent):
        """Test that all features are properly extracted."""
        result = await agent.process_query("What files were modified today?")
        
        features = result.features
        assert 'tokens' in features
        assert 'keywords' in features
        assert 'semantic_scores' in features
        assert 'keyword_scores' in features
        assert 'word_count' in features
        assert 'has_question' in features
        
        # This is a question, so has_question should be True
        assert features['has_question'] is True
        assert features['word_count'] == 5
    
    @pytest.mark.asyncio
    async def test_intent_details_retrieval(self, agent):
        """Test getting details about specific intent types."""
        details = await agent.get_intent_details("query.search")
        
        assert details['type'] == 'query.search'
        assert details['category'] == 'query'
        assert details['subcategory'] == 'search'
        assert 'keywords' in details
        assert 'find' in details['keywords']
    
    @pytest.mark.asyncio
    async def test_pipeline_info(self, agent):
        """Test pipeline information retrieval."""
        info = agent.get_pipeline_info()
        
        assert info['pipeline_name'] == 'IntentRecognitionPipeline'
        assert 'stages' in info
        assert len(info['stages']) >= 6  # At least 6 stages
        assert info['config']['similarity_threshold'] == 0.7
        assert info['config']['confidence_threshold'] == 0.7
        assert info['config']['state_tracking_enabled'] is True
    
    @pytest.mark.asyncio
    async def test_error_state_recovery(self, agent):
        """Test recovery from error states."""
        # Force error state
        if agent.state_manager:
            agent.state_manager.state_machine.current_state = 'EXECUTION_FAILED'
        
        assert agent.is_in_error_state() is True
        
        # Request retry
        retry_success = await agent.request_retry()
        assert retry_success is True
        assert not agent.is_in_error_state()
    
    @pytest.mark.asyncio
    async def test_conversation_reset(self, agent):
        """Test conversation reset functionality."""
        # Process some queries
        await agent.process_query("Find files")
        await agent.process_query("Delete logs")
        
        # Reset conversation
        reset_success = await agent.reset_conversation()
        assert reset_success is True
        assert agent.get_current_state() == 'IDLE'
    
    @pytest.mark.asyncio
    async def test_persistence_integration(self, agent_with_persistence):
        """Test persistence service integration."""
        agent = agent_with_persistence
        
        # Create user and session
        user = await agent.get_or_create_user(username="test_user")
        if user:  # Only if persistence is properly initialized
            assert user['username'] == 'test_user'
            
            session_id = await agent.create_session(user['user_id'], 'testing')
            assert session_id is not None
            
            # Process query with persistence
            result = await agent.process_query_with_persistence(
                "Find Python files",
                session_id=session_id,
                user_id=user['user_id']
            )
            
            assert result.primary_intent.type == 'query.search'
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, agent):
        """Test that processing meets performance requirements."""
        queries = [
            "Find all Python files",
            "Create a new configuration",
            "Delete temporary files",
            "Update the database schema",
            "Monitor system performance"
        ]
        
        processing_times = []
        
        for query in queries:
            result = await agent.process_query(query)
            processing_times.append(result.processing_time_ms)
        
        # Average processing time should be under 100ms
        avg_time = sum(processing_times) / len(processing_times)
        assert avg_time < 100  # Performance requirement from docs
        
        # 95th percentile should be under 200ms
        sorted_times = sorted(processing_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
        assert p95_time < 200


class TestIntentRecognitionEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    async def agent(self):
        agent = IntentRecognitionAgent()
        yield agent
    
    @pytest.mark.asyncio
    async def test_empty_query(self, agent):
        """Test handling of empty queries."""
        result = await agent.process_query("")
        
        assert result.primary_intent is not None
        assert result.confidence_passed is False
    
    @pytest.mark.asyncio
    async def test_very_long_query(self, agent):
        """Test handling of very long queries."""
        long_query = " ".join(["find files"] * 100)  # 200 words
        
        result = await agent.process_query(long_query)
        
        assert result.primary_intent is not None
        assert result.processing_time_ms < 500  # Should still be reasonably fast
    
    @pytest.mark.asyncio
    async def test_special_characters_query(self, agent):
        """Test queries with special characters."""
        queries = [
            "Find files with @#$% in name",
            "Search for <script>alert('test')</script>",
            "Delete files; DROP TABLE users;--"
        ]
        
        for query in queries:
            result = await agent.process_query(query)
            assert result.primary_intent is not None
            # Should handle safely without errors
    
    @pytest.mark.asyncio
    async def test_unicode_query(self, agent):
        """Test queries with unicode characters."""
        queries = [
            "Find files named café.txt",
            "Search for 文档 files",
            "Delete файлы with 🔥 emoji"
        ]
        
        for query in queries:
            result = await agent.process_query(query)
            assert result.primary_intent is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, agent):
        """Test handling of concurrent queries."""
        queries = [
            "Find Python files",
            "Create new configuration",
            "Delete old logs",
            "Update settings",
            "Check status"
        ]
        
        # Process queries concurrently
        tasks = [agent.process_query(q) for q in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(queries)
        assert all(isinstance(r, IntentResult) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])