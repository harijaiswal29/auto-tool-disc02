"""
Integration tests for Orchestrator Agent caching functionality.

Tests the caching behavior in the context of the full orchestration pipeline:
- Cache hits for repeated queries
- Context-aware caching
- Performance improvements
- Cache invalidation on tool updates
- Metrics collection
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

from src.agents.orchestrator_agent import OrchestratorAgent, OrchestrationResult, ToolExecutionResult
from src.agents.intent_models import Intent, IntentResult


class TestOrchestratorCaching:
    """Integration tests for Orchestrator caching functionality."""
    
    @pytest.fixture
    def cache_config(self):
        """Configuration with caching enabled."""
        return {
            'orchestration': {
                'max_tools_per_query': 2,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'result_cache': {
                'enabled': True,
                'max_size': 10,
                'ttl_seconds': 300,  # 5 minutes
                'cache_successful_only': True,
                'consider_context': True,
                'enable_persistence': False
            },
            'intent_recognition': {
                'model': 'test-model',
                'confidence_threshold': 0.7
            }
        }
    
    @pytest.fixture
    async def orchestrator_with_cache(self, cache_config):
        """Create orchestrator with caching enabled and mocked dependencies."""
        with patch('src.agents.orchestrator_agent.IntentRecognitionAgent') as mock_intent:
            with patch('src.agents.orchestrator_agent.MCPIntegration') as mock_mcp:
                with patch('src.agents.orchestrator_agent.ToolRegistry') as mock_registry:
                    with patch('src.agents.orchestrator_agent.ConversationStateMachine') as mock_state:
                        # Create orchestrator
                        orchestrator = OrchestratorAgent(config=cache_config)
                        
                        # Setup mocks
                        orchestrator.intent_agent = mock_intent.return_value
                        orchestrator.mcp_integration = mock_mcp.return_value
                        orchestrator.tool_registry = mock_registry.return_value
                        orchestrator.state_machine = mock_state.return_value
                        
                        # Configure mock methods
                        orchestrator.intent_agent.process_query = AsyncMock()
                        orchestrator.mcp_integration.initialize = AsyncMock()
                        orchestrator.mcp_integration.shutdown = AsyncMock()
                        orchestrator.mcp_integration.execute_tool = AsyncMock()
                        orchestrator.tool_registry.search_tools = AsyncMock()
                        orchestrator.tool_registry.get_tool_relationships = AsyncMock(return_value=[])
                        
                        # Configure state machine
                        orchestrator.state_machine.receive_query = AsyncMock(return_value=True)
                        orchestrator.state_machine.recognize_intent = AsyncMock()
                        orchestrator.state_machine.discover_tools = AsyncMock()
                        orchestrator.state_machine.start_execution = AsyncMock()
                        orchestrator.state_machine.complete_execution = AsyncMock()
                        orchestrator.state_machine.return_to_idle = AsyncMock()
                        orchestrator.state_machine.is_in_state = Mock(return_value=True)
                        orchestrator.state_machine.start = AsyncMock()
                        
                        # Initialize
                        await orchestrator.initialize()
                        
                        yield orchestrator
                        
                        # Cleanup
                        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_cache_hit_for_repeated_query(self, orchestrator_with_cache):
        """Test that repeated queries hit the cache."""
        orchestrator = orchestrator_with_cache
        
        # Setup mock responses
        mock_intent = IntentResult(
            raw_query="Find Python files",
            primary_intent=Intent(
                type='query.search',
                keywords=['find', 'python', 'files'],
                confidence=0.85
            ),
            all_intents=[],
            processed_query="find python files",
            metadata={'processing_time_ms': 50.0}
        )
        
        mock_tools = [
            {
                'id': 'filesystem.search',
                'name': 'File Search',
                'type': 'filesystem',
                'performance_score': 0.9,
                'capabilities': '{"operations": ["search"]}'
            }
        ]
        
        orchestrator.intent_agent.process_query.return_value = mock_intent
        orchestrator.tool_registry.search_tools.return_value = mock_tools
        orchestrator.mcp_integration.execute_tool.return_value = {
            'files': ['file1.py', 'file2.py'],
            'success': True
        }
        
        # First query - should process normally
        start_time = time.time()
        result1 = await orchestrator.process_user_query("Find Python files")
        first_query_time = time.time() - start_time
        
        assert result1.success
        assert len(result1.execution_results) > 0
        
        # Verify cache miss (first query)
        cache_metrics = orchestrator.get_cache_metrics()
        assert cache_metrics['misses'] == 1
        assert cache_metrics['hits'] == 0
        
        # Second identical query - should hit cache
        start_time = time.time()
        result2 = await orchestrator.process_user_query("Find Python files")
        cached_query_time = time.time() - start_time
        
        # Verify cache hit
        cache_metrics = orchestrator.get_cache_metrics()
        assert cache_metrics['hits'] == 1
        assert cache_metrics['misses'] == 1
        assert cache_metrics['hit_rate'] == 0.5
        
        # Cached query should be much faster
        assert cached_query_time < first_query_time * 0.1  # At least 10x faster
        
        # Results should be identical
        assert result2.query == result1.query
        assert result2.success == result1.success
        assert len(result2.execution_results) == len(result1.execution_results)
    
    @pytest.mark.asyncio
    async def test_context_aware_caching(self, orchestrator_with_cache):
        """Test that cache considers context for key generation."""
        orchestrator = orchestrator_with_cache
        
        # Setup mock
        mock_intent = IntentResult(
            raw_query="Get data",
            primary_intent=Intent(
                type='query.retrieve',
                keywords=['get', 'data'],
                confidence=0.9
            ),
            all_intents=[],
            processed_query="get data",
            metadata={'processing_time_ms': 30.0}
        )
        
        orchestrator.intent_agent.process_query.return_value = mock_intent
        orchestrator.tool_registry.search_tools.return_value = [{
            'id': 'db.query',
            'name': 'Database Query',
            'type': 'database',
            'performance_score': 0.85,
            'capabilities': '{"operations": ["query", "retrieve", "get"]}'
        }]
        orchestrator.mcp_integration.execute_tool.return_value = {
            'data': 'result',
            'success': True
        }
        
        # Query with different contexts
        context1 = {'domain': 'finance', 'user_expertise': 'expert'}
        context2 = {'domain': 'healthcare', 'user_expertise': 'expert'}
        
        # First query with context1
        result1 = await orchestrator.process_user_query("Get data", context1)
        assert result1.success
        
        # Same query with different context - should not hit cache
        result2 = await orchestrator.process_user_query("Get data", context2)
        assert result2.success
        
        cache_metrics = orchestrator.get_cache_metrics()
        assert cache_metrics['hits'] == 0  # No cache hits
        assert cache_metrics['misses'] == 2  # Two different contexts
        
        # Same query with same context - should hit cache
        result3 = await orchestrator.process_user_query("Get data", context1)
        
        cache_metrics = orchestrator.get_cache_metrics()
        assert cache_metrics['hits'] == 1  # Now we have a hit
        assert cache_metrics['misses'] == 2
    
    @pytest.mark.asyncio
    async def test_cache_only_successful_results(self, orchestrator_with_cache):
        """Test that only successful results are cached."""
        orchestrator = orchestrator_with_cache
        
        # Setup for failed execution
        mock_intent = IntentResult(
            raw_query="Bad query",
            primary_intent=Intent(
                type='query.search',
                keywords=['bad', 'query'],
                confidence=0.8
            ),
            all_intents=[],
            processed_query="bad query",
            metadata={'processing_time_ms': 25.0}
        )
        
        orchestrator.intent_agent.process_query.return_value = mock_intent
        orchestrator.tool_registry.search_tools.return_value = [{
            'id': 'tool.failing',
            'name': 'Failing Tool',
            'type': 'test',
            'performance_score': 0.7,
            'capabilities': '{"operations": ["search", "query"]}'
        }]
        
        # Make tool execution fail
        orchestrator.mcp_integration.execute_tool.side_effect = Exception("Tool failed")
        
        # First query - fails
        result1 = await orchestrator.process_user_query("Bad query")
        assert not result1.success
        
        # Second identical query - should not hit cache (failed results not cached)
        result2 = await orchestrator.process_user_query("Bad query")
        assert not result2.success
        
        cache_metrics = orchestrator.get_cache_metrics()
        assert cache_metrics['hits'] == 0  # No hits because failures aren't cached
        assert cache_metrics['current_size'] == 0  # Nothing in cache
    
    @pytest.mark.asyncio
    async def test_cache_warming_from_history(self, orchestrator_with_cache):
        """Test warming cache from execution history."""
        orchestrator = orchestrator_with_cache
        
        # Add some execution history
        orchestrator.execution_history = [
            {
                'query': 'Historical query 1',
                'success': True,
                'result': OrchestrationResult(
                    query='Historical query 1',
                    intent=Mock(),
                    discovered_tools=[],
                    selected_tools=[],
                    execution_results=[],
                    total_time_ms=100,
                    success=True,
                    summary='Success'
                ),
                'context': {'domain': 'test'}
            },
            {
                'query': 'Historical query 2',
                'success': False,  # Should not be warmed
                'result': Mock()
            }
        ]
        
        # Warm cache
        await orchestrator.warm_cache_from_history()
        
        # Check cache metrics
        cache_metrics = orchestrator.get_cache_metrics()
        assert cache_metrics['current_size'] == 1  # Only successful entry warmed
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_for_tool(self, orchestrator_with_cache):
        """Test cache invalidation when a tool is updated."""
        orchestrator = orchestrator_with_cache
        
        # Setup successful query
        mock_intent = IntentResult(
            raw_query="Use specific tool",
            primary_intent=Intent(
                type='action.process',
                keywords=['use', 'specific', 'tool'],
                confidence=0.9
            ),
            all_intents=[],
            processed_query="use specific tool",
            metadata={'processing_time_ms': 40.0}
        )
        
        orchestrator.intent_agent.process_query.return_value = mock_intent
        orchestrator.tool_registry.search_tools.return_value = [{
            'id': 'tool.specific',
            'name': 'Specific Tool',
            'type': 'processor',
            'performance_score': 0.8,
            'capabilities': '{"operations": ["process", "use"]}'
        }]
        orchestrator.mcp_integration.execute_tool.return_value = {
            'result': 'processed',
            'success': True
        }
        
        # Execute query
        result1 = await orchestrator.process_user_query("Use specific tool")
        assert result1.success
        
        # Verify cached
        cache_metrics = orchestrator.get_cache_metrics()
        assert cache_metrics['current_size'] == 1
        
        # Invalidate cache for the tool
        orchestrator.invalidate_cache_for_tool('tool.specific')
        
        # Check cache is empty
        cache_metrics = orchestrator.get_cache_metrics()
        assert cache_metrics['current_size'] == 0
    
    @pytest.mark.asyncio
    async def test_cache_metrics_collection(self, orchestrator_with_cache):
        """Test comprehensive cache metrics collection."""
        orchestrator = orchestrator_with_cache
        
        # Setup
        mock_intent = IntentResult(
            raw_query="Test metrics",
            primary_intent=Intent(
                type='query.analyze',
                keywords=['test', 'metrics'],
                confidence=0.85
            ),
            all_intents=[],
            processed_query="test metrics",
            metadata={'processing_time_ms': 35.0}
        )
        
        orchestrator.intent_agent.process_query.return_value = mock_intent
        orchestrator.tool_registry.search_tools.return_value = [{
            'id': 'analyzer.metrics',
            'name': 'Metrics Analyzer',
            'type': 'analyzer',
            'performance_score': 0.85,
            'capabilities': '{"operations": ["analyze", "test", "metrics"]}'
        }]
        orchestrator.mcp_integration.execute_tool.return_value = {
            'metrics': 'data',
            'success': True
        }
        
        # Execute multiple queries
        queries = [
            "Test metrics",  # Will be cached
            "Test metrics",  # Cache hit
            "Test metrics",  # Cache hit
            "Different query",  # Cache miss
            "Test metrics"  # Cache hit
        ]
        
        for query in queries:
            await orchestrator.process_user_query(query)
        
        # Get comprehensive metrics
        metrics = orchestrator.get_cache_metrics()
        
        assert metrics['hits'] == 3
        assert metrics['misses'] == 2
        assert metrics['hit_rate'] == 0.6
        assert metrics['current_size'] == 2  # Two unique queries
        assert metrics['max_size'] == 10
        assert 'avg_retrieval_time_ms' in metrics
        assert 'top_accessed' in metrics
        
        # Check top accessed
        if metrics['top_accessed']:
            # "Test metrics" should be top with 3 hits
            top_key, hit_count = metrics['top_accessed'][0]
            assert hit_count >= 3
    
    @pytest.mark.asyncio
    async def test_cache_persistence(self, cache_config):
        """Test cache persistence across orchestrator instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Enable persistence
            cache_file = os.path.join(tmpdir, "test_orchestrator_cache.pkl")
            cache_config['result_cache']['enable_persistence'] = True
            cache_config['result_cache']['cache_file'] = cache_file
            
            # First orchestrator instance
            with patch('src.agents.orchestrator_agent.IntentRecognitionAgent'):
                with patch('src.agents.orchestrator_agent.MCPIntegration'):
                    with patch('src.agents.orchestrator_agent.ToolRegistry'):
                        with patch('src.agents.orchestrator_agent.ConversationStateMachine'):
                            orchestrator1 = OrchestratorAgent(config=cache_config)
                            
                            # Add to cache manually
                            from src.agents.orchestrator_agent import OrchestrationResult
                            result = OrchestrationResult(
                                query="Persistent query",
                                intent=Mock(),
                                discovered_tools=[],
                                selected_tools=[],
                                execution_results=[],
                                total_time_ms=100,
                                success=True,
                                summary="Success"
                            )
                            
                            key = orchestrator1.result_cache.generate_key("Persistent query")
                            orchestrator1.result_cache.put(key, result)
                            
                            # Save cache
                            orchestrator1.save_cache()
                            
                            assert os.path.exists(cache_file)
            
            # Second orchestrator instance - should load cache
            with patch('src.agents.orchestrator_agent.IntentRecognitionAgent'):
                with patch('src.agents.orchestrator_agent.MCPIntegration'):
                    with patch('src.agents.orchestrator_agent.ToolRegistry'):
                        with patch('src.agents.orchestrator_agent.ConversationStateMachine'):
                            orchestrator2 = OrchestratorAgent(config=cache_config)
                            
                            # Check if cache was loaded
                            key = orchestrator2.result_cache.generate_key("Persistent query")
                            loaded_result = orchestrator2.result_cache.get(key)
                            
                            assert loaded_result is not None
                            assert loaded_result.query == "Persistent query"
                            assert loaded_result.success is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])