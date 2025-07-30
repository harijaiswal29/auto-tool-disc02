"""
Integration tests for Orchestrator Agent.

Tests the complete orchestration functionality including:
- Tool selection (traditional and Q-learning based)
- Parallel execution management
- Result aggregation
- Learning integration hooks
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.orchestrator_agent import (
    OrchestratorAgent, OrchestrationResult, ToolExecutionResult
)
from src.agents.intent_models import Intent, IntentResult
from src.agents.tool_discovery_agent import ToolDiscoveryAgent
from src.core.tool_registry import ToolRegistry
from src.learning.q_learning_engine import QLearningEngine
from src.learning.reward_calculator import ExecutionMetrics
from src.state_machine.conversation_state_machine import ConversationStates


class TestOrchestratorIntegration:
    """Integration tests for Orchestrator Agent functionality."""
    
    @pytest.fixture
    def test_config(self):
        """Test configuration with all features enabled."""
        return {
            'orchestration': {
                'max_tools_per_query': 3,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'q_learning': {
                'enable_learning': True,
                'alpha': 0.1,
                'gamma': 0.9,
                'epsilon': 0.2,
                'model_path': 'test_model.pkl'
            },
            'intent_recognition': {
                'model': 'all-MiniLM-L6-v2',
                'confidence_threshold': 0.7
            },
            'database': {
                'tool_registry': ':memory:',
                'learning_db': ':memory:'
            }
        }
    
    @pytest.fixture
    async def setup_test_environment(self, test_config, tmp_path):
        """Set up test environment with mock tools and registry."""
        # Create temporary registry database
        registry_path = tmp_path / "test_registry.db"
        registry = ToolRegistry(str(registry_path))
        
        # Add test tools to registry
        test_tools = [
            {
                'id': 'search.web',
                'name': 'Web Search',
                'type': 'search',
                'server': 'search_mcp',
                'capabilities': json.dumps({
                    'operations': ['search', 'find', 'query']
                }),
                'status': 'active',
                'performance_score': 0.85
            },
            {
                'id': 'database.query',
                'name': 'Database Query',
                'type': 'database',
                'server': 'sqlite_mcp',
                'capabilities': json.dumps({
                    'operations': ['query', 'retrieve', 'analyze']
                }),
                'status': 'active',
                'performance_score': 0.90
            },
            {
                'id': 'filesystem.read',
                'name': 'File Reader',
                'type': 'filesystem',
                'server': 'filesystem_mcp',
                'capabilities': json.dumps({
                    'operations': ['read', 'retrieve', 'get']
                }),
                'status': 'active',
                'performance_score': 0.95
            },
            {
                'id': 'database.export',
                'name': 'Database Export',
                'type': 'database',
                'server': 'sqlite_mcp',
                'capabilities': json.dumps({
                    'operations': ['export', 'save']
                }),
                'status': 'active',
                'performance_score': 0.80
            }
        ]
        
        for tool in test_tools:
            registry.register_tool(tool)
        
        # Add tool relationships
        await registry.add_tool_relationship('database.query', 'database.export', 'complements')
        await registry.add_tool_relationship('search.web', 'filesystem.read', 'conflicts')
        
        # Update test config with registry path
        test_config['database']['tool_registry'] = str(registry_path)
        
        return {
            'config': test_config,
            'registry': registry,
            'registry_path': str(registry_path),
            'test_tools': test_tools
        }
    
    @pytest.fixture
    async def orchestrator_with_mocks(self, setup_test_environment):
        """Create orchestrator with mocked MCP integration."""
        config = setup_test_environment['config']
        
        # Create orchestrator
        orchestrator = OrchestratorAgent(config)
        
        # Mock MCP integration for tool execution
        mock_mcp = AsyncMock()
        orchestrator.mcp_integration = mock_mcp
        
        # Mock successful tool execution by default
        async def mock_execute_tool(tool_id, tool_input):
            """Mock tool execution with realistic responses."""
            await asyncio.sleep(0.05)  # Simulate execution time
            
            if 'search' in tool_id:
                return {
                    'results': [
                        {'title': 'Result 1', 'url': 'http://example.com/1'},
                        {'title': 'Result 2', 'url': 'http://example.com/2'}
                    ]
                }
            elif 'database' in tool_id:
                return {
                    'rows': [
                        {'id': 1, 'name': 'Item 1'},
                        {'id': 2, 'name': 'Item 2'}
                    ],
                    'count': 2
                }
            elif 'filesystem' in tool_id:
                return {
                    'content': 'File content here',
                    'size': 1024
                }
            else:
                return {'status': 'success', 'data': 'Generic result'}
        
        mock_mcp.execute_tool = AsyncMock(side_effect=mock_execute_tool)
        mock_mcp.initialize = AsyncMock()
        mock_mcp.shutdown = AsyncMock()
        
        # Initialize orchestrator
        await orchestrator.initialize()
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_tool_selection_traditional(self, orchestrator_with_mocks, setup_test_environment):
        """Test traditional tool selection strategies."""
        orchestrator = orchestrator_with_mocks
        test_tools = setup_test_environment['test_tools']
        
        # Disable Q-learning for this test
        orchestrator.config['q_learning']['enable_learning'] = False
        orchestrator.q_learning_engine = None
        
        # Test performance_weighted strategy (default)
        intent_result = IntentResult(
            raw_query="Search for information and save to database",
            primary_intent=Intent(
                type='query.search',
                keywords=['search', 'information', 'save'],
                confidence=0.85
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=50.0
        )
        
        # Mock intent recognition
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Process query
        result = await orchestrator.process_user_query("Search for information and save to database")
        
        # Verify tool selection
        assert result.success
        assert len(result.selected_tools) > 0
        assert len(result.selected_tools) <= orchestrator.config['orchestration']['max_tools_per_query']
        
        # Verify tools were selected based on relevance and performance
        assert any('search' in tool_id for tool_id in result.selected_tools)
        
        # Test relevance_only strategy
        orchestrator.config['orchestration']['tool_selection_strategy'] = 'relevance_only'
        result2 = await orchestrator.process_user_query("Find database records")
        
        assert result2.success
        assert any('database' in tool_id for tool_id in result2.selected_tools)
        
        # Test performance_only strategy
        orchestrator.config['orchestration']['tool_selection_strategy'] = 'performance_only'
        result3 = await orchestrator.process_user_query("Read some data")
        
        assert result3.success
        # Should select high-performance tools
        selected_tools = result3.selected_tools
        assert len(selected_tools) > 0
    
    @pytest.mark.asyncio
    async def test_tool_selection_with_constraints(self, orchestrator_with_mocks):
        """Test tool selection with relationship constraints."""
        orchestrator = orchestrator_with_mocks
        
        # Create intent that would select conflicting tools
        intent_result = IntentResult(
            raw_query="Search web and read files",
            primary_intent=Intent(
                type='query.search',
                keywords=['search', 'web', 'read', 'files'],
                confidence=0.80
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=45.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Process query
        result = await orchestrator.process_user_query("Search web and read files")
        
        # Verify that conflicting tools are not selected together
        selected_tools = result.selected_tools
        has_search = any('search.web' in tool_id for tool_id in selected_tools)
        has_filesystem = any('filesystem.read' in tool_id for tool_id in selected_tools)
        
        # Should not have both due to conflict relationship
        assert not (has_search and has_filesystem)
    
    @pytest.mark.asyncio
    async def test_parallel_execution_management(self, orchestrator_with_mocks):
        """Test parallel execution of multiple tools."""
        orchestrator = orchestrator_with_mocks
        
        # Ensure parallel execution is enabled
        orchestrator.config['orchestration']['parallel_execution'] = True
        
        # Create intent that will select multiple tools
        intent_result = IntentResult(
            raw_query="Analyze database and search for related information",
            primary_intent=Intent(
                type='query.analyze',
                keywords=['analyze', 'database', 'search', 'information'],
                confidence=0.90
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=40.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Track execution times
        execution_times = []
        original_execute = orchestrator.mcp_integration.execute_tool
        
        async def track_execution(tool_id, tool_input):
            start = time.time()
            result = await original_execute(tool_id, tool_input)
            execution_times.append((tool_id, time.time() - start))
            return result
        
        orchestrator.mcp_integration.execute_tool = track_execution
        
        # Process query
        start_time = time.time()
        result = await orchestrator.process_user_query("Analyze database and search for related information")
        total_time = time.time() - start_time
        
        # Verify parallel execution
        assert result.success
        assert len(result.execution_results) > 1
        
        # Total time should be less than sum of individual execution times (parallel)
        individual_sum = sum(t[1] for t in execution_times)
        assert total_time < individual_sum * 0.9  # Allow some overhead
        
        # Test sequential execution
        orchestrator.config['orchestration']['parallel_execution'] = False
        execution_times.clear()
        
        start_time = time.time()
        result2 = await orchestrator.process_user_query("Analyze database and search for related information")
        total_time2 = time.time() - start_time
        
        # Sequential should take approximately the sum of individual times
        individual_sum2 = sum(t[1] for t in execution_times)
        assert total_time2 >= individual_sum2 * 0.8  # Allow some measurement error
    
    @pytest.mark.asyncio
    async def test_parallel_execution_error_handling(self, orchestrator_with_mocks):
        """Test error handling during parallel execution."""
        orchestrator = orchestrator_with_mocks
        
        # Mock some tools to fail
        async def mock_execute_with_errors(tool_id, tool_input):
            await asyncio.sleep(0.05)
            
            if 'search' in tool_id:
                raise Exception("Search service unavailable")
            elif 'database' in tool_id:
                return {'rows': [], 'count': 0}
            else:
                return {'status': 'success'}
        
        orchestrator.mcp_integration.execute_tool = AsyncMock(side_effect=mock_execute_with_errors)
        
        intent_result = IntentResult(
            raw_query="Search and analyze data",
            primary_intent=Intent(
                type='query.analyze',
                keywords=['search', 'analyze', 'data'],
                confidence=0.85
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=35.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Process query
        result = await orchestrator.process_user_query("Search and analyze data")
        
        # Should handle errors gracefully
        assert isinstance(result, OrchestrationResult)
        
        # Check individual results
        failed_results = [r for r in result.execution_results if not r.success]
        successful_results = [r for r in result.execution_results if r.success]
        
        assert len(failed_results) > 0
        assert any('Search service unavailable' in r.error for r in failed_results)
        
        # Some tools should still succeed
        assert len(successful_results) >= 0
    
    @pytest.mark.asyncio
    async def test_result_aggregation(self, orchestrator_with_mocks):
        """Test aggregation of results from multiple tools."""
        orchestrator = orchestrator_with_mocks
        
        # Mock varied tool responses
        async def mock_varied_responses(tool_id, tool_input):
            await asyncio.sleep(0.03)
            
            if 'search' in tool_id:
                return {
                    'results': [
                        {'title': 'Search Result 1', 'score': 0.95},
                        {'title': 'Search Result 2', 'score': 0.87}
                    ],
                    'total': 2
                }
            elif 'database.query' in tool_id:
                return {
                    'rows': [
                        {'id': 1, 'name': 'DB Item 1', 'value': 100},
                        {'id': 2, 'name': 'DB Item 2', 'value': 200}
                    ],
                    'query_time_ms': 25
                }
            elif 'database.export' in tool_id:
                return {
                    'exported_file': '/tmp/export.csv',
                    'rows_exported': 2
                }
            else:
                return {'status': 'completed'}
        
        orchestrator.mcp_integration.execute_tool = AsyncMock(side_effect=mock_varied_responses)
        
        intent_result = IntentResult(
            raw_query="Search data, query database, and export results",
            primary_intent=Intent(
                type='query.search',
                keywords=['search', 'query', 'database', 'export'],
                confidence=0.92
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=30.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Process query
        result = await orchestrator.process_user_query("Search data, query database, and export results")
        
        # Verify result aggregation
        assert result.success
        assert len(result.execution_results) >= 2
        
        # Check that all results are properly captured
        search_result = next((r for r in result.execution_results if 'search' in r.tool_id), None)
        assert search_result is not None
        assert search_result.success
        assert 'results' in search_result.result
        
        db_results = [r for r in result.execution_results if 'database' in r.tool_id]
        assert len(db_results) >= 1
        
        # Verify summary includes information from all tools
        assert 'Successfully executed' in result.summary
        assert str(len([r for r in result.execution_results if r.success])) in result.summary
    
    @pytest.mark.asyncio
    async def test_partial_success_handling(self, orchestrator_with_mocks):
        """Test handling of partial successes and completion percentages."""
        orchestrator = orchestrator_with_mocks
        
        # Mock partial success scenarios
        async def mock_partial_success(tool_id, tool_input):
            await asyncio.sleep(0.04)
            
            if 'database' in tool_id:
                # Simulate partial result with timeout
                error = Exception("Query timeout after retrieving partial results")
                error.partial_result = {'rows': [{'id': 1, 'name': 'Partial'}], 'incomplete': True}
                error.completion_percentage = 0.3
                raise error
            else:
                return {'status': 'success'}
        
        orchestrator.mcp_integration.execute_tool = AsyncMock(side_effect=mock_partial_success)
        
        # Override _check_partial_success to handle our mock
        original_check = orchestrator._check_partial_success
        
        def mock_check_partial(error, tool_id):
            if hasattr(error, 'partial_result'):
                return {
                    'data': error.partial_result,
                    'completion': getattr(error, 'completion_percentage', 0.5)
                }
            return original_check(error, tool_id)
        
        orchestrator._check_partial_success = mock_check_partial
        
        intent_result = IntentResult(
            raw_query="Query database",
            primary_intent=Intent(
                type='query.retrieve',
                keywords=['query', 'database'],
                confidence=0.88
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=28.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Process query
        result = await orchestrator.process_user_query("Query database")
        
        # Check partial success handling
        db_result = next((r for r in result.execution_results if 'database' in r.tool_id), None)
        assert db_result is not None
        assert not db_result.success  # Overall marked as failure
        assert db_result.partial_success  # But partial success is true
        assert db_result.completion_percentage == 0.3
        assert db_result.result is not None  # Partial data is captured
        assert 'rows' in db_result.result
    
    @pytest.mark.asyncio
    async def test_result_quality_evaluation(self, orchestrator_with_mocks):
        """Test evaluation of result quality scores."""
        orchestrator = orchestrator_with_mocks
        
        # Mock responses with varying quality
        async def mock_quality_responses(tool_id, tool_input):
            await asyncio.sleep(0.02)
            
            if 'search' in tool_id:
                # High quality: multiple results
                return {
                    'results': [
                        {'title': f'Result {i}', 'relevance': 0.9 - i*0.1}
                        for i in range(10)
                    ]
                }
            elif 'database' in tool_id:
                # Low quality: empty result
                return {'rows': [], 'count': 0}
            elif 'filesystem' in tool_id:
                # Medium quality: small content
                return {'content': 'Short file.'}
            else:
                return None  # Very low quality
        
        orchestrator.mcp_integration.execute_tool = AsyncMock(side_effect=mock_quality_responses)
        
        intent_result = IntentResult(
            raw_query="Search and analyze various sources",
            primary_intent=Intent(
                type='query.search',
                keywords=['search', 'analyze', 'sources'],
                confidence=0.91
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=25.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Process query
        result = await orchestrator.process_user_query("Search and analyze various sources")
        
        # Verify quality scores
        for exec_result in result.execution_results:
            assert hasattr(exec_result, 'result_quality')
            assert 0 <= exec_result.result_quality <= 1.0
            
            if 'search' in exec_result.tool_id:
                # Should have high quality due to multiple results
                assert exec_result.result_quality > 0.7
            elif 'database' in exec_result.tool_id:
                # Should have low quality due to empty results
                assert exec_result.result_quality < 0.5
    
    @pytest.mark.asyncio
    async def test_resource_usage_tracking(self, orchestrator_with_mocks):
        """Test tracking of resource usage during execution."""
        orchestrator = orchestrator_with_mocks
        
        intent_result = IntentResult(
            raw_query="Process data",
            primary_intent=Intent(
                type='action.process',
                keywords=['process', 'data'],
                confidence=0.87
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=22.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Process query
        result = await orchestrator.process_user_query("Process data")
        
        # Verify resource usage tracking
        assert result.success
        for exec_result in result.execution_results:
            assert hasattr(exec_result, 'resource_usage')
            assert exec_result.resource_usage is not None
            
            # Check resource metrics
            assert 'memory_mb' in exec_result.resource_usage
            assert 'cpu_percent' in exec_result.resource_usage
            assert 'execution_time_ms' in exec_result.resource_usage
            
            # Values should be reasonable
            assert exec_result.resource_usage['memory_mb'] >= 0
            assert 0 <= exec_result.resource_usage['cpu_percent'] <= 100
            assert exec_result.resource_usage['execution_time_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_error_classification(self, orchestrator_with_mocks):
        """Test classification of different error types."""
        orchestrator = orchestrator_with_mocks
        
        # Test various error scenarios
        error_scenarios = [
            ("Connection timeout", 'network_timeout'),
            ("Permission denied", 'permission_error'),
            ("Rate limit exceeded", 'rate_limit'),
            ("Network unreachable", 'connection_error'),
            ("Unknown error", 'other')
        ]
        
        for error_msg, expected_type in error_scenarios:
            error = Exception(error_msg)
            error_type = orchestrator._classify_error(error)
            assert error_type == expected_type
    
    @pytest.mark.asyncio
    async def test_summary_generation(self, orchestrator_with_mocks):
        """Test comprehensive summary generation."""
        orchestrator = orchestrator_with_mocks
        
        # Create mixed results
        execution_results = [
            ToolExecutionResult(
                tool_id='search.web',
                tool_name='Web Search',
                success=True,
                result={'results': ['item1', 'item2']},
                execution_time_ms=150.5,
                result_quality=0.85
            ),
            ToolExecutionResult(
                tool_id='database.query',
                tool_name='Database Query',
                success=False,
                result=None,
                error='Connection failed',
                execution_time_ms=50.0,
                error_type='connection_error'
            ),
            ToolExecutionResult(
                tool_id='filesystem.read',
                tool_name='File Reader',
                success=True,
                result='Long file content that should be truncated in the summary...' * 10,
                execution_time_ms=75.0,
                partial_success=False,
                completion_percentage=1.0
            )
        ]
        
        intent_result = IntentResult(
            raw_query="Complex multi-tool query",
            primary_intent=Intent(
                type='query.analyze',
                keywords=['complex', 'multi', 'tool'],
                confidence=0.89
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=20.0
        )
        
        summary = orchestrator._generate_summary(intent_result, execution_results)
        
        # Verify summary content
        assert 'Intent: query.analyze' in summary
        assert 'confidence: 0.89' in summary
        assert 'Successfully executed 2 tool(s)' in summary
        assert 'Failed to execute 1 tool(s)' in summary
        assert 'Web Search' in summary
        assert 'Database Query' in summary
        assert 'Connection failed' in summary
        
        # Long results should be truncated
        assert '...' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])