"""
Unit tests for Orchestrator Agent.

Tests the orchestration functionality including:
- Intent recognition integration
- Tool discovery and selection
- Tool execution coordination
- State machine transitions
- Error handling
- Parallel and sequential execution
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import time

from src.agents.orchestrator_agent import (
    OrchestratorAgent, OrchestrationResult, ToolExecutionResult
)
from src.agents.intent_recognition_agent import IntentResult, Intent
from src.state_machine.conversation_state_machine import ConversationStates


class TestOrchestratorAgent:
    """Test cases for OrchestratorAgent class."""
    
    @pytest.fixture
    def default_config(self):
        """Default configuration for testing."""
        return {
            'orchestration': {
                'max_tools_per_query': 3,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'intent_recognition': {
                'model': 'test-model',
                'confidence_threshold': 0.7
            },
            'database': {
                'tool_registry': 'test_registry.db'
            }
        }
    
    @pytest.fixture
    def mock_intent_result(self):
        """Create a mock intent result."""
        primary_intent = Intent(
            type='query.search',
            keywords=['find', 'python', 'files'],
            confidence=0.85,
            entities={'file_type': 'python'}
        )
        return IntentResult(
            raw_query="Find all Python files",
            primary_intent=primary_intent,
            alternative_intents=[],
            context={'domain': 'development'},
            processing_time_ms=50.0
        )
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tool data."""
        return [
            {
                'id': 'filesystem.search',
                'name': 'File Search',
                'type': 'filesystem',
                'performance_score': 0.9,
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'search_files', 'category': 'search'}
                    ]
                })
            },
            {
                'id': 'sqlite.query',
                'name': 'SQLite Query',
                'type': 'sqlite',
                'performance_score': 0.8,
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'query', 'category': 'query'}
                    ]
                })
            }
        ]
    
    @pytest.fixture
    async def orchestrator(self, default_config):
        """Create OrchestratorAgent instance with mocks."""
        with patch('src.agents.orchestrator_agent.IntentRecognitionAgent') as mock_intent_agent:
            with patch('src.agents.orchestrator_agent.MCPIntegration') as mock_mcp:
                with patch('src.agents.orchestrator_agent.ToolRegistry') as mock_registry:
                    with patch('src.agents.orchestrator_agent.ConversationStateMachine') as mock_state_machine:
                        # Create orchestrator
                        orchestrator = OrchestratorAgent(config=default_config)
                        
                        # Setup mocks
                        orchestrator.intent_agent = mock_intent_agent.return_value
                        orchestrator.mcp_integration = mock_mcp.return_value
                        orchestrator.tool_registry = mock_registry.return_value
                        orchestrator.state_machine = mock_state_machine.return_value
                        
                        # Configure async methods
                        orchestrator.intent_agent.process_query = AsyncMock()
                        orchestrator.mcp_integration.initialize = AsyncMock()
                        orchestrator.mcp_integration.shutdown = AsyncMock()
                        orchestrator.mcp_integration.execute_tool = AsyncMock()
                        orchestrator.tool_registry.search_tools = AsyncMock()
                        
                        # Configure state machine
                        orchestrator.state_machine.receive_query = AsyncMock(return_value=True)
                        orchestrator.state_machine.recognize_intent = AsyncMock()
                        orchestrator.state_machine.discover_tools = AsyncMock()
                        orchestrator.state_machine.start_execution = AsyncMock()
                        orchestrator.state_machine.complete_execution = AsyncMock()
                        orchestrator.state_machine.return_to_idle = AsyncMock()
                        orchestrator.state_machine.handle_error = AsyncMock()
                        orchestrator.state_machine.is_in_state = Mock(return_value=True)
                        orchestrator.state_machine.start = AsyncMock()
                        
                        yield orchestrator
    
    def test_initialization(self, default_config):
        """Test OrchestratorAgent initialization."""
        with patch('src.agents.orchestrator_agent.IntentRecognitionAgent'):
            with patch('src.agents.orchestrator_agent.MCPIntegration'):
                with patch('src.agents.orchestrator_agent.ToolRegistry'):
                    orchestrator = OrchestratorAgent(config=default_config)
                    
                    assert orchestrator.config == default_config
                    assert orchestrator.intent_capability_map is not None
                    assert 'query.search' in orchestrator.intent_capability_map
                    assert orchestrator._state_machine_initialized is False
    
    @pytest.mark.asyncio
    async def test_process_user_query_success(self, orchestrator, mock_intent_result, mock_tools):
        """Test successful query processing."""
        # Setup mocks
        orchestrator.intent_agent.process_query.return_value = mock_intent_result
        orchestrator.tool_registry.search_tools.return_value = mock_tools
        orchestrator.mcp_integration.execute_tool.return_value = {
            'success': True,
            'result': 'Found 10 Python files'
        }
        orchestrator.state_machine.is_in_state.side_effect = [
            True,  # TOOLS_DISCOVERED check
        ]
        
        # Process query
        result = await orchestrator.process_user_query("Find all Python files")
        
        # Verify result
        assert isinstance(result, OrchestrationResult)
        assert result.success is True
        assert result.query == "Find all Python files"
        assert result.intent == mock_intent_result
        assert len(result.discovered_tools) > 0
        assert len(result.execution_results) > 0
        assert result.total_time_ms > 0
        
        # Verify state machine transitions
        orchestrator.state_machine.receive_query.assert_called_once()
        orchestrator.state_machine.recognize_intent.assert_called_once()
        orchestrator.state_machine.discover_tools.assert_called_once()
        orchestrator.state_machine.start_execution.assert_called_once()
        orchestrator.state_machine.complete_execution.assert_called_once()
        orchestrator.state_machine.return_to_idle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_user_query_no_tools_found(self, orchestrator, mock_intent_result):
        """Test query processing when no tools are found."""
        # Setup mocks
        orchestrator.intent_agent.process_query.return_value = mock_intent_result
        orchestrator.tool_registry.search_tools.return_value = []
        orchestrator.state_machine.is_in_state.side_effect = [
            False,  # Not TOOLS_DISCOVERED
            True,   # NO_TOOLS_FOUND
        ]
        
        # Process query
        result = await orchestrator.process_user_query("Find something impossible")
        
        # Verify result
        assert result.success is False
        assert len(result.discovered_tools) == 0
        assert len(result.execution_results) == 0
        assert "No tools found" in result.summary
    
    @pytest.mark.asyncio
    async def test_process_user_query_clarification_needed(self, orchestrator, mock_intent_result):
        """Test query processing when clarification is needed."""
        # Setup mocks
        orchestrator.intent_agent.process_query.return_value = mock_intent_result
        orchestrator.tool_registry.search_tools.return_value = []
        orchestrator.state_machine.is_in_state.side_effect = [
            False,  # Not TOOLS_DISCOVERED
            False,  # Not NO_TOOLS_FOUND
            True,   # CLARIFICATION_NEEDED
        ]
        
        # Process query
        result = await orchestrator.process_user_query("Do something")
        
        # Verify result
        assert result.success is False
        assert "Query is ambiguous" in result.summary
    
    @pytest.mark.asyncio
    async def test_process_user_query_error_handling(self, orchestrator):
        """Test error handling during query processing."""
        # Setup mock to raise error
        orchestrator.intent_agent.process_query.side_effect = Exception("Intent recognition failed")
        
        # Process query
        result = await orchestrator.process_user_query("Test query")
        
        # Verify result
        assert result.success is False
        assert "Error processing query" in result.summary
        orchestrator.state_machine.handle_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_tools_for_intent(self, orchestrator, mock_intent_result, mock_tools):
        """Test tool discovery based on intent."""
        # Setup mocks
        orchestrator.tool_registry.search_tools.return_value = mock_tools
        
        # Discover tools
        discovered = await orchestrator.discover_tools_for_intent(mock_intent_result)
        
        # Verify results
        assert len(discovered) == len(mock_tools)
        assert all('relevance_score' in tool for tool in discovered)
        # Should be sorted by relevance score
        scores = [tool['relevance_score'] for tool in discovered]
        assert scores == sorted(scores, reverse=True)
    
    def test_calculate_relevance_score(self, orchestrator, mock_intent_result, mock_tools):
        """Test relevance score calculation."""
        tool = mock_tools[0]
        required_capabilities = ['search', 'find']
        query_keywords = ['find', 'python', 'files']
        
        score = orchestrator._calculate_relevance_score(
            tool, mock_intent_result, required_capabilities, query_keywords
        )
        
        assert 0 <= score <= 1.0
        # Should have a good score since capabilities match
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_select_tools_performance_weighted(self, orchestrator, mock_intent_result, mock_tools):
        """Test tool selection with performance weighted strategy."""
        # Add relevance scores
        for i, tool in enumerate(mock_tools):
            tool['relevance_score'] = 0.8 - i * 0.1
        
        selected = await orchestrator.select_tools(mock_tools, mock_intent_result)
        
        assert len(selected) <= orchestrator.config['orchestration']['max_tools_per_query']
        # Should filter out low relevance tools
        assert all(tool['relevance_score'] > 0.3 for tool in selected)
    
    @pytest.mark.asyncio
    async def test_select_tools_different_strategies(self, orchestrator, mock_intent_result, mock_tools):
        """Test different tool selection strategies."""
        # Add scores
        mock_tools[0]['relevance_score'] = 0.9
        mock_tools[0]['performance_score'] = 0.5
        mock_tools[1]['relevance_score'] = 0.6
        mock_tools[1]['performance_score'] = 0.9
        
        # Test relevance_only strategy
        orchestrator.config['orchestration']['tool_selection_strategy'] = 'relevance_only'
        selected = await orchestrator.select_tools(mock_tools, mock_intent_result)
        assert selected[0]['relevance_score'] > selected[1]['relevance_score'] if len(selected) > 1 else True
        
        # Test performance_only strategy
        orchestrator.config['orchestration']['tool_selection_strategy'] = 'performance_only'
        selected = await orchestrator.select_tools(mock_tools, mock_intent_result)
        assert selected[0]['performance_score'] >= selected[1]['performance_score'] if len(selected) > 1 else True
    
    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self, orchestrator, mock_tools):
        """Test parallel tool execution."""
        orchestrator.config['orchestration']['parallel_execution'] = True
        orchestrator.mcp_integration.execute_tool.return_value = {
            'success': True,
            'result': 'Test result'
        }
        
        results = await orchestrator.execute_tools(
            mock_tools, "Test query", {}
        )
        
        assert len(results) == len(mock_tools)
        assert all(isinstance(r, ToolExecutionResult) for r in results)
        assert all(r.success for r in results)
        # Should have called execute_tool for each tool
        assert orchestrator.mcp_integration.execute_tool.call_count == len(mock_tools)
    
    @pytest.mark.asyncio
    async def test_execute_tools_sequential(self, orchestrator, mock_tools):
        """Test sequential tool execution."""
        orchestrator.config['orchestration']['parallel_execution'] = False
        orchestrator.mcp_integration.execute_tool.return_value = {
            'success': True,
            'result': 'Test result'
        }
        
        results = await orchestrator.execute_tools(
            mock_tools, "Test query", {}
        )
        
        assert len(results) == len(mock_tools)
        assert all(isinstance(r, ToolExecutionResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_execute_tools_with_failures(self, orchestrator, mock_tools):
        """Test tool execution with some failures."""
        # Make first tool succeed, second fail
        orchestrator.mcp_integration.execute_tool.side_effect = [
            {'success': True, 'result': 'Success'},
            Exception("Tool execution failed")
        ]
        
        results = await orchestrator.execute_tools(
            mock_tools, "Test query", {}
        )
        
        assert len(results) == len(mock_tools)
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "Tool execution failed"
    
    @pytest.mark.asyncio
    async def test_execute_single_tool(self, orchestrator, mock_tools):
        """Test single tool execution."""
        tool = mock_tools[0]
        orchestrator.mcp_integration.execute_tool.return_value = {
            'success': True,
            'result': 'Tool output'
        }
        
        result = await orchestrator._execute_single_tool(
            tool, "Test query", {}
        )
        
        assert isinstance(result, ToolExecutionResult)
        assert result.tool_id == tool['id']
        assert result.tool_name == tool['name']
        assert result.success is True
        assert result.execution_time_ms > 0
    
    def test_prepare_tool_input(self, orchestrator, mock_tools):
        """Test tool input preparation."""
        # Test search tool
        search_tool = {'type': 'search', 'id': 'search.web'}
        input_data = orchestrator._prepare_tool_input(
            search_tool, "Search for Python", {}
        )
        assert 'search_query' in input_data
        assert input_data['search_query'] == "Search for Python"
        
        # Test sqlite tool
        sqlite_tool = {'type': 'sqlite', 'id': 'sqlite.query'}
        input_data = orchestrator._prepare_tool_input(
            sqlite_tool, "Find records", {}
        )
        assert 'sql_query' in input_data
        
        # Test filesystem tool
        fs_tool = {'type': 'filesystem', 'id': 'fs.read'}
        context = {'working_directory': '/home/user'}
        input_data = orchestrator._prepare_tool_input(
            fs_tool, "Read file", context
        )
        assert input_data['path'] == '/home/user'
    
    def test_convert_to_sql(self, orchestrator):
        """Test SQL conversion from natural language."""
        # Test find/search query
        sql = orchestrator._convert_to_sql("Find Python files")
        assert "SELECT" in sql
        assert "LIKE" in sql
        
        # Test create query
        sql = orchestrator._convert_to_sql("Create new record")
        assert "INSERT" in sql
        
        # Test default query
        sql = orchestrator._convert_to_sql("Something else")
        assert "SELECT" in sql
        assert "LIMIT" in sql
    
    def test_generate_summary(self, orchestrator, mock_intent_result):
        """Test summary generation."""
        # Success results
        success_results = [
            ToolExecutionResult(
                tool_id='tool1',
                tool_name='Tool 1',
                success=True,
                result={'data': 'result1'}
            ),
            ToolExecutionResult(
                tool_id='tool2',
                tool_name='Tool 2',
                success=False,
                result=None,
                error='Connection timeout'
            )
        ]
        
        summary = orchestrator._generate_summary(mock_intent_result, success_results)
        
        assert "Intent: query.search" in summary
        assert "Successfully executed 1 tool(s)" in summary
        assert "Failed to execute 1 tool(s)" in summary
        assert "Tool 1" in summary
        assert "Tool 2" in summary
        assert "Connection timeout" in summary
    
    def test_summarize_result(self, orchestrator):
        """Test result summarization."""
        # Test dict result
        assert "3 items returned" == orchestrator._summarize_result({'a': 1, 'b': 2, 'c': 3})
        
        # Test list result
        assert "5 results found" == orchestrator._summarize_result([1, 2, 3, 4, 5])
        
        # Test string result
        assert "Short string" == orchestrator._summarize_result("Short string")
        
        # Test long string truncation
        long_string = "x" * 150
        summary = orchestrator._summarize_result(long_string)
        assert len(summary) == 103  # 100 chars + "..."
        assert summary.endswith("...")
        
        # Test None
        assert "No result" == orchestrator._summarize_result(None)
    
    @pytest.mark.asyncio
    async def test_state_machine_methods(self, orchestrator):
        """Test state machine related methods."""
        # Test get_state_machine_summary
        summary = orchestrator.get_state_machine_summary()
        assert summary == {"status": "not_initialized"}
        
        orchestrator._state_machine_initialized = True
        orchestrator.state_machine.get_conversation_summary.return_value = {
            "state": "active"
        }
        summary = orchestrator.get_state_machine_summary()
        assert summary == {"state": "active"}
        
        # Test get_current_state
        mock_state = Mock()
        mock_state.name = "QUERY_RECEIVED"
        orchestrator.state_machine.get_current_state.return_value = mock_state
        state = orchestrator.get_current_state()
        assert state == "QUERY_RECEIVED"
    
    @pytest.mark.asyncio
    async def test_handle_user_clarification(self, orchestrator):
        """Test handling user clarification."""
        # Not in clarification state
        orchestrator.state_machine.is_in_state.return_value = False
        result = await orchestrator.handle_user_clarification("More details")
        assert result is False
        
        # In clarification state
        orchestrator.state_machine.is_in_state.return_value = True
        orchestrator.state_machine.handle_clarification = AsyncMock(return_value=True)
        result = await orchestrator.handle_user_clarification("More details")
        assert result is True
        orchestrator.state_machine.handle_clarification.assert_called_with("More details")
    
    @pytest.mark.asyncio
    async def test_request_retry(self, orchestrator):
        """Test retry request."""
        orchestrator.state_machine.request_retry = AsyncMock(return_value=True)
        result = await orchestrator.request_retry()
        assert result is True
        orchestrator.state_machine.request_retry.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_operation(self, orchestrator):
        """Test operation cancellation."""
        orchestrator.state_machine.cancel_operation = AsyncMock(return_value=True)
        result = await orchestrator.cancel_current_operation()
        assert result is True
        orchestrator.state_machine.cancel_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator):
        """Test orchestrator initialization."""
        with patch('src.agents.orchestrator_agent.register_handlers') as mock_register:
            await orchestrator.initialize()
            
            assert orchestrator._state_machine_initialized is True
            mock_register.assert_called_once_with(orchestrator.state_machine)
            orchestrator.state_machine.start.assert_called_once()
            orchestrator.mcp_integration.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator):
        """Test orchestrator shutdown."""
        await orchestrator.shutdown()
        
        orchestrator.mcp_integration.shutdown.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])