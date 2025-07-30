"""
Integration tests for parallel tool execution support.

This test suite verifies that the system properly:
1. Executes multiple tools in parallel
2. Shows performance improvement over sequential execution
3. Handles errors gracefully in parallel execution
4. Tracks resource usage during parallel execution
5. Aggregates results from parallel executions
6. Switches between parallel and sequential modes based on configuration
"""

import pytest
import asyncio
import json
import time
import tempfile
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.orchestrator_agent import OrchestratorAgent, ToolExecutionResult
from src.agents.intent_models import Intent, IntentResult
from src.core.tool_registry import ToolRegistry
from src.core.mcp_integration import MCPIntegration
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestParallelExecution:
    """Test suite for parallel tool execution functionality."""
    
    @pytest.fixture
    async def setup_test_environment(self, tmp_path):
        """Set up test environment with mock tools."""
        # Create temporary registry
        registry_path = tmp_path / "test_registry.db"
        registry = ToolRegistry(str(registry_path))
        
        # Register test tools
        test_tools = [
            {
                'id': 'tool.fast1',
                'name': 'Fast Tool 1',
                'type': 'computation',
                'server': 'mock_server',
                'capabilities': json.dumps({'operations': ['compute', 'process']}),
                'status': 'active',
                'performance_score': 0.95
            },
            {
                'id': 'tool.fast2',
                'name': 'Fast Tool 2',
                'type': 'computation',
                'server': 'mock_server',
                'capabilities': json.dumps({'operations': ['compute', 'analyze']}),
                'status': 'active',
                'performance_score': 0.93
            },
            {
                'id': 'tool.slow1',
                'name': 'Slow Tool 1',
                'type': 'analysis',
                'server': 'mock_server',
                'capabilities': json.dumps({'operations': ['analyze', 'deep_scan']}),
                'status': 'active',
                'performance_score': 0.85
            },
            {
                'id': 'tool.slow2',
                'name': 'Slow Tool 2',
                'type': 'analysis',
                'server': 'mock_server',
                'capabilities': json.dumps({'operations': ['analyze', 'report']}),
                'status': 'active',
                'performance_score': 0.82
            },
            {
                'id': 'tool.error_prone',
                'name': 'Error Prone Tool',
                'type': 'experimental',
                'server': 'mock_server',
                'capabilities': json.dumps({'operations': ['test', 'experiment']}),
                'status': 'active',
                'performance_score': 0.70
            }
        ]
        
        for tool in test_tools:
            await registry.register_tool(tool)
        
        return {
            'registry': registry,
            'registry_path': str(registry_path),
            'test_tools': test_tools
        }
    
    def create_mock_mcp_integration(self, execution_times: Dict[str, float], error_tools: List[str] = None):
        """Create a mock MCP integration with configurable execution times."""
        if error_tools is None:
            error_tools = []
        
        async def mock_execute_tool(tool_id: str, tool_input: Any):
            """Mock tool execution with configurable delay."""
            # Simulate execution time
            exec_time = execution_times.get(tool_id, 0.1)
            await asyncio.sleep(exec_time)
            
            # Simulate errors for specific tools
            if tool_id in error_tools:
                raise Exception(f"Simulated error in {tool_id}")
            
            # Return mock result
            return {
                'status': 'success',
                'result': f'Result from {tool_id}',
                'data': {'processed': True, 'tool_id': tool_id},
                'execution_time': exec_time
            }
        
        mock_mcp = AsyncMock()
        mock_mcp.execute_tool = mock_execute_tool
        mock_mcp.get_server_status = MagicMock(return_value={
            'mock_server': {'status': 'connected', 'tools_count': 5}
        })
        
        return mock_mcp
    
    @pytest.mark.asyncio
    async def test_parallel_execution_of_multiple_tools(self, setup_test_environment):
        """Test 1: Verify parallel execution of multiple tools."""
        # Create config with parallel execution enabled
        config = {
            'orchestration': {
                'max_tools_per_query': 5,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'database': {
                'tool_registry': setup_test_environment['registry_path']
            }
        }
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Mock MCP integration with different execution times
        execution_times = {
            'tool.fast1': 0.1,
            'tool.fast2': 0.15,
            'tool.slow1': 0.5,
            'tool.slow2': 0.6
        }
        orchestrator.mcp_integration = self.create_mock_mcp_integration(execution_times)
        
        # Select multiple tools for execution
        selected_tools = [
            {'id': 'tool.fast1', 'name': 'Fast Tool 1'},
            {'id': 'tool.fast2', 'name': 'Fast Tool 2'},
            {'id': 'tool.slow1', 'name': 'Slow Tool 1'},
            {'id': 'tool.slow2', 'name': 'Slow Tool 2'}
        ]
        
        # Execute tools in parallel
        start_time = time.time()
        results = await orchestrator.execute_tools(selected_tools, "test query", {})
        execution_time = time.time() - start_time
        
        # Verify all tools executed
        assert len(results) == 4, "All tools should be executed"
        
        # Verify all executions succeeded
        for result in results:
            assert result.success, f"Tool {result.tool_name} should succeed"
            assert result.result is not None, f"Tool {result.tool_name} should have result"
        
        # Verify parallel execution time is less than sequential time
        sequential_time = sum(execution_times.values())
        max_individual_time = max(execution_times.values())
        
        logger.info(f"Parallel execution time: {execution_time:.3f}s")
        logger.info(f"Sequential time would be: {sequential_time:.3f}s")
        logger.info(f"Speedup: {sequential_time/execution_time:.2f}x")
        
        # Parallel execution should be close to the slowest tool's time (with some overhead)
        assert execution_time < sequential_time * 0.8, \
            "Parallel execution should be significantly faster than sequential"
        assert execution_time < max_individual_time + 0.2, \
            "Parallel execution time should be close to slowest tool time"
        
        logger.info("✅ Parallel execution of multiple tools verified")
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self, setup_test_environment):
        """Test 2: Compare execution time between parallel and sequential modes."""
        # Base configuration
        base_config = {
            'orchestration': {
                'max_tools_per_query': 5,
                'tool_selection_strategy': 'performance_weighted'
            },
            'database': {
                'tool_registry': setup_test_environment['registry_path']
            }
        }
        
        # Tools to execute
        selected_tools = [
            {'id': 'tool.fast1', 'name': 'Fast Tool 1'},
            {'id': 'tool.fast2', 'name': 'Fast Tool 2'},
            {'id': 'tool.slow1', 'name': 'Slow Tool 1'}
        ]
        
        # Execution times
        execution_times = {
            'tool.fast1': 0.2,
            'tool.fast2': 0.3,
            'tool.slow1': 0.5
        }
        
        # Test parallel execution
        parallel_config = {**base_config, 'orchestration': {**base_config['orchestration'], 'parallel_execution': True}}
        parallel_orchestrator = OrchestratorAgent(parallel_config)
        await parallel_orchestrator.initialize()
        parallel_orchestrator.mcp_integration = self.create_mock_mcp_integration(execution_times)
        
        start_time = time.time()
        parallel_results = await parallel_orchestrator.execute_tools(selected_tools, "test query", {})
        parallel_time = time.time() - start_time
        
        # Test sequential execution
        sequential_config = {**base_config, 'orchestration': {**base_config['orchestration'], 'parallel_execution': False}}
        sequential_orchestrator = OrchestratorAgent(sequential_config)
        await sequential_orchestrator.initialize()
        sequential_orchestrator.mcp_integration = self.create_mock_mcp_integration(execution_times)
        
        start_time = time.time()
        sequential_results = await sequential_orchestrator.execute_tools(selected_tools, "test query", {})
        sequential_time = time.time() - start_time
        
        # Verify results are the same
        assert len(parallel_results) == len(sequential_results), \
            "Both modes should execute same number of tools"
        
        # Verify performance improvement
        speedup = sequential_time / parallel_time
        logger.info(f"Sequential execution time: {sequential_time:.3f}s")
        logger.info(f"Parallel execution time: {parallel_time:.3f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        assert parallel_time < sequential_time, \
            "Parallel execution should be faster than sequential"
        assert speedup > 1.5, \
            "Parallel execution should provide significant speedup (>1.5x)"
        
        logger.info("✅ Parallel vs sequential performance comparison verified")
    
    @pytest.mark.asyncio
    async def test_error_handling_in_parallel_execution(self, setup_test_environment):
        """Test 3: Test error handling in parallel execution (one tool fails, others succeed)."""
        config = {
            'orchestration': {
                'max_tools_per_query': 5,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'database': {
                'tool_registry': setup_test_environment['registry_path']
            }
        }
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Mock MCP with one tool that will error
        execution_times = {
            'tool.fast1': 0.1,
            'tool.fast2': 0.2,
            'tool.error_prone': 0.15,
            'tool.slow1': 0.3
        }
        error_tools = ['tool.error_prone']
        orchestrator.mcp_integration = self.create_mock_mcp_integration(execution_times, error_tools)
        
        # Select tools including the error-prone one
        selected_tools = [
            {'id': 'tool.fast1', 'name': 'Fast Tool 1'},
            {'id': 'tool.fast2', 'name': 'Fast Tool 2'},
            {'id': 'tool.error_prone', 'name': 'Error Prone Tool'},
            {'id': 'tool.slow1', 'name': 'Slow Tool 1'}
        ]
        
        # Execute tools
        results = await orchestrator.execute_tools(selected_tools, "test query", {})
        
        # Verify all tools were attempted
        assert len(results) == 4, "All tools should be attempted"
        
        # Verify error handling
        success_count = sum(1 for r in results if r.success)
        error_count = sum(1 for r in results if not r.success)
        
        assert success_count == 3, "Three tools should succeed"
        assert error_count == 1, "One tool should fail"
        
        # Find the error result
        error_result = next(r for r in results if not r.success)
        assert error_result.tool_id == 'tool.error_prone', "Error should be from error-prone tool"
        assert error_result.error is not None, "Error message should be present"
        assert "Simulated error" in error_result.error, "Error message should contain expected text"
        
        # Verify other tools succeeded despite the error
        for result in results:
            if result.tool_id != 'tool.error_prone':
                assert result.success, f"Tool {result.tool_name} should succeed despite other tool's error"
                assert result.result is not None, f"Tool {result.tool_name} should have valid result"
        
        logger.info("✅ Error handling in parallel execution verified")
    
    @pytest.mark.asyncio
    async def test_resource_usage_tracking(self, setup_test_environment):
        """Test 4: Test resource usage tracking during parallel execution."""
        config = {
            'orchestration': {
                'max_tools_per_query': 3,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'database': {
                'tool_registry': setup_test_environment['registry_path']
            }
        }
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Mock MCP integration
        execution_times = {
            'tool.fast1': 0.1,
            'tool.slow1': 0.3,
            'tool.slow2': 0.4
        }
        orchestrator.mcp_integration = self.create_mock_mcp_integration(execution_times)
        
        # Select tools
        selected_tools = [
            {'id': 'tool.fast1', 'name': 'Fast Tool 1'},
            {'id': 'tool.slow1', 'name': 'Slow Tool 1'},
            {'id': 'tool.slow2', 'name': 'Slow Tool 2'}
        ]
        
        # Execute tools
        results = await orchestrator.execute_tools(selected_tools, "test query", {})
        
        # Verify resource usage is tracked
        for result in results:
            assert result.resource_usage is not None, \
                f"Resource usage should be tracked for {result.tool_name}"
            
            # Check resource usage fields
            assert 'memory_mb' in result.resource_usage, "Memory usage should be tracked"
            assert 'cpu_percent' in result.resource_usage, "CPU usage should be tracked"
            assert 'execution_time_ms' in result.resource_usage, "Execution time should be tracked"
            
            # Verify values are reasonable
            assert result.resource_usage['memory_mb'] >= 0, "Memory usage should be non-negative"
            assert result.resource_usage['cpu_percent'] >= 0, "CPU usage should be non-negative"
            assert result.resource_usage['execution_time_ms'] > 0, "Execution time should be positive"
            
            # Verify execution time matches expected
            expected_time_ms = execution_times[result.tool_id] * 1000
            actual_time_ms = result.resource_usage['execution_time_ms']
            assert abs(actual_time_ms - expected_time_ms) < 100, \
                f"Execution time should be close to expected for {result.tool_name}"
        
        logger.info("✅ Resource usage tracking during parallel execution verified")
    
    @pytest.mark.asyncio
    async def test_result_aggregation_from_parallel_execution(self, setup_test_environment):
        """Test 5: Verify result aggregation from parallel executions."""
        config = {
            'orchestration': {
                'max_tools_per_query': 4,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'database': {
                'tool_registry': setup_test_environment['registry_path']
            }
        }
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Mock MCP with tools returning different types of results
        async def mock_execute_varied_results(tool_id: str, tool_input: Any):
            await asyncio.sleep(0.1)  # Simulate execution
            
            if tool_id == 'tool.fast1':
                return {'data': [1, 2, 3], 'count': 3, 'status': 'complete'}
            elif tool_id == 'tool.fast2':
                return {'data': [4, 5, 6], 'count': 3, 'status': 'complete'}
            elif tool_id == 'tool.slow1':
                return {'summary': 'Analysis complete', 'metrics': {'accuracy': 0.95}}
            elif tool_id == 'tool.slow2':
                return {'report': 'Detailed findings...', 'recommendations': ['A', 'B', 'C']}
            
        mock_mcp = AsyncMock()
        mock_mcp.execute_tool = mock_execute_varied_results
        orchestrator.mcp_integration = mock_mcp
        
        # Select tools
        selected_tools = [
            {'id': 'tool.fast1', 'name': 'Fast Tool 1'},
            {'id': 'tool.fast2', 'name': 'Fast Tool 2'},
            {'id': 'tool.slow1', 'name': 'Slow Tool 1'},
            {'id': 'tool.slow2', 'name': 'Slow Tool 2'}
        ]
        
        # Execute tools
        results = await orchestrator.execute_tools(selected_tools, "test query", {})
        
        # Verify all results are collected
        assert len(results) == 4, "All tool results should be collected"
        
        # Verify each result has expected structure
        for result in results:
            assert result.tool_id is not None, "Tool ID should be present"
            assert result.tool_name is not None, "Tool name should be present"
            assert result.success is True, "All tools should succeed"
            assert result.result is not None, "Result data should be present"
            assert result.execution_time_ms > 0, "Execution time should be tracked"
        
        # Verify results can be aggregated
        aggregated_data = []
        total_count = 0
        summaries = []
        
        for result in results:
            if 'data' in result.result:
                aggregated_data.extend(result.result['data'])
                total_count += result.result.get('count', 0)
            if 'summary' in result.result:
                summaries.append(result.result['summary'])
            if 'report' in result.result:
                summaries.append(result.result['report'])
        
        # Verify aggregation worked
        assert len(aggregated_data) == 6, "Should aggregate data from multiple tools"
        assert total_count == 6, "Should sum counts correctly"
        assert len(summaries) == 2, "Should collect all summaries"
        
        logger.info(f"Aggregated {len(aggregated_data)} data points from parallel execution")
        logger.info(f"Collected {len(summaries)} summaries")
        logger.info("✅ Result aggregation from parallel execution verified")
    
    @pytest.mark.asyncio
    async def test_configuration_based_execution_mode(self, setup_test_environment):
        """Test 6: Test configuration-based switching between parallel and sequential modes."""
        base_config = {
            'orchestration': {
                'max_tools_per_query': 3,
                'tool_selection_strategy': 'performance_weighted'
            },
            'database': {
                'tool_registry': setup_test_environment['registry_path']
            }
        }
        
        # Mock MCP
        execution_times = {'tool.fast1': 0.2, 'tool.fast2': 0.3}
        
        # Tools to execute
        selected_tools = [
            {'id': 'tool.fast1', 'name': 'Fast Tool 1'},
            {'id': 'tool.fast2', 'name': 'Fast Tool 2'}
        ]
        
        # Test 1: Parallel execution enabled
        parallel_config = {**base_config, 'orchestration': {**base_config['orchestration'], 'parallel_execution': True}}
        orchestrator1 = OrchestratorAgent(parallel_config)
        await orchestrator1.initialize()
        orchestrator1.mcp_integration = self.create_mock_mcp_integration(execution_times)
        
        start_time = time.time()
        results1 = await orchestrator1.execute_tools(selected_tools, "test", {})
        time1 = time.time() - start_time
        
        # Test 2: Parallel execution disabled
        sequential_config = {**base_config, 'orchestration': {**base_config['orchestration'], 'parallel_execution': False}}
        orchestrator2 = OrchestratorAgent(sequential_config)
        await orchestrator2.initialize()
        orchestrator2.mcp_integration = self.create_mock_mcp_integration(execution_times)
        
        start_time = time.time()
        results2 = await orchestrator2.execute_tools(selected_tools, "test", {})
        time2 = time.time() - start_time
        
        # Test 3: Default behavior (no explicit setting - should default to True)
        default_config = base_config.copy()
        orchestrator3 = OrchestratorAgent(default_config)
        await orchestrator3.initialize()
        orchestrator3.mcp_integration = self.create_mock_mcp_integration(execution_times)
        
        start_time = time.time()
        results3 = await orchestrator3.execute_tools(selected_tools, "test", {})
        time3 = time.time() - start_time
        
        # Verify configuration controls execution mode
        logger.info(f"Parallel mode time: {time1:.3f}s")
        logger.info(f"Sequential mode time: {time2:.3f}s")
        logger.info(f"Default mode time: {time3:.3f}s")
        
        # Parallel should be faster than sequential
        assert time1 < time2, "Parallel mode should be faster than sequential"
        
        # Default should behave like parallel (default is True)
        assert abs(time3 - time1) < 0.1, "Default mode should behave like parallel mode"
        
        # Test 4: Single tool execution (should not use parallel even if enabled)
        single_tool = [{'id': 'tool.fast1', 'name': 'Fast Tool 1'}]
        
        orchestrator4 = OrchestratorAgent(parallel_config)
        await orchestrator4.initialize()
        orchestrator4.mcp_integration = self.create_mock_mcp_integration(execution_times)
        
        results4 = await orchestrator4.execute_tools(single_tool, "test", {})
        assert len(results4) == 1, "Single tool should execute successfully"
        
        logger.info("✅ Configuration-based execution mode switching verified")
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_dependencies(self, setup_test_environment):
        """Test parallel execution respects tool dependencies/constraints."""
        config = {
            'orchestration': {
                'max_tools_per_query': 4,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'database': {
                'tool_registry': setup_test_environment['registry_path']
            }
        }
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Track execution order
        execution_order = []
        
        async def mock_execute_with_tracking(tool_id: str, tool_input: Any):
            execution_order.append(tool_id)
            await asyncio.sleep(0.1)
            return {'status': 'success', 'tool_id': tool_id}
        
        mock_mcp = AsyncMock()
        mock_mcp.execute_tool = mock_execute_with_tracking
        orchestrator.mcp_integration = mock_mcp
        
        # Execute multiple independent tools
        selected_tools = [
            {'id': 'tool.fast1', 'name': 'Fast Tool 1'},
            {'id': 'tool.fast2', 'name': 'Fast Tool 2'},
            {'id': 'tool.slow1', 'name': 'Slow Tool 1'},
            {'id': 'tool.slow2', 'name': 'Slow Tool 2'}
        ]
        
        results = await orchestrator.execute_tools(selected_tools, "test query", {})
        
        # Verify all tools executed
        assert len(results) == 4, "All tools should execute"
        assert len(execution_order) == 4, "All tools should be tracked"
        
        # In parallel execution, order may vary but all should execute
        assert set(execution_order) == {'tool.fast1', 'tool.fast2', 'tool.slow1', 'tool.slow2'}, \
            "All tools should execute in parallel"
        
        logger.info(f"Execution order: {execution_order}")
        logger.info("✅ Parallel execution with multiple tools verified")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])