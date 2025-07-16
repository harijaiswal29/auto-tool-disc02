"""Integration tests for failure learning system."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import numpy as np

from src.agents.orchestrator_agent import OrchestratorAgent, ToolExecutionResult
from src.learning.q_learning_engine import QLearningEngine
from src.learning.reward_calculator import RewardCalculator, ExecutionMetrics
from src.database.database import DatabaseManager


class TestFailureLearningIntegration:
    """Integration tests for the failure learning system."""
    
    @pytest.fixture
    async def test_config(self):
        """Test configuration with learning enabled."""
        return {
            'q_learning': {
                'enable_learning': True,
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'exploration_rate': 0.2,
                'max_tools': 3
            },
            'reward_calculation': {
                'base_weights': {
                    'success': 1.0,
                    'failure': -0.5,
                    'partial_success': 0.3
                },
                'failure_penalties': {
                    'network_timeout': -0.2,
                    'permission_error': -0.8,
                    'rate_limit': -0.3
                },
                'resource_penalties': {
                    'memory_weight': 0.05,
                    'cpu_weight': 0.05,
                    'api_calls_weight': 0.1,
                    'time_weight': 0.1
                }
            }
        }
    
    @pytest.fixture
    async def orchestrator(self, test_config):
        """Create orchestrator with mocked components."""
        with patch('src.agents.orchestrator_agent.IntentRecognitionAgent') as mock_intent:
            with patch('src.agents.orchestrator_agent.ToolDiscoveryAgent') as mock_discovery:
                with patch('src.agents.orchestrator_agent.MCPIntegration') as mock_mcp:
                    with patch('src.agents.orchestrator_agent.ConversationStateMachine') as mock_state:
                        orchestrator = OrchestratorAgent(test_config)
                        
                        # Mock intent recognition
                        mock_intent_result = Mock()
                        mock_intent_result.primary_intent.type = 'query.search'
                        mock_intent_result.primary_intent.confidence = 0.8
                        mock_intent_result.primary_intent.keywords = ['find', 'files']
                        mock_intent_result.embedding = np.random.rand(384).tolist()
                        orchestrator.intent_agent.process_query = AsyncMock(return_value=mock_intent_result)
                        
                        # Mock tool discovery
                        orchestrator.discovery_agent.discover_tools = AsyncMock(return_value=[
                            {'id': 'filesystem_mcp', 'name': 'Filesystem Tool'},
                            {'id': 'search_mcp', 'name': 'Search Tool'}
                        ])
                        
                        # Initialize database
                        await orchestrator.db_manager.initialize()
                        
                        yield orchestrator
    
    @pytest.mark.asyncio
    async def test_failure_tracking_in_state(self, orchestrator):
        """Test that failures are tracked and included in state representation."""
        # Simulate a failed execution
        failed_result = ToolExecutionResult(
            tool_id='filesystem_mcp',
            tool_name='Filesystem Tool',
            success=False,
            result=None,
            error='Connection timeout',
            execution_time_ms=5000,
            error_type='network_timeout',
            retry_count=2
        )
        
        # Update failure metrics
        await orchestrator._update_failure_metrics([failed_result])
        
        # Check failure metrics are updated
        assert 'filesystem_mcp' in orchestrator.failure_metrics['failure_rates']
        assert orchestrator.failure_metrics['failure_rates']['filesystem_mcp'] > 0
        assert orchestrator.failure_metrics['failure_types']['network_timeout'] == 1
    
    @pytest.mark.asyncio
    async def test_reward_calculation_with_failures(self, orchestrator):
        """Test that reward calculation properly handles different failure types."""
        # Create different failure scenarios
        network_failure = ToolExecutionResult(
            tool_id='tool1',
            tool_name='Tool 1',
            success=False,
            result=None,
            error_type='network_timeout',
            execution_time_ms=1000
        )
        
        permission_failure = ToolExecutionResult(
            tool_id='tool2',
            tool_name='Tool 2',
            success=False,
            result=None,
            error_type='permission_error',
            execution_time_ms=500
        )
        
        # Calculate rewards
        network_reward = await orchestrator._calculate_reward([network_failure])
        permission_reward = await orchestrator._calculate_reward([permission_failure])
        
        # Permission errors should be penalized more heavily
        assert permission_reward < network_reward
        assert network_reward < 0  # Both should be negative
        assert permission_reward < -0.5  # Severe penalty
    
    @pytest.mark.asyncio
    async def test_partial_success_handling(self, orchestrator):
        """Test handling of partial success scenarios."""
        partial_result = ToolExecutionResult(
            tool_id='search_mcp',
            tool_name='Search Tool',
            success=False,
            result={'partial_data': [1, 2, 3]},
            partial_success=True,
            completion_percentage=0.6,
            result_quality=0.7,
            execution_time_ms=2000
        )
        
        full_failure = ToolExecutionResult(
            tool_id='search_mcp',
            tool_name='Search Tool',
            success=False,
            result=None,
            execution_time_ms=2000
        )
        
        partial_reward = await orchestrator._calculate_reward([partial_result])
        failure_reward = await orchestrator._calculate_reward([full_failure])
        
        # Partial success should get better reward than complete failure
        assert partial_reward > failure_reward
        assert partial_reward > -0.5  # Should be better than base failure
    
    @pytest.mark.asyncio
    async def test_resource_tracking(self, orchestrator):
        """Test resource usage tracking and penalties."""
        resource_heavy = ToolExecutionResult(
            tool_id='heavy_tool',
            tool_name='Resource Heavy Tool',
            success=True,
            result={'data': 'success'},
            execution_time_ms=10000,  # 10 seconds
            resource_usage={
                'memory_mb': 1000,  # 1GB
                'cpu_percent': 90,
                'api_calls': 100
            }
        )
        
        resource_light = ToolExecutionResult(
            tool_id='light_tool',
            tool_name='Resource Light Tool',
            success=True,
            result={'data': 'success'},
            execution_time_ms=100,
            resource_usage={
                'memory_mb': 10,
                'cpu_percent': 10,
                'api_calls': 1
            }
        )
        
        heavy_reward = await orchestrator._calculate_reward([resource_heavy])
        light_reward = await orchestrator._calculate_reward([resource_light])
        
        # Light resource usage should get better reward
        assert light_reward > heavy_reward
        # Both succeeded, so should be positive, but heavy should be penalized
        assert light_reward > 0
        assert 0 < heavy_reward < light_reward
    
    @pytest.mark.asyncio
    async def test_user_feedback_integration(self, orchestrator):
        """Test user feedback affecting rewards."""
        # Create a successful execution
        execution_id = 'test-exec-123'
        orchestrator.current_session_id = execution_id
        
        success_result = ToolExecutionResult(
            tool_id='tool1',
            tool_name='Tool 1',
            success=True,
            result={'data': 'result'},
            execution_time_ms=500
        )
        
        # Add to execution history
        orchestrator.execution_history.append({
            'execution_id': execution_id,
            'query': 'test query',
            'execution_results': [success_result],
            'reward': 0.8,
            'timestamp': datetime.now(),
            'context': {'mode': 'production'}
        })
        
        # Record positive feedback
        await orchestrator.record_user_feedback(
            execution_id=execution_id,
            feedback_type='positive',
            rating=5
        )
        
        # Check that feedback was recorded
        # In a real test, we'd verify database was updated
        assert len(orchestrator.execution_history) > 0
        assert 'user_feedback' in orchestrator.execution_history[0]
    
    @pytest.mark.asyncio
    async def test_failure_learning_over_time(self, orchestrator):
        """Test that system learns from repeated failures."""
        # Simulate multiple failures of same type
        for i in range(5):
            failure = ToolExecutionResult(
                tool_id='unreliable_tool',
                tool_name='Unreliable Tool',
                success=False,
                result=None,
                error_type='network_timeout',
                execution_time_ms=1000
            )
            
            await orchestrator._update_failure_metrics([failure])
        
        # Check failure rate increases
        failure_rate = orchestrator.failure_metrics['failure_rates'].get('unreliable_tool', 0)
        assert failure_rate > 0.4  # Should have high failure rate
        
        # Check failure type tracking
        timeout_count = orchestrator.failure_metrics['failure_types'].get('network_timeout', 0)
        assert timeout_count >= 5
    
    @pytest.mark.asyncio
    async def test_tool_synergy_recognition(self, orchestrator):
        """Test that tool synergies are recognized and rewarded."""
        # Create successful tool combination
        tool1_result = ToolExecutionResult(
            tool_id='filesystem_mcp',
            tool_name='Filesystem',
            success=True,
            result={'files': ['a.py', 'b.py']},
            execution_time_ms=100
        )
        
        tool2_result = ToolExecutionResult(
            tool_id='search_mcp',
            tool_name='Search',
            success=True,
            result={'matches': [{'file': 'a.py', 'line': 10}]},
            execution_time_ms=200
        )
        
        # Calculate reward for combination
        combo_reward = await orchestrator._calculate_reward([tool1_result, tool2_result])
        
        # Single tool reward
        single_reward = await orchestrator._calculate_reward([tool1_result])
        
        # Combination should potentially get synergy bonus
        # (depends on whether this combo is in known_synergies)
        assert combo_reward >= single_reward
    
    @pytest.mark.asyncio
    async def test_error_classification(self, orchestrator):
        """Test error classification for different error types."""
        error_types = [
            ('Connection timed out', 'network_timeout'),
            ('Permission denied: cannot access file', 'permission_error'),
            ('Rate limit exceeded', 'rate_limit'),
            ('Unknown error occurred', 'other')
        ]
        
        for error_msg, expected_type in error_types:
            error = Exception(error_msg)
            classified_type = orchestrator._classify_error(error)
            assert classified_type == expected_type
    
    @pytest.mark.asyncio 
    async def test_query_reformulation_detection(self, orchestrator):
        """Test detection of query reformulations."""
        query1 = "find Python files in project"
        query2 = "search for Python files in the project directory"
        query3 = "what is the weather today"
        
        # Similar queries should be detected as reformulation
        assert orchestrator._is_query_reformulation(query1, query2) == True
        
        # Completely different queries should not
        assert orchestrator._is_query_reformulation(query1, query3) == False
        
        # Identical queries should not be reformulation
        assert orchestrator._is_query_reformulation(query1, query1) == False
    
    @pytest.mark.asyncio
    async def test_state_encoding_with_failures(self, orchestrator):
        """Test that failure metrics are properly encoded in state."""
        # Set up failure metrics
        orchestrator.failure_metrics = {
            'failure_rates': {'tool1': 0.3, 'tool2': 0.1},
            'failure_types': {'network_timeout': 5, 'permission_error': 2},
            'retry_patterns': {
                'avg_retry_count': 2,
                'retry_success_rate': 0.6,
                'avg_retry_delay_ms': 3000,
                'circuit_breaker_triggers': 1,
                'max_consecutive_failures': 2
            }
        }
        
        # Create mock intent result
        mock_intent = Mock()
        mock_intent.embedding = np.random.rand(384).tolist()
        
        # Create context with failure metrics
        context = {
            'domain': 'engineering',
            'failure_rates': orchestrator.failure_metrics['failure_rates'],
            'failure_types': orchestrator.failure_metrics['failure_types'],
            'retry_patterns': orchestrator.failure_metrics['retry_patterns']
        }
        
        # Encode state
        state = orchestrator.q_learning_engine.state_encoder.encode_state(
            mock_intent, context, ['tool1', 'tool2']
        )
        
        # Verify state has correct dimensions (439 with new failure dimensions)
        assert len(state) == 439
        
        # Verify failure information is encoded (not all zeros)
        failure_start_idx = 384 + 10 + 20 + 5  # After intent, context, history, metrics
        failure_section = state[failure_start_idx:]
        assert not np.all(failure_section == 0)  # Should have non-zero failure data