"""
Integration tests for Orchestrator Agent Q-Learning functionality.

Tests the learning integration including:
- Q-learning based tool selection
- State encoding and representation
- Reward calculation and updates
- User feedback integration
- Failure tracking and learning
- Model persistence
"""

import pytest
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
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
from src.learning.q_learning_engine import QLearningEngine
from src.learning.reward_calculator import ExecutionMetrics, RewardCalculator
from src.learning.context_extractor import UserContext, ContextExtractor
from src.core.tool_registry import ToolRegistry
from src.database.database import DatabaseManager


class TestOrchestratorLearning:
    """Integration tests for Q-learning functionality in Orchestrator."""
    
    @pytest.fixture
    def learning_config(self, tmp_path):
        """Configuration with Q-learning enabled."""
        return {
            'orchestration': {
                'max_tools_per_query': 3,
                'tool_selection_strategy': 'q_learning',
                'parallel_execution': True
            },
            'q_learning': {
                'enable_learning': True,
                'alpha': 0.1,  # Learning rate
                'gamma': 0.9,  # Discount factor
                'epsilon': 0.3,  # Exploration rate (higher for testing)
                'model_path': str(tmp_path / 'test_q_model.pkl'),
                'state_size': 447,
                'action_space_type': 'combination'
            },
            'intent_recognition': {
                'model': 'all-MiniLM-L6-v2',
                'confidence_threshold': 0.7
            },
            'database': {
                'tool_registry': str(tmp_path / 'test_registry.db'),
                'learning_db': str(tmp_path / 'test_learning.db')
            },
            'reward': {
                'base_rewards': {
                    'success': 1.0,
                    'failure': -0.5,
                    'partial_success': 0.3
                },
                'performance_weights': {
                    'execution_time': 0.2,
                    'resource_usage': 0.1,
                    'result_quality': 0.3
                }
            }
        }
    
    @pytest.fixture
    async def setup_learning_environment(self, learning_config):
        """Set up environment with Q-learning components."""
        # Create tool registry with test tools
        registry = ToolRegistry(learning_config['database']['tool_registry'])
        
        # Add diverse test tools
        test_tools = [
            {
                'id': 'fast.search',
                'name': 'Fast Search',
                'type': 'search',
                'performance_score': 0.95,
                'avg_execution_time': 50  # ms
            },
            {
                'id': 'slow.search',
                'name': 'Slow Search',
                'type': 'search',
                'performance_score': 0.98,
                'avg_execution_time': 500  # ms
            },
            {
                'id': 'reliable.db',
                'name': 'Reliable Database',
                'type': 'database',
                'performance_score': 0.90,
                'failure_rate': 0.05
            },
            {
                'id': 'unstable.db',
                'name': 'Unstable Database',
                'type': 'database',
                'performance_score': 0.70,
                'failure_rate': 0.30
            },
            {
                'id': 'efficient.processor',
                'name': 'Efficient Processor',
                'type': 'processor',
                'performance_score': 0.85,
                'resource_efficiency': 0.9
            }
        ]
        
        for tool in test_tools:
            registry.register_tool({
                'id': tool['id'],
                'name': tool['name'],
                'type': tool['type'],
                'server': f"{tool['type']}_mcp",
                'capabilities': {
                    'operations': [tool['type'], 'process', 'analyze']
                },
                'status': 'active',
                'performance_score': tool['performance_score']
            })
        
        # Add complementary relationships
        await registry.add_tool_relationship('fast.search', 'efficient.processor', 'complements')
        await registry.add_tool_relationship('reliable.db', 'efficient.processor', 'complements')
        
        return {
            'config': learning_config,
            'registry': registry,
            'test_tools': test_tools
        }
    
    @pytest.fixture
    async def learning_orchestrator(self, setup_learning_environment):
        """Create orchestrator with Q-learning enabled."""
        config = setup_learning_environment['config']
        
        # Create orchestrator
        orchestrator = OrchestratorAgent(config)
        
        # Mock MCP integration
        mock_mcp = AsyncMock()
        orchestrator.mcp_integration = mock_mcp
        
        # Create execution behavior based on tool characteristics
        async def smart_mock_execute(tool_id, tool_input):
            """Mock execution with tool-specific behavior."""
            tool_info = next(
                (t for t in setup_learning_environment['test_tools'] if t['id'] == tool_id),
                None
            )
            
            if not tool_info:
                raise ValueError(f"Unknown tool: {tool_id}")
            
            # Simulate execution time
            exec_time = tool_info.get('avg_execution_time', 100)
            await asyncio.sleep(exec_time / 1000.0)  # Convert to seconds
            
            # Simulate failures based on failure rate
            failure_rate = tool_info.get('failure_rate', 0.1)
            if np.random.random() < failure_rate:
                raise Exception(f"{tool_id} failed due to random error")
            
            # Return realistic results
            if 'search' in tool_id:
                quality = 0.9 if 'fast' in tool_id else 0.95
                return {
                    'results': [f'Result {i}' for i in range(int(quality * 10))],
                    'quality_score': quality
                }
            elif 'db' in tool_id:
                return {
                    'rows': [{'id': i, 'data': f'Row {i}'} for i in range(5)],
                    'execution_time_ms': exec_time
                }
            else:
                return {
                    'processed': True,
                    'efficiency': tool_info.get('resource_efficiency', 0.7)
                }
        
        mock_mcp.execute_tool = AsyncMock(side_effect=smart_mock_execute)
        mock_mcp.initialize = AsyncMock()
        mock_mcp.shutdown = AsyncMock()
        
        # Initialize orchestrator
        await orchestrator.initialize()
        
        # Ensure Q-learning is initialized
        assert orchestrator.q_learning_engine is not None
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_q_learning_tool_selection(self, learning_orchestrator):
        """Test Q-learning based tool selection."""
        orchestrator = learning_orchestrator
        
        # Create intent for testing
        intent_result = IntentResult(
            raw_query="Search and process data efficiently",
            primary_intent=Intent(
                type='query.search',
                keywords=['search', 'process', 'efficient'],
                confidence=0.85
            ),
            alternative_intents=[],
            context={'user_preference': 'speed'},
            processing_time_ms=30.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Track tool selections over multiple queries
        selected_tools_history = []
        
        # Run multiple queries to observe learning
        for i in range(5):
            result = await orchestrator.process_user_query(
                f"Search and process data efficiently - iteration {i}"
            )
            
            selected_tools_history.append(result.selected_tools)
            
            # Verify Q-learning was used
            assert orchestrator.current_state is not None
            assert len(result.selected_tools) > 0
            assert len(result.selected_tools) <= orchestrator.config['orchestration']['max_tools_per_query']
        
        # Q-learning should explore different combinations
        unique_combinations = set(tuple(sorted(tools)) for tools in selected_tools_history)
        assert len(unique_combinations) > 1  # Should try different combinations
        
        # Check that complementary tools are sometimes selected together
        complementary_selections = sum(
            1 for tools in selected_tools_history
            if 'fast.search' in tools and 'efficient.processor' in tools
        )
        assert complementary_selections > 0
    
    @pytest.mark.asyncio
    async def test_state_encoding_and_representation(self, learning_orchestrator):
        """Test state encoding includes all relevant features."""
        orchestrator = learning_orchestrator
        
        # Create diverse intents to test state encoding
        test_cases = [
            {
                'query': 'Simple search query',
                'intent_type': 'query.search',
                'confidence': 0.95,
                'keywords': ['simple', 'search'],
                'context': {'domain': 'general', 'user_expertise': 'beginner'}
            },
            {
                'query': 'Complex database analysis with exports',
                'intent_type': 'query.analyze',
                'confidence': 0.75,
                'keywords': ['complex', 'database', 'analysis', 'export'],
                'context': {'domain': 'analytics', 'user_expertise': 'expert'}
            }
        ]
        
        encoded_states = []
        
        for test_case in test_cases:
            intent_result = IntentResult(
                raw_query=test_case['query'],
                primary_intent=Intent(
                    type=test_case['intent_type'],
                    keywords=test_case['keywords'],
                    confidence=test_case['confidence']
                ),
                alternative_intents=[],
                context=test_case['context'],
                processing_time_ms=25.0
            )
            
            # Create user context
            user_context = UserContext(
                user_id='test_user',
                session_id='test_session',
                timestamp=datetime.now(),
                user_expertise=test_case['context']['user_expertise'],
                domain=test_case['context']['domain'],
                query_complexity='complex' if 'complex' in test_case['query'] else 'simple',
                time_of_day='business_hours',
                day_of_week='weekday'
            )
            
            orchestrator.context['user_context'] = user_context
            orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
            
            # Process query to trigger state encoding
            await orchestrator.process_user_query(test_case['query'])
            
            # Capture encoded state
            encoded_states.append(orchestrator.current_state)
        
        # Verify state encoding
        assert len(encoded_states) == len(test_cases)
        
        # States should be different for different contexts
        assert not np.array_equal(encoded_states[0], encoded_states[1])
        
        # Check state dimensions
        for state in encoded_states:
            assert state is not None
            assert len(state) == orchestrator.config['q_learning']['state_size']
            assert state.dtype == np.float32
            assert np.all(np.isfinite(state))  # No NaN or inf values
    
    @pytest.mark.asyncio
    async def test_reward_calculation_scenarios(self, learning_orchestrator):
        """Test reward calculation for various execution scenarios."""
        orchestrator = learning_orchestrator
        
        # Test scenarios with expected reward ranges
        scenarios = [
            {
                'name': 'All tools succeed quickly',
                'results': [
                    ToolExecutionResult(
                        tool_id='fast.search',
                        tool_name='Fast Search',
                        success=True,
                        result={'results': ['r1', 'r2', 'r3']},
                        execution_time_ms=50,
                        result_quality=0.9,
                        resource_usage={'memory_mb': 10, 'cpu_percent': 20}
                    ),
                    ToolExecutionResult(
                        tool_id='efficient.processor',
                        tool_name='Efficient Processor',
                        success=True,
                        result={'processed': True},
                        execution_time_ms=100,
                        result_quality=0.85,
                        resource_usage={'memory_mb': 50, 'cpu_percent': 30}
                    )
                ],
                'expected_reward_min': 0.8,  # High reward
                'expected_reward_max': 2.0
            },
            {
                'name': 'Partial success with retries',
                'results': [
                    ToolExecutionResult(
                        tool_id='unstable.db',
                        tool_name='Unstable Database',
                        success=False,
                        result=None,
                        error='Connection failed',
                        execution_time_ms=200,
                        partial_success=True,
                        completion_percentage=0.6,
                        retry_count=2,
                        error_type='connection_error'
                    ),
                    ToolExecutionResult(
                        tool_id='fast.search',
                        tool_name='Fast Search',
                        success=True,
                        result={'results': ['r1']},
                        execution_time_ms=80,
                        result_quality=0.7
                    )
                ],
                'expected_reward_min': 0.0,  # Mixed results
                'expected_reward_max': 0.6
            },
            {
                'name': 'Complete failure',
                'results': [
                    ToolExecutionResult(
                        tool_id='unstable.db',
                        tool_name='Unstable Database',
                        success=False,
                        result=None,
                        error='Fatal error',
                        execution_time_ms=500,
                        error_type='non_retryable'
                    )
                ],
                'expected_reward_min': -1.0,  # Negative reward
                'expected_reward_max': -0.3
            }
        ]
        
        for scenario in scenarios:
            # Calculate reward
            reward = await orchestrator._calculate_reward(scenario['results'])
            
            # Verify reward is in expected range
            assert scenario['expected_reward_min'] <= reward <= scenario['expected_reward_max'], \
                f"Scenario '{scenario['name']}': reward {reward} not in expected range " \
                f"[{scenario['expected_reward_min']}, {scenario['expected_reward_max']}]"
    
    @pytest.mark.asyncio
    async def test_learning_from_experience(self, learning_orchestrator):
        """Test that Q-values are updated based on experience."""
        orchestrator = learning_orchestrator
        
        # Get initial Q-values for a state-action pair
        intent_result = IntentResult(
            raw_query="Test query for learning",
            primary_intent=Intent(
                type='query.search',
                keywords=['test', 'learning'],
                confidence=0.8
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=20.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Capture Q-values before and after learning
        q_values_before = {}
        q_values_after = {}
        
        # Run query and track Q-values
        result1 = await orchestrator.process_user_query("Test query for learning")
        
        # Get Q-values for the state and executed action
        if orchestrator.current_state is not None and result1.selected_tools:
            state_key = orchestrator.q_learning_engine._hash_state(orchestrator.current_state)
            action_key = tuple(sorted(result1.selected_tools))
            
            # Store initial Q-value
            q_table = orchestrator.q_learning_engine.q_table
            if state_key in q_table and action_key in q_table[state_key]:
                q_values_before[action_key] = q_table[state_key][action_key]
        
        # Simulate multiple experiences with the same state-action
        for i in range(3):
            await orchestrator.process_user_query("Test query for learning")
        
        # Check Q-values after learning
        if state_key in q_table and action_key in q_table[state_key]:
            q_values_after[action_key] = q_table[state_key][action_key]
        
        # Q-values should change after experiences
        if action_key in q_values_before and action_key in q_values_after:
            assert q_values_before[action_key] != q_values_after[action_key], \
                "Q-values should update after learning from experience"
    
    @pytest.mark.asyncio
    async def test_exploration_vs_exploitation(self, learning_orchestrator):
        """Test balance between exploration and exploitation."""
        orchestrator = learning_orchestrator
        
        # Set different exploration rates
        original_epsilon = orchestrator.q_learning_engine.exploration_rate
        
        # Test high exploration (should see variety)
        orchestrator.q_learning_engine.exploration_rate = 0.9
        high_exploration_selections = []
        
        intent_result = IntentResult(
            raw_query="Standard query",
            primary_intent=Intent(
                type='query.search',
                keywords=['standard', 'query'],
                confidence=0.85
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=15.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        for i in range(10):
            result = await orchestrator.process_user_query(f"Standard query {i}")
            high_exploration_selections.append(tuple(sorted(result.selected_tools)))
        
        # Test low exploration (should converge)
        orchestrator.q_learning_engine.exploration_rate = 0.1
        low_exploration_selections = []
        
        for i in range(10):
            result = await orchestrator.process_user_query(f"Standard query {i}")
            low_exploration_selections.append(tuple(sorted(result.selected_tools)))
        
        # High exploration should have more variety
        unique_high = len(set(high_exploration_selections))
        unique_low = len(set(low_exploration_selections))
        
        assert unique_high >= unique_low, \
            "High exploration should produce more variety in tool selection"
        
        # Restore original epsilon
        orchestrator.q_learning_engine.exploration_rate = original_epsilon
    
    @pytest.mark.asyncio
    async def test_failure_tracking_and_learning(self, learning_orchestrator):
        """Test that the system learns from failures."""
        orchestrator = learning_orchestrator
        
        # Force specific tools to fail
        original_execute = orchestrator.mcp_integration.execute_tool
        fail_count = {'unstable.db': 0}
        
        async def mock_with_failures(tool_id, tool_input):
            if tool_id == 'unstable.db':
                fail_count['unstable.db'] += 1
                raise Exception("Database connection failed")
            return await original_execute(tool_id, tool_input)
        
        orchestrator.mcp_integration.execute_tool = mock_with_failures
        
        intent_result = IntentResult(
            raw_query="Query that needs database",
            primary_intent=Intent(
                type='query.retrieve',
                keywords=['query', 'database'],
                confidence=0.9
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=18.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Track how often unstable.db is selected
        unstable_selections = []
        
        for i in range(10):
            result = await orchestrator.process_user_query(f"Query that needs database {i}")
            unstable_selected = 'unstable.db' in result.selected_tools
            unstable_selections.append(unstable_selected)
        
        # Should learn to avoid unstable.db over time
        early_selections = sum(unstable_selections[:5])
        late_selections = sum(unstable_selections[5:])
        
        assert late_selections <= early_selections, \
            "System should learn to avoid failing tools over time"
        
        # Check failure metrics are updated
        assert 'unstable.db' in orchestrator.failure_metrics['failure_rates']
        assert orchestrator.failure_metrics['failure_rates']['unstable.db'] > 0
    
    @pytest.mark.asyncio
    async def test_user_feedback_integration(self, learning_orchestrator):
        """Test integration of user feedback into learning."""
        orchestrator = learning_orchestrator
        
        # Process a query
        intent_result = IntentResult(
            raw_query="Search for important information",
            primary_intent=Intent(
                type='query.search',
                keywords=['search', 'important', 'information'],
                confidence=0.88
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=22.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        result = await orchestrator.process_user_query("Search for important information")
        execution_id = orchestrator.current_session_id
        
        # Simulate positive user feedback
        await orchestrator.record_user_feedback(
            execution_id=execution_id,
            feedback_type='positive',
            rating=5,
            follow_up_query=None
        )
        
        # Check that feedback was recorded
        matching_history = [
            h for h in orchestrator.execution_history
            if h.get('execution_id') == execution_id
        ]
        
        assert len(matching_history) > 0
        if 'user_feedback' in matching_history[0]:
            assert matching_history[0]['user_feedback']['rating'] == 5
        
        # Simulate negative feedback with reformulation
        result2 = await orchestrator.process_user_query("Find specific data in database")
        execution_id2 = orchestrator.current_session_id
        
        await orchestrator.record_user_feedback(
            execution_id=execution_id2,
            feedback_type='negative',
            rating=2,
            follow_up_query="Find customer data in sales database"
        )
        
        # Test query reformulation detection
        is_reformulation = orchestrator._is_query_reformulation(
            "Find specific data in database",
            "Find customer data in sales database"
        )
        assert is_reformulation
    
    @pytest.mark.asyncio
    async def test_context_aware_tool_selection(self, learning_orchestrator):
        """Test that tool selection adapts based on user context."""
        orchestrator = learning_orchestrator
        
        # Test different user contexts
        contexts = [
            {
                'expertise': 'beginner',
                'domain': 'general',
                'expected_preference': 'simple'  # Should prefer simpler, more reliable tools
            },
            {
                'expertise': 'expert',
                'domain': 'analytics',
                'expected_preference': 'advanced'  # Can handle more complex tools
            }
        ]
        
        selections_by_context = {}
        
        for ctx in contexts:
            # Set user context
            user_context = UserContext(
                user_id='test_user',
                session_id=f"session_{ctx['expertise']}",
                timestamp=datetime.now(),
                user_expertise=ctx['expertise'],
                domain=ctx['domain'],
                query_complexity='simple',
                time_of_day='business_hours',
                day_of_week='weekday'
            )
            
            orchestrator.context['user_context'] = user_context
            orchestrator.user_stats['success_rate'] = 0.8 if ctx['expertise'] == 'expert' else 0.5
            
            intent_result = IntentResult(
                raw_query="Analyze data",
                primary_intent=Intent(
                    type='query.analyze',
                    keywords=['analyze', 'data'],
                    confidence=0.85
                ),
                alternative_intents=[],
                context={'domain': ctx['domain']},
                processing_time_ms=20.0
            )
            
            orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
            
            # Run multiple queries to see patterns
            selected_tools = []
            for i in range(5):
                result = await orchestrator.process_user_query(f"Analyze data - {ctx['expertise']}")
                selected_tools.extend(result.selected_tools)
            
            selections_by_context[ctx['expertise']] = selected_tools
        
        # Beginners should see more reliable tools selected
        beginner_tools = selections_by_context['beginner']
        expert_tools = selections_by_context['expert']
        
        # Count selections of reliable vs unstable tools
        beginner_reliable = sum(1 for t in beginner_tools if 'reliable' in t or 'fast' in t)
        expert_unstable = sum(1 for t in expert_tools if 'unstable' in t or 'slow' in t)
        
        # Beginners should prefer reliable tools more
        assert beginner_reliable > 0, "Beginners should be given reliable tools"
    
    @pytest.mark.asyncio
    async def test_model_persistence(self, learning_orchestrator, tmp_path):
        """Test saving and loading Q-learning model."""
        orchestrator = learning_orchestrator
        
        # Run some queries to build Q-table
        intent_result = IntentResult(
            raw_query="Test persistence",
            primary_intent=Intent(
                type='query.search',
                keywords=['test', 'persistence'],
                confidence=0.9
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=15.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Execute multiple queries to build experience
        for i in range(5):
            await orchestrator.process_user_query(f"Test persistence query {i}")
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        orchestrator.config['q_learning']['model_path'] = str(model_path)
        await orchestrator.q_learning_engine.save_model()
        
        assert model_path.exists()
        
        # Create new orchestrator and load model
        new_orchestrator = OrchestratorAgent(orchestrator.config)
        new_orchestrator.mcp_integration = orchestrator.mcp_integration
        new_orchestrator.intent_agent = orchestrator.intent_agent
        
        # Load the saved model
        await new_orchestrator.q_learning_engine.load_model()
        
        # Verify Q-table was loaded
        assert len(new_orchestrator.q_learning_engine.q_table) > 0
        
        # Should make similar decisions for same query
        result1 = await orchestrator.process_user_query("Test persistence verification")
        result2 = await new_orchestrator.process_user_query("Test persistence verification")
        
        # With low exploration, should select similar tools
        if orchestrator.q_learning_engine.exploration_rate < 0.3:
            assert set(result1.selected_tools) == set(result2.selected_tools)
    
    @pytest.mark.asyncio
    async def test_adaptive_exploration_decay(self, learning_orchestrator):
        """Test that exploration rate decays over time."""
        orchestrator = learning_orchestrator
        
        initial_epsilon = orchestrator.q_learning_engine.exploration_rate
        
        intent_result = IntentResult(
            raw_query="Repeated query",
            primary_intent=Intent(
                type='query.search',
                keywords=['repeated', 'query'],
                confidence=0.85
            ),
            alternative_intents=[],
            context={},
            processing_time_ms=12.0
        )
        
        orchestrator.intent_agent.process_query = AsyncMock(return_value=intent_result)
        
        # Run multiple queries
        for i in range(10):
            await orchestrator.process_user_query(f"Repeated query {i}")
        
        # Check exploration rate has decayed
        final_epsilon = orchestrator.q_learning_engine.exploration_rate
        assert final_epsilon < initial_epsilon, \
            f"Exploration rate should decay: {initial_epsilon} -> {final_epsilon}"
        
        # But not below minimum
        min_epsilon = orchestrator.q_learning_engine.min_epsilon
        assert final_epsilon >= min_epsilon, \
            f"Exploration rate should not go below minimum: {final_epsilon} >= {min_epsilon}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])