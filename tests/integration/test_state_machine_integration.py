"""
Integration tests for state machine with other system components.

Tests the integration of state machine with pipeline stages, agents,
and complete conversation flows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from src.state_machine.conversation_state_machine import (
    ConversationStateMachine, ConversationStates
)
from src.state_machine.handlers import register_handlers
from src.pipeline.stages.state_manager import StateManagerStage
from src.pipeline.base import PipelineData
from src.agents.intent_recognition_agent import IntentResult, Intent


class TestStateManagerIntegration:
    """Test StateManagerStage integration with pipeline."""
    
    @pytest.fixture
    async def state_manager(self):
        """Create state manager stage."""
        config = {
            'auto_transition': True,
            'track_metrics': True
        }
        manager = StateManagerStage(config)
        await manager.initialize()
        yield manager
    
    @pytest.fixture
    def pipeline_data(self):
        """Create sample pipeline data."""
        data = PipelineData(
            raw_input="Find Python files in the project",
            context={"user_id": "test_user", "session_id": "test_session"}
        )
        return data
    
    @pytest.mark.asyncio
    async def test_state_manager_initialization(self, state_manager):
        """Test state manager initializes properly."""
        assert state_manager.state_machine is not None
        assert state_manager.state_machine.is_in_state(ConversationStates.IDLE)
        assert state_manager.auto_transition is True
        assert state_manager.track_metrics is True
    
    @pytest.mark.asyncio
    async def test_query_reception_flow(self, state_manager, pipeline_data):
        """Test state updates during query reception."""
        # Process initial query
        result = await state_manager.process(pipeline_data)
        
        # Should transition to QUERY_RECEIVED
        assert result.get_metadata('conversation_state') == ConversationStates.QUERY_RECEIVED
        
        # Should add state info to results
        state_info = result.get_stage_result('StateManager', 'current_state')
        assert state_info == ConversationStates.QUERY_RECEIVED
        
        # Should track conversation summary
        summary = result.get_stage_result('StateManager', 'conversation_summary')
        assert summary['query'] == "Find Python files in the project"
    
    @pytest.mark.asyncio
    async def test_intent_recognition_flow(self, state_manager, pipeline_data):
        """Test state updates during intent recognition."""
        # First process query
        await state_manager.process(pipeline_data)
        
        # Add intent recognition results
        intent = Intent(
            type="query.search",
            keywords=["find", "python", "files"],
            confidence=0.85,
            entities={"language": "Python", "target": "files"}
        )
        pipeline_data.add_stage_result('ConfidenceScorer', 'primary_intent', intent)
        pipeline_data.add_stage_result('ConfidenceScorer', 'confidence_passed', True)
        
        # Process with intent results
        result = await state_manager.process(pipeline_data)
        
        # Should transition to INTENT_RECOGNIZED
        assert result.get_metadata('conversation_state') == ConversationStates.INTENT_RECOGNIZED
        
        # Context should have intent info
        state_context = result.get_metadata('state_machine_context')
        assert state_context['intent']['type'] == "query.search"
        assert state_context['confidence'] == 0.85
    
    @pytest.mark.asyncio
    async def test_low_confidence_clarification_flow(self, state_manager, pipeline_data):
        """Test clarification flow for low confidence intent."""
        # Process query
        await state_manager.process(pipeline_data)
        
        # Add low confidence intent
        intent = Intent(
            type="unknown",
            keywords=[],
            confidence=0.3,
            entities={}
        )
        pipeline_data.add_stage_result('ConfidenceScorer', 'primary_intent', intent)
        pipeline_data.add_stage_result('ConfidenceScorer', 'confidence_passed', False)
        
        # Process with low confidence
        result = await state_manager.process(pipeline_data)
        
        # Should transition to CLARIFICATION_NEEDED
        assert result.get_metadata('conversation_state') == ConversationStates.CLARIFICATION_NEEDED
        
        # Should halt pipeline
        assert result.get_metadata('pipeline_halt') is True
        assert result.get_metadata('halt_reason') == 'State machine flow control'
    
    @pytest.mark.asyncio
    async def test_error_handling_flow(self, state_manager, pipeline_data):
        """Test error handling in state manager."""
        # Add error to pipeline data
        pipeline_data.add_metadata('error', 'Processing failed')
        
        # Process with error
        result = await state_manager.process(pipeline_data)
        
        # Should transition to ERROR state
        assert result.get_metadata('conversation_state') == ConversationStates.ERROR
        
        # Should halt pipeline
        assert result.get_metadata('pipeline_halt') is True
        
        # Error should be in context
        state_context = result.get_metadata('state_machine_context')
        assert 'error' in state_context
        assert state_context['error'] == 'Processing failed'
    
    @pytest.mark.asyncio
    async def test_tool_discovery_integration(self, state_manager):
        """Test tool discovery state transitions."""
        # Handle tool discovery
        tools = ["filesystem_mcp", "search_mcp", "git_mcp"]
        result = await state_manager.handle_tool_discovery(tools)
        assert result is True
        
        # Should be in TOOLS_DISCOVERED state
        assert state_manager.get_current_state_name() == ConversationStates.TOOLS_DISCOVERED
        
        # Context should have tools
        context = state_manager.state_machine.get_context()
        assert context['discovered_tools'] == tools
    
    @pytest.mark.asyncio
    async def test_execution_flow(self, state_manager):
        """Test execution state transitions."""
        # Setup: get to tools discovered state
        await state_manager.state_machine.receive_query("Test", {})
        await state_manager.state_machine.recognize_intent({"type": "test"}, 0.9)
        await state_manager.handle_tool_discovery(["test_tool"])
        
        # Start execution
        result = await state_manager.handle_execution_start(["test_tool"])
        assert result is True
        assert state_manager.get_current_state_name() == ConversationStates.EXECUTION_STARTED
        
        # Complete execution
        results = [{"tool": "test_tool", "result": "Success"}]
        result = await state_manager.handle_execution_complete(results, success=True)
        assert result is True
        assert state_manager.get_current_state_name() == ConversationStates.EXECUTION_COMPLETE
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, state_manager):
        """Test retry handling."""
        # Setup: get to execution failed state
        await state_manager.state_machine.receive_query("Test", {})
        await state_manager.state_machine.recognize_intent({"type": "test"}, 0.9)
        await state_manager.handle_tool_discovery(["test_tool"])
        await state_manager.handle_execution_start(["test_tool"])
        await state_manager.handle_execution_complete([], success=False)
        
        # Request retry
        result = await state_manager.request_retry()
        assert result is True
        assert state_manager.get_current_state_name() == ConversationStates.RETRY_REQUESTED
    
    @pytest.mark.asyncio
    async def test_cancellation_flow(self, state_manager):
        """Test operation cancellation."""
        # Start some operation
        await state_manager.state_machine.receive_query("Test", {})
        await state_manager.state_machine.recognize_intent({"type": "test"}, 0.9)
        
        # Cancel
        result = await state_manager.cancel_operation()
        assert result is True
        assert state_manager.get_current_state_name() == ConversationStates.USER_CANCELLED
    
    @pytest.mark.asyncio
    async def test_state_metrics_tracking(self, state_manager, pipeline_data):
        """Test state metrics are tracked correctly."""
        # Execute several transitions
        await state_manager.process(pipeline_data)
        
        # Add intent
        intent = Intent("query.search", ["test"], 0.9)
        pipeline_data.add_stage_result('ConfidenceScorer', 'primary_intent', intent)
        await state_manager.process(pipeline_data)
        
        # Check metrics
        metrics = pipeline_data.get_stage_result('StateManager', 'state_metrics')
        assert metrics['state_transitions'] >= 2
        assert metrics['error_count'] == 0
        assert metrics['retry_count'] == 0
        assert metrics['clarification_attempts'] == 0
    
    @pytest.mark.asyncio
    async def test_needs_user_input_detection(self, state_manager):
        """Test detection of states requiring user input."""
        # Initially should not need input
        assert state_manager.needs_user_input() is False
        
        # Get to clarification needed state
        await state_manager.state_machine.receive_query("Unclear", {})
        await state_manager.state_machine.recognize_intent({"type": "unknown"}, 0.3)
        
        # Should need user input
        assert state_manager.needs_user_input() is True
        assert state_manager.get_current_state_name() == ConversationStates.CLARIFICATION_NEEDED
    
    @pytest.mark.asyncio
    async def test_error_state_detection(self, state_manager):
        """Test detection of error states."""
        # Initially not in error state
        assert state_manager.is_in_error_state() is False
        
        # Cause error
        await state_manager.state_machine.handle_error(Exception("Test"), {})
        
        # Should be in error state
        assert state_manager.is_in_error_state() is True
        assert state_manager.get_current_state_name() == ConversationStates.ERROR
    
    @pytest.mark.asyncio
    async def test_state_history_retrieval(self, state_manager):
        """Test retrieving state transition history."""
        # Execute several transitions
        await state_manager.state_machine.receive_query("Test", {})
        await state_manager.state_machine.recognize_intent({"type": "test"}, 0.9)
        await state_manager.handle_tool_discovery(["tool1", "tool2"])
        
        # Get history
        history = state_manager.get_state_history(limit=5)
        
        assert len(history) >= 3
        assert history[0]['to_state'] == ConversationStates.TOOLS_DISCOVERED
        assert history[1]['to_state'] == ConversationStates.INTENT_RECOGNIZED
        assert history[2]['to_state'] == ConversationStates.QUERY_RECEIVED


class TestCompleteConversationFlows:
    """Test complete conversation flows with state machine."""
    
    @pytest.fixture
    async def conversation_system(self):
        """Create a complete conversation system."""
        # Create state machine with handlers
        state_machine = ConversationStateMachine()
        register_handlers(state_machine)
        await state_machine.start()
        
        # Create state manager
        state_manager = StateManagerStage()
        await state_manager.initialize()
        
        return {
            'state_machine': state_machine,
            'state_manager': state_manager
        }
    
    @pytest.mark.asyncio
    async def test_successful_query_flow(self, conversation_system):
        """Test successful end-to-end query processing."""
        sm = conversation_system['state_machine']
        
        # Execute complete flow
        assert await sm.receive_query("Find all Python files", {"user": "test"})
        assert sm.is_in_state(ConversationStates.QUERY_RECEIVED)
        
        intent = {
            "type": "query.search",
            "keywords": ["find", "python", "files"],
            "confidence": 0.92
        }
        assert await sm.recognize_intent(intent, 0.92)
        assert sm.is_in_state(ConversationStates.INTENT_RECOGNIZED)
        
        tools = ["filesystem_mcp", "search_mcp"]
        assert await sm.discover_tools(tools)
        assert sm.is_in_state(ConversationStates.TOOLS_DISCOVERED)
        
        assert await sm.start_execution(["filesystem_mcp"])
        assert sm.is_in_state(ConversationStates.EXECUTION_STARTED)
        
        results = [
            {
                "tool": "filesystem_mcp",
                "result": ["main.py", "utils.py", "test.py"],
                "count": 3
            }
        ]
        assert await sm.complete_execution(results, success=True)
        assert sm.is_in_state(ConversationStates.EXECUTION_COMPLETE)
        
        # Get summary
        summary = sm.get_conversation_summary()
        assert summary['query'] == "Find all Python files"
        assert summary['intent']['type'] == "query.search"
        assert summary['execution_success'] is True
        assert len(summary['execution_results']) == 1
    
    @pytest.mark.asyncio
    async def test_clarification_flow(self, conversation_system):
        """Test flow requiring clarification."""
        sm = conversation_system['state_machine']
        
        # Ambiguous query
        await sm.receive_query("Do the thing", {})
        
        # Low confidence intent
        await sm.recognize_intent({"type": "unknown", "keywords": ["thing"]}, 0.25)
        assert sm.is_in_state(ConversationStates.CLARIFICATION_NEEDED)
        
        # Provide clarification
        await sm.handle_clarification("I want to search for configuration files")
        assert sm.is_in_state(ConversationStates.CLARIFICATION_RECEIVED)
        
        # Continue with clarified intent
        await sm.transition_to(ConversationStates.QUERY_RECEIVED)
        await sm.recognize_intent({"type": "query.search", "keywords": ["config"]}, 0.85)
        assert sm.is_in_state(ConversationStates.INTENT_RECOGNIZED)
    
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, conversation_system):
        """Test error handling and recovery flow."""
        sm = conversation_system['state_machine']
        
        # Start normal flow
        await sm.receive_query("Create new file", {})
        await sm.recognize_intent({"type": "action.create"}, 0.9)
        await sm.discover_tools(["filesystem_mcp"])
        await sm.start_execution(["filesystem_mcp"])
        
        # Execution fails
        await sm.complete_execution(
            [{"tool": "filesystem_mcp", "error": "Permission denied"}],
            success=False
        )
        assert sm.is_in_state(ConversationStates.EXECUTION_FAILED)
        
        # Request retry
        await sm.request_retry()
        assert sm.is_in_state(ConversationStates.RETRY_REQUESTED)
        
        # Retry execution
        await sm.transition_to(ConversationStates.EXECUTION_STARTED)
        
        # Success this time
        await sm.complete_execution(
            [{"tool": "filesystem_mcp", "result": "File created"}],
            success=True
        )
        assert sm.is_in_state(ConversationStates.EXECUTION_COMPLETE)
    
    @pytest.mark.asyncio
    async def test_timeout_handling_flow(self, conversation_system):
        """Test timeout handling in conversation."""
        sm = conversation_system['state_machine']
        
        # Start execution
        await sm.receive_query("Long running task", {})
        await sm.recognize_intent({"type": "action.analyze"}, 0.9)
        await sm.discover_tools(["analyzer_mcp"])
        await sm.start_execution(["analyzer_mcp"])
        
        # Simulate timeout
        await sm.handle_timeout()
        assert sm.is_in_state(ConversationStates.TIMEOUT)
        
        # Can retry or return to idle
        await sm.return_to_idle()
        assert sm.is_in_state(ConversationStates.IDLE)
    
    @pytest.mark.asyncio
    async def test_no_tools_found_flow(self, conversation_system):
        """Test flow when no tools are found."""
        sm = conversation_system['state_machine']
        
        # Query for something with no tools
        await sm.receive_query("Do something impossible", {})
        await sm.recognize_intent({"type": "action.impossible"}, 0.95)
        
        # No tools discovered
        await sm.discover_tools([])
        assert sm.is_in_state(ConversationStates.NO_TOOLS_FOUND)
        
        # User can retry with different query or cancel
        await sm.return_to_idle()
        assert sm.is_in_state(ConversationStates.IDLE)
    
    @pytest.mark.asyncio
    async def test_feedback_collection_flow(self, conversation_system):
        """Test feedback collection after execution."""
        sm = conversation_system['state_machine']
        
        # Complete successful execution
        await sm.receive_query("Search for logs", {})
        await sm.recognize_intent({"type": "query.search"}, 0.9)
        await sm.discover_tools(["log_analyzer"])
        await sm.start_execution(["log_analyzer"])
        await sm.complete_execution([{"result": "10 logs found"}], success=True)
        
        # Collect feedback
        feedback = {
            "rating": 5,
            "helpful": True,
            "comment": "Found exactly what I needed"
        }
        await sm.receive_feedback(feedback)
        assert sm.is_in_state(ConversationStates.FEEDBACK_RECEIVED)
        
        # Return to idle for next query
        await sm.return_to_idle()
        assert sm.is_in_state(ConversationStates.IDLE)
    
    @pytest.mark.asyncio
    async def test_cancellation_at_various_stages(self, conversation_system):
        """Test cancellation can happen at different stages."""
        sm = conversation_system['state_machine']
        
        # Test 1: Cancel during intent recognition
        await sm.receive_query("Test 1", {})
        await sm.cancel_operation()
        assert sm.is_in_state(ConversationStates.USER_CANCELLED)
        
        # Reset
        await sm.return_to_idle()
        
        # Test 2: Cancel during tool discovery
        await sm.receive_query("Test 2", {})
        await sm.recognize_intent({"type": "test"}, 0.9)
        await sm.cancel_operation()
        assert sm.is_in_state(ConversationStates.USER_CANCELLED)
        
        # Reset
        await sm.return_to_idle()
        
        # Test 3: Cancel during execution
        await sm.receive_query("Test 3", {})
        await sm.recognize_intent({"type": "test"}, 0.9)
        await sm.discover_tools(["test_tool"])
        await sm.start_execution(["test_tool"])
        await sm.cancel_operation()
        assert sm.is_in_state(ConversationStates.USER_CANCELLED)


class TestConcurrentStateMachines:
    """Test multiple concurrent state machines."""
    
    @pytest.mark.asyncio
    async def test_multiple_conversations(self):
        """Test multiple independent conversations."""
        # Create multiple state machines
        sm1 = ConversationStateMachine()
        sm2 = ConversationStateMachine()
        sm3 = ConversationStateMachine()
        
        # Start all
        await asyncio.gather(
            sm1.start(),
            sm2.start(),
            sm3.start()
        )
        
        # Process different queries concurrently
        results = await asyncio.gather(
            sm1.receive_query("Query 1", {"user": "user1"}),
            sm2.receive_query("Query 2", {"user": "user2"}),
            sm3.receive_query("Query 3", {"user": "user3"})
        )
        
        # All should succeed
        assert all(results)
        
        # Each should have independent state
        assert sm1.context['query'] == "Query 1"
        assert sm2.context['query'] == "Query 2"
        assert sm3.context['query'] == "Query 3"
        
        # Each should be in QUERY_RECEIVED state
        assert sm1.is_in_state(ConversationStates.QUERY_RECEIVED)
        assert sm2.is_in_state(ConversationStates.QUERY_RECEIVED)
        assert sm3.is_in_state(ConversationStates.QUERY_RECEIVED)
    
    @pytest.mark.asyncio
    async def test_state_isolation(self):
        """Test state machines are properly isolated."""
        sm1 = ConversationStateMachine()
        sm2 = ConversationStateMachine()
        
        await sm1.start()
        await sm2.start()
        
        # Progress sm1 through several states
        await sm1.receive_query("Test 1", {})
        await sm1.recognize_intent({"type": "test"}, 0.9)
        await sm1.discover_tools(["tool1"])
        
        # sm2 should still be idle
        assert sm2.is_in_state(ConversationStates.IDLE)
        assert sm1.is_in_state(ConversationStates.TOOLS_DISCOVERED)
        
        # Error in sm1 shouldn't affect sm2
        await sm1.handle_error(Exception("Error in sm1"), {})
        assert sm1.is_in_state(ConversationStates.ERROR)
        assert sm2.is_in_state(ConversationStates.IDLE)