"""
Unit tests for conversation state machine.

Tests the conversation-specific state machine implementation including
all conversation states, transitions, and business logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.state_machine.conversation_state_machine import (
    ConversationStates, ConversationStateMachine
)
from src.state_machine.base import StateType


class TestConversationStates:
    """Test the ConversationStates constants."""
    
    def test_all_states_defined(self):
        """Test that all expected conversation states are defined."""
        expected_states = [
            'IDLE', 'QUERY_RECEIVED', 'INTENT_RECOGNIZED', 
            'CLARIFICATION_NEEDED', 'CLARIFICATION_RECEIVED',
            'TOOLS_DISCOVERED', 'NO_TOOLS_FOUND', 'EXECUTION_STARTED',
            'EXECUTION_COMPLETE', 'EXECUTION_FAILED', 'FEEDBACK_RECEIVED',
            'ERROR', 'ERROR_RECOVERY', 'TIMEOUT', 'RETRY_REQUESTED',
            'USER_CANCELLED'
        ]
        
        for state in expected_states:
            assert hasattr(ConversationStates, state)
            assert isinstance(getattr(ConversationStates, state), str)


class TestConversationStateMachine:
    """Test the ConversationStateMachine class."""
    
    @pytest.fixture
    async def state_machine(self):
        """Create and start a conversation state machine."""
        sm = ConversationStateMachine()
        await sm.start()
        yield sm
        # Cleanup if needed
    
    @pytest.mark.asyncio
    async def test_initialization(self, state_machine):
        """Test state machine initialization."""
        sm = state_machine
        
        # Should start in IDLE state
        assert sm.is_in_state(ConversationStates.IDLE)
        
        # Check all states are registered
        assert len(sm.states) >= 16  # At least 16 conversation states
        
        # Check key states exist
        assert ConversationStates.IDLE in sm.states
        assert ConversationStates.QUERY_RECEIVED in sm.states
        assert ConversationStates.INTENT_RECOGNIZED in sm.states
    
    @pytest.mark.asyncio
    async def test_receive_query(self, state_machine):
        """Test receiving a user query."""
        sm = state_machine
        
        # Receive query
        result = await sm.receive_query("Find Python files", {"user_id": "123"})
        assert result is True
        
        # Should transition to QUERY_RECEIVED
        assert sm.is_in_state(ConversationStates.QUERY_RECEIVED)
        
        # Check context
        assert sm.context["query"] == "Find Python files"
        assert sm.context["user_context"]["user_id"] == "123"
        assert "query_timestamp" in sm.context
    
    @pytest.mark.asyncio
    async def test_receive_query_empty(self, state_machine):
        """Test receiving empty query."""
        sm = state_machine
        
        # Empty query should fail
        result = await sm.receive_query("", {})
        assert result is False
        assert sm.is_in_state(ConversationStates.IDLE)
    
    @pytest.mark.asyncio
    async def test_recognize_intent_success(self, state_machine):
        """Test successful intent recognition."""
        sm = state_machine
        
        # First receive query
        await sm.receive_query("Find Python files", {})
        
        # Recognize intent with high confidence
        intent = {
            "type": "query.search",
            "keywords": ["find", "python", "files"],
            "confidence": 0.85
        }
        result = await sm.recognize_intent(intent, 0.85)
        assert result is True
        
        # Should transition to INTENT_RECOGNIZED
        assert sm.is_in_state(ConversationStates.INTENT_RECOGNIZED)
        
        # Check context
        assert sm.context["intent"] == intent
        assert sm.context["confidence"] == 0.85
    
    @pytest.mark.asyncio
    async def test_recognize_intent_low_confidence(self, state_machine):
        """Test intent recognition with low confidence."""
        sm = state_machine
        
        # First receive query
        await sm.receive_query("Ambiguous query", {})
        
        # Recognize intent with low confidence
        intent = {
            "type": "unknown",
            "keywords": [],
            "confidence": 0.3
        }
        result = await sm.recognize_intent(intent, 0.3)
        assert result is True
        
        # Should transition to CLARIFICATION_NEEDED
        assert sm.is_in_state(ConversationStates.CLARIFICATION_NEEDED)
        assert sm.context["clarification_attempts"] == 0
    
    @pytest.mark.asyncio
    async def test_handle_clarification(self, state_machine):
        """Test handling user clarification."""
        sm = state_machine
        
        # Setup: get to clarification needed state
        await sm.receive_query("Ambiguous", {})
        await sm.recognize_intent({"type": "unknown"}, 0.3)
        assert sm.is_in_state(ConversationStates.CLARIFICATION_NEEDED)
        
        # Handle clarification
        result = await sm.handle_clarification("I meant search for Python files")
        assert result is True
        
        # Should transition to CLARIFICATION_RECEIVED
        assert sm.is_in_state(ConversationStates.CLARIFICATION_RECEIVED)
        assert sm.context["clarification"] == "I meant search for Python files"
        assert sm.context["clarification_attempts"] == 1
    
    @pytest.mark.asyncio
    async def test_clarification_max_attempts(self, state_machine):
        """Test maximum clarification attempts."""
        sm = state_machine
        sm.max_clarification_attempts = 2
        
        # Setup: get to clarification needed state
        await sm.receive_query("Ambiguous", {})
        await sm.recognize_intent({"type": "unknown"}, 0.3)
        
        # First clarification
        await sm.handle_clarification("Still unclear")
        sm.current_state = sm.states[ConversationStates.CLARIFICATION_NEEDED]
        
        # Second clarification  
        await sm.handle_clarification("Still unclear again")
        sm.current_state = sm.states[ConversationStates.CLARIFICATION_NEEDED]
        
        # Should hit max attempts on third try
        sm.context["clarification_attempts"] = 2
        result = await sm.handle_clarification("Third time")
        
        # Should transition to ERROR state
        assert sm.is_in_state(ConversationStates.ERROR)
    
    @pytest.mark.asyncio
    async def test_discover_tools_success(self, state_machine):
        """Test successful tool discovery."""
        sm = state_machine
        
        # Setup: get to intent recognized state
        await sm.receive_query("Find files", {})
        await sm.recognize_intent({"type": "query.search"}, 0.9)
        
        # Discover tools
        tools = ["filesystem_mcp", "search_mcp"]
        result = await sm.discover_tools(tools)
        assert result is True
        
        # Should transition to TOOLS_DISCOVERED
        assert sm.is_in_state(ConversationStates.TOOLS_DISCOVERED)
        assert sm.context["discovered_tools"] == tools
    
    @pytest.mark.asyncio
    async def test_discover_tools_none_found(self, state_machine):
        """Test tool discovery with no tools found."""
        sm = state_machine
        
        # Setup: get to intent recognized state
        await sm.receive_query("Do something impossible", {})
        await sm.recognize_intent({"type": "action.unknown"}, 0.8)
        
        # Discover no tools
        result = await sm.discover_tools([])
        assert result is True
        
        # Should transition to NO_TOOLS_FOUND
        assert sm.is_in_state(ConversationStates.NO_TOOLS_FOUND)
    
    @pytest.mark.asyncio
    async def test_start_execution(self, state_machine):
        """Test starting tool execution."""
        sm = state_machine
        
        # Setup: get to tools discovered state
        await sm.receive_query("Find files", {})
        await sm.recognize_intent({"type": "query.search"}, 0.9)
        await sm.discover_tools(["filesystem_mcp", "search_mcp"])
        
        # Start execution
        selected_tools = ["filesystem_mcp"]
        result = await sm.start_execution(selected_tools)
        assert result is True
        
        # Should transition to EXECUTION_STARTED
        assert sm.is_in_state(ConversationStates.EXECUTION_STARTED)
        assert sm.context["selected_tools"] == selected_tools
        assert "execution_start_time" in sm.context
    
    @pytest.mark.asyncio
    async def test_complete_execution_success(self, state_machine):
        """Test successful execution completion."""
        sm = state_machine
        
        # Setup: get to execution started state
        await sm.receive_query("Find files", {})
        await sm.recognize_intent({"type": "query.search"}, 0.9)
        await sm.discover_tools(["filesystem_mcp"])
        await sm.start_execution(["filesystem_mcp"])
        
        # Complete execution
        results = [{"tool": "filesystem_mcp", "result": ["file1.py", "file2.py"]}]
        result = await sm.complete_execution(results, success=True)
        assert result is True
        
        # Should transition to EXECUTION_COMPLETE
        assert sm.is_in_state(ConversationStates.EXECUTION_COMPLETE)
        assert sm.context["execution_results"] == results
        assert sm.context["execution_success"] is True
    
    @pytest.mark.asyncio
    async def test_complete_execution_failure(self, state_machine):
        """Test failed execution completion."""
        sm = state_machine
        
        # Setup: get to execution started state
        await sm.receive_query("Do something", {})
        await sm.recognize_intent({"type": "action.create"}, 0.9)
        await sm.discover_tools(["create_mcp"])
        await sm.start_execution(["create_mcp"])
        
        # Complete with failure
        results = [{"tool": "create_mcp", "error": "Permission denied"}]
        result = await sm.complete_execution(results, success=False)
        assert result is True
        
        # Should transition to EXECUTION_FAILED
        assert sm.is_in_state(ConversationStates.EXECUTION_FAILED)
        assert sm.context["execution_success"] is False
    
    @pytest.mark.asyncio
    async def test_receive_feedback(self, state_machine):
        """Test receiving user feedback."""
        sm = state_machine
        
        # Setup: complete execution
        await sm.receive_query("Find files", {})
        await sm.recognize_intent({"type": "query.search"}, 0.9)
        await sm.discover_tools(["filesystem_mcp"])
        await sm.start_execution(["filesystem_mcp"])
        await sm.complete_execution([{"result": "success"}], success=True)
        
        # Receive feedback
        feedback = {"rating": 5, "comment": "Great results!"}
        result = await sm.receive_feedback(feedback)
        assert result is True
        
        # Should transition to FEEDBACK_RECEIVED
        assert sm.is_in_state(ConversationStates.FEEDBACK_RECEIVED)
        assert sm.context["feedback"] == feedback
    
    @pytest.mark.asyncio
    async def test_handle_error(self, state_machine):
        """Test error handling."""
        sm = state_machine
        
        # Can handle error from any state
        await sm.receive_query("Test query", {})
        
        # Handle error
        error = RuntimeError("Something went wrong")
        error_context = {"stage": "processing", "details": "test"}
        result = await sm.handle_error(error, error_context)
        assert result is True
        
        # Should transition to ERROR state
        assert sm.is_in_state(ConversationStates.ERROR)
        assert sm.context["error"] == str(error)
        assert sm.context["error_context"] == error_context
        assert sm.context["error_timestamp"] is not None
    
    @pytest.mark.asyncio
    async def test_request_retry(self, state_machine):
        """Test retry request."""
        sm = state_machine
        
        # Setup: get to execution failed state
        await sm.receive_query("Do something", {})
        await sm.recognize_intent({"type": "action.create"}, 0.9)
        await sm.discover_tools(["create_mcp"])
        await sm.start_execution(["create_mcp"])
        await sm.complete_execution([], success=False)
        
        # Request retry
        result = await sm.request_retry()
        assert result is True
        
        # Should transition to RETRY_REQUESTED
        assert sm.is_in_state(ConversationStates.RETRY_REQUESTED)
        assert sm.context["retry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_max_retry_attempts(self, state_machine):
        """Test maximum retry attempts."""
        sm = state_machine
        sm.max_retry_attempts = 2
        
        # Setup: get to execution failed state
        await sm.receive_query("Do something", {})
        await sm.recognize_intent({"type": "action.create"}, 0.9)
        await sm.discover_tools(["create_mcp"])
        await sm.start_execution(["create_mcp"])
        await sm.complete_execution([], success=False)
        
        # Set retry count to max
        sm.context["retry_count"] = 2
        
        # Request retry should fail
        result = await sm.request_retry()
        assert result is False
        assert sm.is_in_state(ConversationStates.EXECUTION_FAILED)
    
    @pytest.mark.asyncio
    async def test_handle_timeout(self, state_machine):
        """Test timeout handling."""
        sm = state_machine
        
        # Start some operation
        await sm.receive_query("Long operation", {})
        
        # Handle timeout
        result = await sm.handle_timeout()
        assert result is True
        
        # Should transition to TIMEOUT state
        assert sm.is_in_state(ConversationStates.TIMEOUT)
        assert sm.context["timeout_timestamp"] is not None
    
    @pytest.mark.asyncio
    async def test_cancel_operation(self, state_machine):
        """Test operation cancellation."""
        sm = state_machine
        
        # Start execution
        await sm.receive_query("Cancel this", {})
        await sm.recognize_intent({"type": "query.search"}, 0.9)
        await sm.discover_tools(["search_mcp"])
        await sm.start_execution(["search_mcp"])
        
        # Cancel operation
        result = await sm.cancel_operation()
        assert result is True
        
        # Should transition to USER_CANCELLED
        assert sm.is_in_state(ConversationStates.USER_CANCELLED)
        assert sm.context["cancellation_timestamp"] is not None
    
    @pytest.mark.asyncio
    async def test_return_to_idle(self, state_machine):
        """Test returning to idle state."""
        sm = state_machine
        
        # Complete a full flow
        await sm.receive_query("Test", {})
        await sm.recognize_intent({"type": "query.search"}, 0.9)
        await sm.discover_tools(["test_mcp"])
        await sm.start_execution(["test_mcp"])
        await sm.complete_execution([{"result": "done"}], success=True)
        
        # Return to idle
        result = await sm.return_to_idle()
        assert result is True
        
        # Should be in IDLE state
        assert sm.is_in_state(ConversationStates.IDLE)
    
    @pytest.mark.asyncio
    async def test_get_conversation_summary(self, state_machine):
        """Test getting conversation summary."""
        sm = state_machine
        
        # Execute a flow
        await sm.receive_query("Find Python files", {"user_id": "123"})
        await sm.recognize_intent({"type": "query.search", "keywords": ["python", "files"]}, 0.9)
        await sm.discover_tools(["filesystem_mcp", "search_mcp"])
        await sm.start_execution(["filesystem_mcp"])
        await sm.complete_execution([{"tool": "filesystem_mcp", "result": "5 files found"}], success=True)
        
        # Get summary
        summary = sm.get_conversation_summary()
        
        assert summary["current_state"] == ConversationStates.EXECUTION_COMPLETE
        assert summary["query"] == "Find Python files"
        assert summary["intent"]["type"] == "query.search"
        assert summary["selected_tools"] == ["filesystem_mcp"]
        assert summary["execution_success"] is True
        assert summary["state_transitions"] >= 5
    
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, state_machine):
        """Test error recovery flow."""
        sm = state_machine
        
        # Cause an error
        await sm.receive_query("Test", {})
        await sm.handle_error(Exception("Test error"), {})
        assert sm.is_in_state(ConversationStates.ERROR)
        
        # Attempt recovery
        result = await sm.transition_to(ConversationStates.ERROR_RECOVERY)
        assert result is True
        assert sm.is_in_state(ConversationStates.ERROR_RECOVERY)
        
        # Should be able to return to idle
        result = await sm.return_to_idle()
        assert result is True
        assert sm.is_in_state(ConversationStates.IDLE)
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, state_machine):
        """Test that context persists across transitions."""
        sm = state_machine
        
        # Add custom context
        sm.context["custom_data"] = {"key": "value"}
        
        # Go through several transitions
        await sm.receive_query("Test persistence", {})
        assert sm.context["custom_data"]["key"] == "value"
        
        await sm.recognize_intent({"type": "test"}, 0.9)
        assert sm.context["custom_data"]["key"] == "value"
        
        await sm.discover_tools(["test_tool"])
        assert sm.context["custom_data"]["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_invalid_state_transitions(self, state_machine):
        """Test that invalid transitions are prevented."""
        sm = state_machine
        
        # Try to go directly from IDLE to EXECUTION_STARTED
        result = await sm.transition_to(ConversationStates.EXECUTION_STARTED)
        assert result is False
        assert sm.is_in_state(ConversationStates.IDLE)
        
        # Try to go from QUERY_RECEIVED to FEEDBACK_RECEIVED
        await sm.receive_query("Test", {})
        result = await sm.transition_to(ConversationStates.FEEDBACK_RECEIVED)
        assert result is False
        assert sm.is_in_state(ConversationStates.QUERY_RECEIVED)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, state_machine):
        """Test handling of concurrent operations."""
        sm = state_machine
        
        # Start multiple concurrent operations
        tasks = [
            sm.receive_query("Query 1", {}),
            sm.receive_query("Query 2", {}),
            sm.receive_query("Query 3", {})
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Only the first should succeed
        assert results[0] is True
        assert results[1] is False or isinstance(results[1], Exception)
        assert results[2] is False or isinstance(results[2], Exception)
        
        # Should be in QUERY_RECEIVED state with first query
        assert sm.is_in_state(ConversationStates.QUERY_RECEIVED)
        assert sm.context["query"] == "Query 1"
    
    @pytest.mark.asyncio
    async def test_operation_timing(self, state_machine):
        """Test operation timing tracking."""
        sm = state_machine
        
        # Execute timed operations
        start_time = datetime.now()
        
        await sm.receive_query("Test timing", {})
        query_time = sm.context["query_timestamp"]
        
        await asyncio.sleep(0.1)  # Simulate processing time
        
        await sm.recognize_intent({"type": "test"}, 0.9)
        await sm.discover_tools(["test_tool"])
        await sm.start_execution(["test_tool"])
        
        exec_start_time = sm.context["execution_start_time"]
        
        await asyncio.sleep(0.1)  # Simulate execution time
        
        await sm.complete_execution([{"result": "done"}], success=True)
        
        # Verify timing
        assert isinstance(query_time, datetime)
        assert isinstance(exec_start_time, datetime)
        assert exec_start_time > query_time
        assert (datetime.now() - start_time).total_seconds() > 0.2