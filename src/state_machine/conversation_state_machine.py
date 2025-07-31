"""
Conversation State Machine for the Autonomous Tool Discovery System.

This module implements the main state machine that manages conversation flow,
from query reception through intent recognition, tool discovery, execution,
and feedback collection.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List

from src.state_machine.base import (
    StateMachine, State, StateType, StateHandler, TransitionGuard
)


# State constants
class ConversationStates:
    """Constants for all conversation states."""
    IDLE = "IDLE"
    QUERY_RECEIVED = "QUERY_RECEIVED"
    INTENT_RECOGNIZED = "INTENT_RECOGNIZED"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"
    CLARIFICATION_RECEIVED = "CLARIFICATION_RECEIVED"
    TOOLS_DISCOVERED = "TOOLS_DISCOVERED"
    NO_TOOLS_FOUND = "NO_TOOLS_FOUND"
    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_COMPLETE = "EXECUTION_COMPLETE"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    FEEDBACK_RECEIVED = "FEEDBACK_RECEIVED"
    ERROR = "ERROR"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    TIMEOUT = "TIMEOUT"
    USER_CANCELLED = "USER_CANCELLED"
    RETRY_REQUESTED = "RETRY_REQUESTED"


class ConversationStateMachine(StateMachine):
    """
    State machine for managing conversation flow in the tool discovery system.
    
    This state machine tracks the progression of a user query from initial
    reception through intent recognition, tool discovery, execution, and
    feedback collection.
    """
    
    def __init__(self):
        """Initialize the conversation state machine."""
        super().__init__(
            initial_state=ConversationStates.IDLE,
            name="ConversationStateMachine"
        )
        
        # Conversation-specific context
        self.context.update({
            'query': None,
            'intent': None,
            'discovered_tools': [],
            'selected_tools': [],
            'execution_results': [],
            'clarification_attempts': 0,
            'retry_count': 0,
            'start_time': None,
            'error_history': []
        })
        
        # Configuration
        self.max_clarification_attempts = 3
        self.max_retry_attempts = 3
        self.timeout_seconds = 300  # 5 minutes
    
    def _initialize_states(self) -> None:
        """Initialize all states and valid transitions."""
        
        # Define all states
        states = [
            # Normal flow states
            State(ConversationStates.IDLE, StateType.NORMAL, {
                ConversationStates.QUERY_RECEIVED,
                ConversationStates.ERROR
            }),
            
            State(ConversationStates.QUERY_RECEIVED, StateType.TRANSIENT, {
                ConversationStates.INTENT_RECOGNIZED,
                ConversationStates.CLARIFICATION_NEEDED,
                ConversationStates.ERROR,
                ConversationStates.TIMEOUT,
                ConversationStates.USER_CANCELLED,
                ConversationStates.IDLE  # Allow direct return to IDLE
            }),
            
            State(ConversationStates.INTENT_RECOGNIZED, StateType.TRANSIENT, {
                ConversationStates.TOOLS_DISCOVERED,
                ConversationStates.NO_TOOLS_FOUND,
                ConversationStates.ERROR,
                ConversationStates.USER_CANCELLED
            }),
            
            State(ConversationStates.CLARIFICATION_NEEDED, StateType.NORMAL, {
                ConversationStates.CLARIFICATION_RECEIVED,
                ConversationStates.TIMEOUT,
                ConversationStates.USER_CANCELLED,
                ConversationStates.ERROR
            }),
            
            State(ConversationStates.CLARIFICATION_RECEIVED, StateType.TRANSIENT, {
                ConversationStates.INTENT_RECOGNIZED,
                ConversationStates.CLARIFICATION_NEEDED,
                ConversationStates.ERROR
            }),
            
            State(ConversationStates.TOOLS_DISCOVERED, StateType.TRANSIENT, {
                ConversationStates.EXECUTION_STARTED,
                ConversationStates.USER_CANCELLED,
                ConversationStates.ERROR
            }),
            
            State(ConversationStates.NO_TOOLS_FOUND, StateType.NORMAL, {
                ConversationStates.IDLE,
                ConversationStates.RETRY_REQUESTED,
                ConversationStates.ERROR
            }),
            
            State(ConversationStates.EXECUTION_STARTED, StateType.NORMAL, {
                ConversationStates.EXECUTION_COMPLETE,
                ConversationStates.EXECUTION_FAILED,
                ConversationStates.TIMEOUT,
                ConversationStates.USER_CANCELLED,
                ConversationStates.ERROR
            }),
            
            State(ConversationStates.EXECUTION_COMPLETE, StateType.NORMAL, {
                ConversationStates.FEEDBACK_RECEIVED,
                ConversationStates.IDLE,
                ConversationStates.ERROR
            }),
            
            State(ConversationStates.EXECUTION_FAILED, StateType.ERROR, {
                ConversationStates.RETRY_REQUESTED,
                ConversationStates.IDLE,
                ConversationStates.ERROR_RECOVERY
            }),
            
            State(ConversationStates.FEEDBACK_RECEIVED, StateType.FINAL, {
                ConversationStates.IDLE
            }),
            
            # Error and special states
            State(ConversationStates.ERROR, StateType.ERROR, {
                ConversationStates.ERROR_RECOVERY,
                ConversationStates.IDLE
            }),
            
            State(ConversationStates.ERROR_RECOVERY, StateType.TRANSIENT, {
                ConversationStates.IDLE,
                ConversationStates.QUERY_RECEIVED,
                ConversationStates.ERROR
            }),
            
            State(ConversationStates.TIMEOUT, StateType.ERROR, {
                ConversationStates.IDLE,
                ConversationStates.RETRY_REQUESTED
            }),
            
            State(ConversationStates.USER_CANCELLED, StateType.FINAL, {
                ConversationStates.IDLE
            }),
            
            State(ConversationStates.RETRY_REQUESTED, StateType.TRANSIENT, {
                ConversationStates.QUERY_RECEIVED,
                ConversationStates.IDLE,
                ConversationStates.ERROR
            })
        ]
        
        # Add all states to the state machine
        for state in states:
            self.add_state(state)
        
        # Add transition guards
        self._add_transition_guards()
    
    def _add_transition_guards(self) -> None:
        """Add guard conditions for specific transitions."""
        
        # Guard for clarification attempts
        clarification_guard = TransitionGuard(
            lambda ctx: ctx.get('clarification_attempts', 0) < self.max_clarification_attempts,
            "Max clarification attempts not exceeded"
        )
        self.add_transition(
            ConversationStates.CLARIFICATION_RECEIVED,
            ConversationStates.CLARIFICATION_NEEDED,
            guard=clarification_guard
        )
        
        # Guard for retry attempts
        retry_guard = TransitionGuard(
            lambda ctx: ctx.get('retry_count', 0) < self.max_retry_attempts,
            "Max retry attempts not exceeded"
        )
        self.add_transition(
            ConversationStates.RETRY_REQUESTED,
            ConversationStates.QUERY_RECEIVED,
            guard=retry_guard
        )
        
        # Guard for timeout check
        timeout_guard = TransitionGuard(
            lambda ctx: self._check_timeout(ctx),
            "Operation not timed out"
        )
        # Apply to long-running states
        for state in [ConversationStates.QUERY_RECEIVED, 
                     ConversationStates.EXECUTION_STARTED]:
            self.add_transition(state, ConversationStates.TIMEOUT, guard=timeout_guard)
    
    def _check_timeout(self, context: Dict[str, Any]) -> bool:
        """Check if operation has timed out."""
        start_time = context.get('start_time')
        if not start_time:
            return True
        
        elapsed = (datetime.now() - start_time).total_seconds()
        return elapsed < self.timeout_seconds
    
    async def receive_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Receive a new user query and transition to QUERY_RECEIVED state.
        
        Args:
            query: The user's query
            user_context: Optional user context
            
        Returns:
            True if transition successful
        """
        if not self.is_in_state(ConversationStates.IDLE):
            self.logger.warning(f"Cannot receive query in state: {self.current_state.name}")
            return False
        
        # Validate query is not empty
        if not query or not query.strip():
            self.logger.warning("Cannot process empty query")
            return False
        
        # Update context
        self.context.update({
            'query': query,
            'query_timestamp': datetime.now(),
            'user_context': user_context or {},
            'start_time': datetime.now(),
            'clarification_attempts': 0,
            'retry_count': 0
        })
        
        return await self.transition(ConversationStates.QUERY_RECEIVED, trigger="query_received")
    
    async def recognize_intent(self, intent: Dict[str, Any], confidence: float) -> bool:
        """
        Process intent recognition result.
        
        Args:
            intent: Recognized intent information
            confidence: Confidence score
            
        Returns:
            True if transition successful
        """
        if not self.is_in_state(ConversationStates.QUERY_RECEIVED):
            self.logger.warning(f"Cannot recognize intent in state: {self.current_state.name}")
            return False
        
        self.context['intent'] = intent
        self.context['intent_confidence'] = confidence
        self.context['confidence'] = confidence
        
        # Decide next state based on confidence
        if confidence >= 0.7:  # High confidence
            return await self.transition(
                ConversationStates.INTENT_RECOGNIZED, 
                trigger="high_confidence_intent"
            )
        else:  # Low confidence, need clarification
            return await self.transition(
                ConversationStates.CLARIFICATION_NEEDED,
                trigger="low_confidence_intent"
            )
    
    async def discover_tools(self, tools: List[Dict[str, Any]]) -> bool:
        """
        Process tool discovery results.
        
        Args:
            tools: List of discovered tools
            
        Returns:
            True if transition successful
        """
        if not self.is_in_state(ConversationStates.INTENT_RECOGNIZED):
            self.logger.warning(f"Cannot discover tools in state: {self.current_state.name}")
            return False
        
        self.context['discovered_tools'] = tools
        
        if tools:
            return await self.transition(
                ConversationStates.TOOLS_DISCOVERED,
                trigger="tools_found"
            )
        else:
            return await self.transition(
                ConversationStates.NO_TOOLS_FOUND,
                trigger="no_tools_found"
            )
    
    async def start_execution(self, selected_tools: List[str]) -> bool:
        """
        Start tool execution.
        
        Args:
            selected_tools: List of tool IDs to execute
            
        Returns:
            True if transition successful
        """
        if not self.is_in_state(ConversationStates.TOOLS_DISCOVERED):
            self.logger.warning(f"Cannot start execution in state: {self.current_state.name}")
            return False
        
        self.context['selected_tools'] = selected_tools
        self.context['execution_start_time'] = datetime.now()
        
        return await self.transition(
            ConversationStates.EXECUTION_STARTED,
            trigger="execution_started"
        )
    
    async def complete_execution(self, results: List[Dict[str, Any]], success: bool) -> bool:
        """
        Complete tool execution.
        
        Args:
            results: Execution results
            success: Whether execution was successful
            
        Returns:
            True if transition successful
        """
        if not self.is_in_state(ConversationStates.EXECUTION_STARTED):
            self.logger.warning(f"Cannot complete execution in state: {self.current_state.name}")
            return False
        
        self.context['execution_results'] = results
        self.context['execution_success'] = success
        
        if success:
            return await self.transition(
                ConversationStates.EXECUTION_COMPLETE,
                trigger="execution_success"
            )
        else:
            return await self.transition(
                ConversationStates.EXECUTION_FAILED,
                trigger="execution_failure"
            )
    
    async def receive_feedback(self, feedback: Dict[str, Any]) -> bool:
        """
        Receive user feedback.
        
        Args:
            feedback: User feedback data
            
        Returns:
            True if transition successful
        """
        if not self.is_in_state(ConversationStates.EXECUTION_COMPLETE):
            self.logger.warning(f"Cannot receive feedback in state: {self.current_state.name}")
            return False
        
        self.context['user_feedback'] = feedback
        self.context['feedback'] = feedback
        
        return await self.transition(
            ConversationStates.FEEDBACK_RECEIVED,
            trigger="feedback_received"
        )
    
    async def handle_clarification(self, clarification: str) -> bool:
        """
        Handle user clarification.
        
        Args:
            clarification: User's clarification
            
        Returns:
            True if transition successful
        """
        if not self.is_in_state(ConversationStates.CLARIFICATION_NEEDED):
            self.logger.warning(f"Cannot handle clarification in state: {self.current_state.name}")
            return False
        
        self.context['clarification_attempts'] += 1
        self.context['last_clarification'] = clarification
        self.context['clarification'] = clarification
        
        # Add clarification to query
        original_query = self.context.get('query', '')
        self.context['query'] = f"{original_query} {clarification}"
        
        # Check if max attempts reached
        if self.context['clarification_attempts'] >= self.max_clarification_attempts:
            self.logger.warning("Max clarification attempts reached")
            return await self.transition(
                ConversationStates.ERROR,
                trigger="max_clarification_attempts"
            )
        
        return await self.transition(
            ConversationStates.CLARIFICATION_RECEIVED,
            trigger="clarification_provided"
        )
    
    async def request_retry(self) -> bool:
        """
        Request to retry the operation.
        
        Returns:
            True if transition successful
        """
        valid_states = [
            ConversationStates.NO_TOOLS_FOUND,
            ConversationStates.EXECUTION_FAILED,
            ConversationStates.TIMEOUT
        ]
        
        if not any(self.is_in_state(state) for state in valid_states):
            self.logger.warning(f"Cannot retry in state: {self.current_state.name}")
            return False
        
        # Check if max retries reached
        if self.context.get('retry_count', 0) >= self.max_retry_attempts:
            self.logger.warning("Max retry attempts reached")
            return False
        
        self.context['retry_count'] += 1
        
        return await self.transition(
            ConversationStates.RETRY_REQUESTED,
            trigger="retry_requested"
        )
    
    async def cancel_operation(self) -> bool:
        """
        Cancel the current operation.
        
        Returns:
            True if transition successful
        """
        # Allow cancellation from more states
        cancellable_states = [
            ConversationStates.QUERY_RECEIVED,
            ConversationStates.INTENT_RECOGNIZED,
            ConversationStates.CLARIFICATION_NEEDED,
            ConversationStates.TOOLS_DISCOVERED,
            ConversationStates.EXECUTION_STARTED
        ]
        
        if not any(self.is_in_state(state) for state in cancellable_states):
            self.logger.warning(f"Cannot cancel in state: {self.current_state.name}")
            return False
        
        self.context['cancelled_at'] = datetime.now()
        self.context['cancellation_timestamp'] = datetime.now()
        self.context['cancelled_from_state'] = self.current_state.name
        
        return await self.transition(
            ConversationStates.USER_CANCELLED,
            trigger="user_cancelled"
        )
    
    async def handle_error(self, error: Exception, error_context: Dict[str, Any]) -> bool:
        """
        Handle an error condition.
        
        Args:
            error: The exception that occurred
            error_context: Additional error context
            
        Returns:
            True if transition successful
        """
        # Record error in history
        self.context['error_history'].append({
            'error': str(error),
            'type': type(error).__name__,
            'context': error_context,
            'state': self.current_state.name if self.current_state else None,
            'timestamp': datetime.now()
        })
        self.context['error'] = str(error)
        self.context['error_context'] = error_context
        self.context['error_timestamp'] = datetime.now()
        
        return await self.transition(
            ConversationStates.ERROR,
            trigger="error_occurred"
        )
    
    async def attempt_recovery(self) -> bool:
        """
        Attempt to recover from error state.
        
        Returns:
            True if transition successful
        """
        if not self.is_in_state(ConversationStates.ERROR):
            self.logger.warning(f"Cannot attempt recovery in state: {self.current_state.name}")
            return False
        
        return await self.transition(
            ConversationStates.ERROR_RECOVERY,
            trigger="recovery_attempt"
        )
    
    async def handle_timeout(self) -> bool:
        """
        Handle operation timeout.
        
        Returns:
            True if transition successful
        """
        timeout_states = [
            ConversationStates.QUERY_RECEIVED,
            ConversationStates.CLARIFICATION_NEEDED,
            ConversationStates.EXECUTION_STARTED
        ]
        
        if not any(self.is_in_state(state) for state in timeout_states):
            self.logger.warning(f"Cannot handle timeout in state: {self.current_state.name}")
            return False
        
        self.context['timeout_timestamp'] = datetime.now()
        
        return await self.transition(
            ConversationStates.TIMEOUT,
            trigger="operation_timeout"
        )
    
    async def transition_to(self, target_state: str) -> bool:
        """
        Alias for transition method to support legacy tests.
        
        Args:
            target_state: Target state name
            
        Returns:
            True if transition successful
        """
        return await self.transition(target_state)
    
    async def return_to_idle(self) -> bool:
        """
        Return to idle state from various end states.
        
        Returns:
            True if transition successful
        """
        valid_states = [
            ConversationStates.FEEDBACK_RECEIVED,
            ConversationStates.NO_TOOLS_FOUND,
            ConversationStates.USER_CANCELLED,
            ConversationStates.ERROR,
            ConversationStates.ERROR_RECOVERY,
            ConversationStates.TIMEOUT,
            ConversationStates.EXECUTION_COMPLETE,
            ConversationStates.EXECUTION_FAILED,
            ConversationStates.QUERY_RECEIVED,  # Allow returning to IDLE after query processing
            ConversationStates.INTENT_RECOGNIZED  # Also allow from intent recognized state
        ]
        
        if not any(self.is_in_state(state) for state in valid_states):
            self.logger.warning(f"Cannot return to idle from state: {self.current_state.name}")
            return False
        
        # Clear transient context data
        self.context.update({
            'query': None,
            'intent': None,  
            'discovered_tools': [],
            'selected_tools': [],
            'execution_results': [],
            'clarification_attempts': 0,
            'start_time': None
        })
        
        return await self.transition(
            ConversationStates.IDLE,
            trigger="reset_to_idle"
        )
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state."""
        return {
            'current_state': self.current_state.name if self.current_state else None,
            'query': self.context.get('query'),
            'intent': self.context.get('intent'),
            'discovered_tools_count': len(self.context.get('discovered_tools', [])),
            'selected_tools': self.context.get('selected_tools', []),
            'execution_success': self.context.get('execution_success'),
            'execution_results': self.context.get('execution_results', []),
            'clarification_attempts': self.context.get('clarification_attempts', 0),
            'retry_count': self.context.get('retry_count', 0),
            'error_count': len(self.context.get('error_history', [])),
            'duration': self._get_conversation_duration(),
            'state_transitions': len(self.get_state_history())
        }
    
    def _get_conversation_duration(self) -> Optional[float]:
        """Get the duration of the current conversation in seconds."""
        start_time = self.context.get('start_time')
        if not start_time:
            return None
        
        return (datetime.now() - start_time).total_seconds()