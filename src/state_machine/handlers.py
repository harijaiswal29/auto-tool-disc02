"""
State transition handlers for the Conversation State Machine.

This module provides specific handlers for different state transitions,
managing context updates, logging, and triggering appropriate actions.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from src.state_machine.base import StateHandler, State, Transition
from src.state_machine.conversation_state_machine import ConversationStates
from src.utils.logger import get_logger


class BaseConversationHandler(StateHandler):
    """Base handler with common functionality for conversation states."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Log entry to state."""
        self.logger.info(f"Entering state: {state.name}")
        context['last_state_entry'] = datetime.now()
    
    async def on_exit(self, state: State, context: Dict[str, Any]) -> None:
        """Log exit from state."""
        self.logger.info(f"Exiting state: {state.name}")
        
        # Calculate time spent in state
        entry_time = context.get('last_state_entry')
        if entry_time:
            duration = (datetime.now() - entry_time).total_seconds()
            self.logger.debug(f"Time in state {state.name}: {duration:.2f}s")
    
    async def on_transition(self, transition: Transition) -> None:
        """Log state transition."""
        self.logger.info(
            f"Transition: {transition.from_state} → {transition.to_state}"
            f"{f' [trigger: {transition.trigger}]' if transition.trigger else ''}"
        )


class QueryReceivedHandler(BaseConversationHandler):
    """Handler for QUERY_RECEIVED state."""
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Initialize query processing."""
        await super().on_enter(state, context)
        
        query = context.get('query', '')
        self.logger.info(f"Processing query: {query[:100]}...")
        
        # Initialize processing metrics
        context['processing_start_time'] = datetime.now()
        context['processing_steps'] = []
        
        # Mark query as being processed
        context['query_status'] = 'processing'


class IntentRecognitionHandler(BaseConversationHandler):
    """Handler for intent recognition states."""
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Handle intent recognition entry."""
        await super().on_enter(state, context)
        
        if state.name == ConversationStates.INTENT_RECOGNIZED:
            intent = context.get('intent', {})
            confidence = context.get('intent_confidence', 0.0)
            
            self.logger.info(
                f"Intent recognized: {intent.get('type', 'unknown')} "
                f"(confidence: {confidence:.2f})"
            )
            
            # Record intent recognition time
            context['intent_recognition_time'] = datetime.now()
            
        elif state.name == ConversationStates.CLARIFICATION_NEEDED:
            self.logger.info("Low confidence intent - clarification needed")
            context['clarification_request_time'] = datetime.now()


class ToolDiscoveryHandler(BaseConversationHandler):
    """Handler for tool discovery states."""
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Handle tool discovery state entry."""
        await super().on_enter(state, context)
        
        if state.name == ConversationStates.TOOLS_DISCOVERED:
            tools = context.get('discovered_tools', [])
            self.logger.info(f"Discovered {len(tools)} tools")
            
            # Log tool names
            for tool in tools[:5]:  # Log first 5 tools
                self.logger.debug(f"  - {tool.get('name', 'unknown')}")
            
            if len(tools) > 5:
                self.logger.debug(f"  ... and {len(tools) - 5} more")
            
        elif state.name == ConversationStates.NO_TOOLS_FOUND:
            self.logger.warning("No tools found for the given intent")
            context['no_tools_reason'] = 'No matching tools in registry'


class ExecutionHandler(BaseConversationHandler):
    """Handler for execution states."""
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Handle execution state entry."""
        await super().on_enter(state, context)
        
        if state.name == ConversationStates.EXECUTION_STARTED:
            selected_tools = context.get('selected_tools', [])
            self.logger.info(f"Starting execution of {len(selected_tools)} tools")
            
            # Initialize execution tracking
            context['execution_progress'] = {
                'total': len(selected_tools),
                'completed': 0,
                'failed': 0,
                'in_progress': len(selected_tools)
            }
            
        elif state.name == ConversationStates.EXECUTION_COMPLETE:
            results = context.get('execution_results', [])
            self.logger.info(f"Execution completed with {len(results)} results")
            
            # Calculate execution statistics
            successful = sum(1 for r in results if r.get('success', False))
            failed = len(results) - successful
            
            context['execution_stats'] = {
                'total': len(results),
                'successful': successful,
                'failed': failed,
                'duration': self._calculate_execution_duration(context)
            }
            
        elif state.name == ConversationStates.EXECUTION_FAILED:
            self.logger.error("Execution failed")
            
            # Record failure details
            context['failure_details'] = {
                'timestamp': datetime.now(),
                'failed_tools': [
                    r.get('tool') for r in context.get('execution_results', [])
                    if not r.get('success', False)
                ]
            }
    
    def _calculate_execution_duration(self, context: Dict[str, Any]) -> float:
        """Calculate execution duration in seconds."""
        start_time = context.get('execution_start_time')
        if not start_time:
            return 0.0
        return (datetime.now() - start_time).total_seconds()


class ErrorHandler(BaseConversationHandler):
    """Handler for error states."""
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Handle error state entry."""
        await super().on_enter(state, context)
        
        if state.name == ConversationStates.ERROR:
            error_history = context.get('error_history', [])
            if error_history:
                last_error = error_history[-1]
                self.logger.error(
                    f"Error occurred: {last_error.get('type', 'Unknown')} - "
                    f"{last_error.get('error', 'No details')}"
                )
            
            # Determine if recovery is possible
            context['recovery_possible'] = self._can_recover(context)
            
        elif state.name == ConversationStates.ERROR_RECOVERY:
            self.logger.info("Attempting error recovery")
            
            # Initialize recovery attempts
            context['recovery_attempts'] = context.get('recovery_attempts', 0) + 1
            
        elif state.name == ConversationStates.TIMEOUT:
            self.logger.warning("Operation timed out")
            
            # Record timeout details
            context['timeout_details'] = {
                'timestamp': datetime.now(),
                'duration': self._calculate_operation_duration(context),
                'last_state': context.get('last_state_entry')
            }
    
    def _can_recover(self, context: Dict[str, Any]) -> bool:
        """Determine if recovery from error is possible."""
        # Check recovery attempts
        recovery_attempts = context.get('recovery_attempts', 0)
        if recovery_attempts >= 3:
            return False
        
        # Check error type
        error_history = context.get('error_history', [])
        if error_history:
            last_error_type = error_history[-1].get('type', '')
            # Some errors are not recoverable
            if last_error_type in ['AuthenticationError', 'PermissionError']:
                return False
        
        return True
    
    def _calculate_operation_duration(self, context: Dict[str, Any]) -> float:
        """Calculate total operation duration."""
        start_time = context.get('start_time')
        if not start_time:
            return 0.0
        return (datetime.now() - start_time).total_seconds()


class FeedbackHandler(BaseConversationHandler):
    """Handler for feedback states."""
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Handle feedback state entry."""
        await super().on_enter(state, context)
        
        if state.name == ConversationStates.FEEDBACK_RECEIVED:
            feedback = context.get('user_feedback', {})
            self.logger.info(
                f"Received feedback: {feedback.get('type', 'unknown')} - "
                f"Rating: {feedback.get('rating', 'N/A')}"
            )
            
            # Process feedback for learning
            context['feedback_processed'] = {
                'timestamp': datetime.now(),
                'type': feedback.get('type'),
                'rating': feedback.get('rating'),
                'comments': feedback.get('comments', '')
            }


class ClarificationHandler(BaseConversationHandler):
    """Handler for clarification states."""
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Handle clarification state entry."""
        await super().on_enter(state, context)
        
        if state.name == ConversationStates.CLARIFICATION_RECEIVED:
            clarification = context.get('last_clarification', '')
            attempts = context.get('clarification_attempts', 0)
            
            self.logger.info(
                f"Received clarification (attempt {attempts}): {clarification[:100]}..."
            )
            
            # Update query with clarification
            context['clarified_query'] = context.get('query', '') + ' ' + clarification


class RetryHandler(BaseConversationHandler):
    """Handler for retry states."""
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Handle retry state entry."""
        await super().on_enter(state, context)
        
        if state.name == ConversationStates.RETRY_REQUESTED:
            retry_count = context.get('retry_count', 0)
            self.logger.info(f"Retry requested (attempt {retry_count})")
            
            # Reset relevant context for retry
            context['retry_timestamp'] = datetime.now()
            
            # Clear previous results
            context['execution_results'] = []
            context['discovered_tools'] = []
            
            # Keep error history for debugging
            context['previous_errors'] = context.get('error_history', []).copy()


class StateTransitionLogger(StateHandler):
    """Generic handler that logs all state transitions."""
    
    def __init__(self):
        self.logger = get_logger("StateTransitionLogger")
        self.transition_log = []
    
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Log state entry."""
        entry = {
            'timestamp': datetime.now(),
            'event': 'enter',
            'state': state.name,
            'state_type': state.type.value,
            'context_keys': list(context.keys())
        }
        self.transition_log.append(entry)
        self.logger.debug(f"State entry logged: {state.name}")
    
    async def on_exit(self, state: State, context: Dict[str, Any]) -> None:
        """Log state exit."""
        entry = {
            'timestamp': datetime.now(),
            'event': 'exit',
            'state': state.name,
            'state_type': state.type.value
        }
        self.transition_log.append(entry)
        self.logger.debug(f"State exit logged: {state.name}")
    
    async def on_transition(self, transition: Transition) -> None:
        """Log state transition."""
        entry = {
            'timestamp': transition.timestamp,
            'event': 'transition',
            'from_state': transition.from_state,
            'to_state': transition.to_state,
            'trigger': transition.trigger
        }
        self.transition_log.append(entry)
        self.logger.debug(
            f"Transition logged: {transition.from_state} → {transition.to_state}"
        )
    
    def get_transition_log(self) -> list:
        """Get the complete transition log."""
        return self.transition_log.copy()


# Handler registry for easy access
HANDLER_REGISTRY = {
    ConversationStates.QUERY_RECEIVED: QueryReceivedHandler(),
    ConversationStates.INTENT_RECOGNIZED: IntentRecognitionHandler(),
    ConversationStates.CLARIFICATION_NEEDED: IntentRecognitionHandler(),
    ConversationStates.CLARIFICATION_RECEIVED: ClarificationHandler(),
    ConversationStates.TOOLS_DISCOVERED: ToolDiscoveryHandler(),
    ConversationStates.NO_TOOLS_FOUND: ToolDiscoveryHandler(),
    ConversationStates.EXECUTION_STARTED: ExecutionHandler(),
    ConversationStates.EXECUTION_COMPLETE: ExecutionHandler(),
    ConversationStates.EXECUTION_FAILED: ExecutionHandler(),
    ConversationStates.FEEDBACK_RECEIVED: FeedbackHandler(),
    ConversationStates.ERROR: ErrorHandler(),
    ConversationStates.ERROR_RECOVERY: ErrorHandler(),
    ConversationStates.TIMEOUT: ErrorHandler(),
    ConversationStates.RETRY_REQUESTED: RetryHandler()
}


def register_handlers(state_machine) -> None:
    """
    Register all handlers with a conversation state machine.
    
    Args:
        state_machine: The ConversationStateMachine instance
    """
    # Register specific handlers
    for state_name, handler in HANDLER_REGISTRY.items():
        state_machine.set_state_handler(state_name, handler)
    
    # Add a general transition logger
    transition_logger = StateTransitionLogger()
    
    # Register the logger for all transitions
    for state in state_machine.states.values():
        for target_state in state.allowed_transitions:
            state_machine.add_transition(
                state.name,
                target_state,
                handler=transition_logger
            )