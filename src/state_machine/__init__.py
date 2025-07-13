"""
State Machine infrastructure for the Autonomous Tool Discovery System.

This package provides state machine functionality for managing conversation
flow, state transitions, and error handling.
"""

from .base import (
    State,
    StateType,
    Transition,
    StateHandler,
    TransitionGuard,
    StateMachine
)

from .conversation_state_machine import (
    ConversationStates,
    ConversationStateMachine
)

from .handlers import (
    BaseConversationHandler,
    QueryReceivedHandler,
    IntentRecognitionHandler,
    ToolDiscoveryHandler,
    ExecutionHandler,
    ErrorHandler,
    FeedbackHandler,
    ClarificationHandler,
    RetryHandler,
    StateTransitionLogger,
    HANDLER_REGISTRY,
    register_handlers
)

__all__ = [
    # Base classes
    'State',
    'StateType',
    'Transition',
    'StateHandler',
    'TransitionGuard',
    'StateMachine',
    
    # Conversation state machine
    'ConversationStates',
    'ConversationStateMachine',
    
    # Handlers
    'BaseConversationHandler',
    'QueryReceivedHandler',
    'IntentRecognitionHandler',
    'ToolDiscoveryHandler',
    'ExecutionHandler',
    'ErrorHandler',
    'FeedbackHandler',
    'ClarificationHandler',
    'RetryHandler',
    'StateTransitionLogger',
    'HANDLER_REGISTRY',
    'register_handlers'
]