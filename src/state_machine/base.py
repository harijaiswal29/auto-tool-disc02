"""
Base state machine infrastructure for the Autonomous Tool Discovery System.

This module provides the foundation for building state machines that manage
conversation flow, state transitions, and error handling.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
import logging

from src.utils.logger import get_logger


class StateType(Enum):
    """Types of states in the state machine."""
    NORMAL = "normal"
    ERROR = "error"
    FINAL = "final"
    TRANSIENT = "transient"


@dataclass
class State:
    """
    Represents a state in the state machine.
    
    Attributes:
        name: Unique name of the state
        type: Type of state (normal, error, final, transient)
        allowed_transitions: Set of states this state can transition to
        metadata: Additional state-specific metadata
    """
    name: str
    type: StateType = StateType.NORMAL
    allowed_transitions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_transition_to(self, target_state: str) -> bool:
        """Check if transition to target state is allowed."""
        return target_state in self.allowed_transitions
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.name == other.name
        return self.name == other


@dataclass
class Transition:
    """
    Represents a transition between states.
    
    Attributes:
        from_state: Source state name
        to_state: Target state name
        timestamp: When the transition occurred
        context: Context data at time of transition
        trigger: What triggered the transition
    """
    from_state: str
    to_state: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    trigger: Optional[str] = None


class StateHandler(ABC):
    """
    Abstract base class for state transition handlers.
    
    Handlers are executed during state transitions to perform
    necessary actions and updates.
    """
    
    @abstractmethod
    async def on_enter(self, state: State, context: Dict[str, Any]) -> None:
        """Called when entering a state."""
        pass
    
    @abstractmethod
    async def on_exit(self, state: State, context: Dict[str, Any]) -> None:
        """Called when exiting a state."""
        pass
    
    @abstractmethod
    async def on_transition(self, transition: Transition) -> None:
        """Called during a state transition."""
        pass


class TransitionGuard:
    """
    Guard conditions that must be satisfied for a transition to occur.
    """
    
    def __init__(self, condition: Callable[[Dict[str, Any]], bool], 
                 description: str = ""):
        """
        Initialize a transition guard.
        
        Args:
            condition: Function that takes context and returns True if transition allowed
            description: Human-readable description of the guard condition
        """
        self.condition = condition
        self.description = description
    
    def check(self, context: Dict[str, Any]) -> bool:
        """Check if the guard condition is satisfied."""
        return self.condition(context)


class StateMachine(ABC):
    """
    Abstract base class for state machines.
    
    Provides core functionality for state management, transitions,
    and history tracking.
    """
    
    def __init__(self, initial_state: str, name: str = "StateMachine"):
        """
        Initialize the state machine.
        
        Args:
            initial_state: Name of the initial state
            name: Name of the state machine for logging
        """
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
        
        # State management
        self.states: Dict[str, State] = {}
        self.current_state: Optional[State] = None
        self.initial_state_name = initial_state
        
        # Transition management
        self.transition_handlers: Dict[Tuple[str, str], List[StateHandler]] = {}
        self.state_handlers: Dict[str, StateHandler] = {}
        self.transition_guards: Dict[Tuple[str, str], List[TransitionGuard]] = {}
        
        # History tracking
        self.state_history: List[Transition] = []
        self.max_history_length = 100
        
        # Event callbacks
        self.on_state_change_callbacks: List[Callable] = []
        
        # Context data
        self.context: Dict[str, Any] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Error handling
        self.error_handlers: Dict[str, Callable] = {}
        
        # Initialize states
        self._initialize_states()
    
    @abstractmethod
    def _initialize_states(self) -> None:
        """Initialize all states and transitions. Must be implemented by subclasses."""
        pass
    
    def add_state(self, state: State) -> None:
        """Add a state to the state machine."""
        if state.name in self.states:
            raise ValueError(f"State '{state.name}' already exists")
        
        self.states[state.name] = state
        self.logger.debug(f"Added state: {state.name}")
    
    def add_transition(self, from_state: str, to_state: str, 
                      handler: Optional[StateHandler] = None,
                      guard: Optional[TransitionGuard] = None) -> None:
        """
        Add a valid transition between states.
        
        Args:
            from_state: Source state name
            to_state: Target state name
            handler: Optional handler for this transition
            guard: Optional guard condition
        """
        if from_state not in self.states:
            raise ValueError(f"State '{from_state}' does not exist")
        
        if to_state not in self.states:
            raise ValueError(f"State '{to_state}' does not exist")
        
        # Add to allowed transitions
        self.states[from_state].allowed_transitions.add(to_state)
        
        # Add handler if provided
        if handler:
            key = (from_state, to_state)
            if key not in self.transition_handlers:
                self.transition_handlers[key] = []
            self.transition_handlers[key].append(handler)
        
        # Add guard if provided
        if guard:
            key = (from_state, to_state)
            if key not in self.transition_guards:
                self.transition_guards[key] = []
            self.transition_guards[key].append(guard)
        
        self.logger.debug(f"Added transition: {from_state} → {to_state}")
    
    def set_state_handler(self, state_name: str, handler: StateHandler) -> None:
        """Set a handler for a specific state."""
        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' does not exist")
        
        self.state_handlers[state_name] = handler
    
    async def start(self) -> None:
        """Start the state machine."""
        if self.initial_state_name not in self.states:
            raise ValueError(f"Initial state '{self.initial_state_name}' does not exist")
        
        self.current_state = self.states[self.initial_state_name]
        self.logger.info(f"State machine '{self.name}' started in state: {self.current_state.name}")
        
        # Execute entry handler for initial state
        await self._execute_state_entry(self.current_state)
    
    async def transition(self, new_state_name: str, trigger: Optional[str] = None) -> bool:
        """
        Transition to a new state.
        
        Args:
            new_state_name: Name of the target state
            trigger: Optional trigger that caused the transition
            
        Returns:
            True if transition successful, False otherwise
        """
        async with self._lock:
            if not self.current_state:
                self.logger.error("State machine not started")
                return False
            
            # Check if transition is allowed
            if not self.current_state.can_transition_to(new_state_name):
                self.logger.warning(
                    f"Invalid transition: {self.current_state.name} → {new_state_name}"
                )
                return False
            
            # Check guard conditions
            if not await self._check_guards(self.current_state.name, new_state_name):
                self.logger.warning(
                    f"Guard condition failed: {self.current_state.name} → {new_state_name}"
                )
                return False
            
            # Get target state
            new_state = self.states[new_state_name]
            
            # Create transition record
            transition = Transition(
                from_state=self.current_state.name,
                to_state=new_state_name,
                context=self.context.copy(),
                trigger=trigger
            )
            
            try:
                # Execute exit handler for current state
                await self._execute_state_exit(self.current_state)
                
                # Execute transition handlers
                await self._execute_transition_handlers(transition)
                
                # Update current state
                old_state = self.current_state
                self.current_state = new_state
                
                # Execute entry handler for new state
                await self._execute_state_entry(new_state)
                
                # Record transition in history
                self._record_transition(transition)
                
                # Notify callbacks
                await self._notify_state_change(old_state, new_state, transition)
                
                self.logger.info(
                    f"Transitioned: {transition.from_state} → {transition.to_state}"
                    f"{f' (trigger: {trigger})' if trigger else ''}"
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error during transition: {e}")
                await self._handle_transition_error(e, transition)
                return False
    
    async def _check_guards(self, from_state: str, to_state: str) -> bool:
        """Check all guard conditions for a transition."""
        key = (from_state, to_state)
        guards = self.transition_guards.get(key, [])
        
        for guard in guards:
            if not guard.check(self.context):
                self.logger.debug(f"Guard failed: {guard.description}")
                return False
        
        return True
    
    async def _execute_state_entry(self, state: State) -> None:
        """Execute entry handlers for a state."""
        # General state handler
        if state.name in self.state_handlers:
            handler = self.state_handlers[state.name]
            await handler.on_enter(state, self.context)
    
    async def _execute_state_exit(self, state: State) -> None:
        """Execute exit handlers for a state."""
        # General state handler
        if state.name in self.state_handlers:
            handler = self.state_handlers[state.name]
            await handler.on_exit(state, self.context)
    
    async def _execute_transition_handlers(self, transition: Transition) -> None:
        """Execute handlers for a specific transition."""
        key = (transition.from_state, transition.to_state)
        handlers = self.transition_handlers.get(key, [])
        
        for handler in handlers:
            await handler.on_transition(transition)
    
    def _record_transition(self, transition: Transition) -> None:
        """Record transition in history."""
        self.state_history.append(transition)
        
        # Limit history size
        if len(self.state_history) > self.max_history_length:
            self.state_history = self.state_history[-self.max_history_length:]
    
    async def _notify_state_change(self, old_state: State, new_state: State, 
                                  transition: Transition) -> None:
        """Notify all registered callbacks of state change."""
        for callback in self.on_state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_state, new_state, transition)
                else:
                    callback(old_state, new_state, transition)
            except Exception as e:
                self.logger.error(f"Error in state change callback: {e}")
    
    async def _handle_transition_error(self, error: Exception, 
                                     transition: Transition) -> None:
        """Handle errors during state transitions."""
        error_type = type(error).__name__
        
        if error_type in self.error_handlers:
            await self.error_handlers[error_type](error, transition)
        else:
            # Default error handling
            self.logger.error(
                f"Unhandled error during transition "
                f"{transition.from_state} → {transition.to_state}: {error}"
            )
    
    def register_state_change_callback(self, callback: Callable) -> None:
        """Register a callback to be notified of state changes."""
        self.on_state_change_callbacks.append(callback)
    
    def register_error_handler(self, error_type: str, 
                             handler: Callable[[Exception, Transition], None]) -> None:
        """Register an error handler for specific error types."""
        self.error_handlers[error_type] = handler
    
    def get_state_history(self, limit: Optional[int] = None) -> List[Transition]:
        """Get state transition history."""
        if limit:
            return self.state_history[-limit:]
        return self.state_history.copy()
    
    def get_current_state(self) -> Optional[State]:
        """Get the current state."""
        return self.current_state
    
    def update_context(self, updates: Dict[str, Any]) -> None:
        """Update the state machine context."""
        self.context.update(updates)
    
    def get_context(self) -> Dict[str, Any]:
        """Get the current context."""
        return self.context.copy()
    
    def is_in_state(self, state_name: str) -> bool:
        """Check if currently in a specific state."""
        return self.current_state and self.current_state.name == state_name
    
    def get_allowed_transitions(self) -> Set[str]:
        """Get all states the machine can currently transition to."""
        if not self.current_state:
            return set()
        return self.current_state.allowed_transitions.copy()
    
    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self.current_state = None
        self.state_history.clear()
        self.context.clear()
        self.logger.info(f"State machine '{self.name}' reset")