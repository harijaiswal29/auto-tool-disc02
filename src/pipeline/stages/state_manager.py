"""
State Manager Pipeline Stage.

This module integrates the Conversation State Machine with the pipeline
architecture, tracking state transitions during query processing.
"""

from typing import Dict, Any, Optional

from src.pipeline.base import PipelineStage, PipelineData
from src.state_machine.conversation_state_machine import (
    ConversationStateMachine, ConversationStates
)
from src.state_machine.handlers import register_handlers


class StateManagerStage(PipelineStage):
    """
    Pipeline stage that manages conversation state throughout the pipeline.
    
    This stage:
    - Tracks the conversation state machine
    - Updates state based on pipeline progress
    - Provides state-based flow control
    - Records state history for debugging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the state manager stage."""
        super().__init__(name="StateManager", config=config)
        
        # Create and configure state machine
        self.state_machine = ConversationStateMachine()
        register_handlers(self.state_machine)
        
        # Track pipeline stage to state mapping
        self.stage_state_mapping = {
            'TextPreprocessor': ConversationStates.QUERY_RECEIVED,
            'Tokenizer': ConversationStates.QUERY_RECEIVED,
            'FeatureExtractor': ConversationStates.QUERY_RECEIVED,
            'IntentClassifier': ConversationStates.QUERY_RECEIVED,
            'ConfidenceScorer': ConversationStates.INTENT_RECOGNIZED
        }
        
        # Configuration
        self.auto_transition = config.get('auto_transition', True) if config else True
        self.track_metrics = config.get('track_metrics', True) if config else True
    
    async def _initialize(self):
        """Initialize the state machine."""
        await self.state_machine.start()
        self.logger.info("State manager initialized with conversation state machine")
    
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Process pipeline data and update state machine accordingly.
        
        Args:
            data: Pipeline data
            
        Returns:
            Pipeline data with state information
        """
        # Get current pipeline stage from processed data
        current_stage = self._get_current_stage(data)
        
        # Update state machine based on pipeline progress
        await self._update_state_machine(data, current_stage)
        
        # Add state information to pipeline data
        self._add_state_info(data)
        
        # Check for state-based flow control
        if not await self._check_flow_control(data):
            data.add_metadata('pipeline_halt', True)
            data.add_metadata('halt_reason', 'State machine flow control')
        
        return data
    
    def _get_current_stage(self, data: PipelineData) -> Optional[str]:
        """Determine the current pipeline stage."""
        # Find the last processed stage
        stages = list(data.processed_data.keys())
        return stages[-1] if stages else None
    
    async def _update_state_machine(self, data: PipelineData, current_stage: str):
        """Update state machine based on pipeline progress."""
        current_state = self.state_machine.get_current_state()
        
        # Handle initial query reception
        if current_state and current_state.name == ConversationStates.IDLE:
            query = data.raw_input
            if isinstance(query, str):
                await self.state_machine.receive_query(query, data.context)
        
        # Handle intent recognition completion
        if current_stage == 'ConfidenceScorer':
            primary_intent = data.get_stage_result('ConfidenceScorer', 'primary_intent')
            confidence_passed = data.get_stage_result('ConfidenceScorer', 'confidence_passed')
            
            if primary_intent and self.state_machine.is_in_state(ConversationStates.QUERY_RECEIVED):
                intent_dict = {
                    'type': primary_intent.type,
                    'keywords': primary_intent.keywords,
                    'confidence': primary_intent.confidence
                }
                
                confidence = primary_intent.confidence
                await self.state_machine.recognize_intent(intent_dict, confidence)
        
        # Track any errors
        if 'error' in data.metadata:
            error = data.metadata['error']
            error_context = {
                'stage': current_stage,
                'pipeline_data': data.metadata
            }
            await self.state_machine.handle_error(Exception(error), error_context)
    
    def _add_state_info(self, data: PipelineData):
        """Add state machine information to pipeline data."""
        current_state = self.state_machine.get_current_state()
        
        # Add current state info
        data.add_stage_result(self.name, 'current_state', 
                            current_state.name if current_state else None)
        data.add_stage_result(self.name, 'state_type',
                            current_state.type.value if current_state else None)
        
        # Add allowed transitions
        allowed_transitions = self.state_machine.get_allowed_transitions()
        data.add_stage_result(self.name, 'allowed_transitions', list(allowed_transitions))
        
        # Add conversation summary
        summary = self.state_machine.get_conversation_summary()
        data.add_stage_result(self.name, 'conversation_summary', summary)
        
        # Add to metadata for easy access
        data.add_metadata('conversation_state', current_state.name if current_state else None)
        data.add_metadata('state_machine_context', self.state_machine.get_context())
        
        # Track metrics if enabled
        if self.track_metrics:
            self._track_state_metrics(data)
    
    def _track_state_metrics(self, data: PipelineData):
        """Track state machine metrics."""
        metrics = {
            'state_transitions': len(self.state_machine.get_state_history()),
            'error_count': len(self.state_machine.context.get('error_history', [])),
            'retry_count': self.state_machine.context.get('retry_count', 0),
            'clarification_attempts': self.state_machine.context.get('clarification_attempts', 0)
        }
        
        data.add_stage_result(self.name, 'state_metrics', metrics)
    
    async def _check_flow_control(self, data: PipelineData) -> bool:
        """
        Check if pipeline should continue based on state.
        
        Returns:
            True if pipeline should continue, False to halt
        """
        current_state = self.state_machine.get_current_state()
        if not current_state:
            return True
        
        # States that require pipeline halt
        halt_states = [
            ConversationStates.CLARIFICATION_NEEDED,
            ConversationStates.NO_TOOLS_FOUND,
            ConversationStates.ERROR,
            ConversationStates.TIMEOUT,
            ConversationStates.USER_CANCELLED
        ]
        
        if current_state.name in halt_states:
            self.logger.info(f"Halting pipeline due to state: {current_state.name}")
            return False
        
        return True
    
    async def validate_input(self, data: PipelineData) -> bool:
        """Validate that state machine is ready for input."""
        if not self.state_machine.get_current_state():
            self.logger.error("State machine not initialized")
            return False
        
        return True
    
    async def handle_clarification(self, clarification: str) -> bool:
        """
        Handle user clarification.
        
        Args:
            clarification: User's clarification text
            
        Returns:
            True if handled successfully
        """
        return await self.state_machine.handle_clarification(clarification)
    
    async def handle_tool_discovery(self, tools: list) -> bool:
        """
        Handle tool discovery results.
        
        Args:
            tools: List of discovered tools
            
        Returns:
            True if handled successfully
        """
        return await self.state_machine.discover_tools(tools)
    
    async def handle_execution_start(self, selected_tools: list) -> bool:
        """
        Handle start of tool execution.
        
        Args:
            selected_tools: List of selected tool IDs
            
        Returns:
            True if handled successfully
        """
        return await self.state_machine.start_execution(selected_tools)
    
    async def handle_execution_complete(self, results: list, success: bool) -> bool:
        """
        Handle completion of tool execution.
        
        Args:
            results: Execution results
            success: Whether execution was successful
            
        Returns:
            True if handled successfully
        """
        return await self.state_machine.complete_execution(results, success)
    
    async def handle_feedback(self, feedback: Dict[str, Any]) -> bool:
        """
        Handle user feedback.
        
        Args:
            feedback: User feedback data
            
        Returns:
            True if handled successfully
        """
        return await self.state_machine.receive_feedback(feedback)
    
    async def request_retry(self) -> bool:
        """
        Request to retry the operation.
        
        Returns:
            True if retry allowed
        """
        return await self.state_machine.request_retry()
    
    async def cancel_operation(self) -> bool:
        """
        Cancel the current operation.
        
        Returns:
            True if cancelled successfully
        """
        return await self.state_machine.cancel_operation()
    
    async def reset_to_idle(self) -> bool:
        """
        Reset state machine to idle state.
        
        Returns:
            True if reset successfully
        """
        return await self.state_machine.return_to_idle()
    
    def get_state_history(self, limit: Optional[int] = None) -> list:
        """
        Get state transition history.
        
        Args:
            limit: Maximum number of transitions to return
            
        Returns:
            List of state transitions
        """
        return self.state_machine.get_state_history(limit)
    
    def get_current_state_name(self) -> Optional[str]:
        """Get the current state name."""
        state = self.state_machine.get_current_state()
        return state.name if state else None
    
    def is_in_error_state(self) -> bool:
        """Check if currently in an error state."""
        current_state = self.state_machine.get_current_state()
        if not current_state:
            return False
        
        error_states = [
            ConversationStates.ERROR,
            ConversationStates.EXECUTION_FAILED,
            ConversationStates.TIMEOUT
        ]
        
        return current_state.name in error_states
    
    def needs_user_input(self) -> bool:
        """Check if waiting for user input."""
        current_state = self.state_machine.get_current_state()
        if not current_state:
            return False
        
        input_states = [
            ConversationStates.CLARIFICATION_NEEDED,
            ConversationStates.NO_TOOLS_FOUND,
            ConversationStates.EXECUTION_FAILED
        ]
        
        return current_state.name in input_states