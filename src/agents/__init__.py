"""
Agents module for Autonomous Tool Discovery System.

This module contains intelligent agents that handle different aspects
of the tool discovery and execution pipeline.
"""

# Avoid circular imports by not importing IntentRecognitionAgent here
# Import it directly where needed: from src.agents.intent_recognition_agent import IntentRecognitionAgent

from .intent_models import (
    Intent,
    IntentResult,
    TextPreprocessor,
    MultiIntentHandler
)

from .orchestrator_agent import (
    OrchestratorAgent,
    ToolExecutionResult,
    OrchestrationResult
)

from .tool_discovery_agent import (
    ToolDiscoveryAgent,
    ToolCandidate
)

__all__ = [
    # Intent Recognition
    # 'IntentRecognitionAgent',  # Import directly to avoid circular imports
    'Intent',
    'IntentResult',
    'TextPreprocessor',
    'MultiIntentHandler',
    # Orchestrator
    'OrchestratorAgent',
    'ToolExecutionResult',
    'OrchestrationResult',
    # Tool Discovery
    'ToolDiscoveryAgent',
    'ToolCandidate'
]