"""
Orchestrator Agent for Autonomous Tool Discovery System.

This agent coordinates between intent recognition, tool discovery, and tool execution
to provide end-to-end query processing capabilities.
"""

import asyncio
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.agents.intent_models import IntentResult
from src.agents.result_cache import ResultCache
from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger
from src.state_machine.conversation_state_machine import ConversationStateMachine, ConversationStates
from src.state_machine.handlers import register_handlers
from src.learning.q_learning_engine import QLearningEngine
from src.learning.reward_calculator import RewardCalculator, ExecutionMetrics
from src.learning.context_extractor import ContextExtractor, UserContext
from src.database.database import DatabaseManager
import numpy as np
import psutil
import resource
import uuid


@dataclass
class ToolExecutionResult:
    """Result from executing a tool with enhanced metrics."""
    tool_id: str
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    # Enhanced fields for partial success and resource tracking
    partial_success: bool = False
    completion_percentage: float = 0.0
    error_type: Optional[str] = None
    retry_count: int = 0
    resource_usage: Optional[Dict[str, float]] = None
    result_quality: float = 1.0  # Quality score 0-1


@dataclass
class OrchestrationResult:
    """Final result from orchestrating a user query."""
    query: str
    intent: IntentResult
    discovered_tools: List[Dict[str, Any]]
    selected_tools: List[str]
    execution_results: List[ToolExecutionResult]
    total_time_ms: float
    success: bool
    summary: str


class OrchestratorAgent:
    """
    Main orchestrator that coordinates the entire tool discovery and execution pipeline.
    
    This agent:
    1. Receives user queries
    2. Identifies intent using IntentRecognitionAgent
    3. Discovers relevant tools based on intent
    4. Selects optimal tools for execution
    5. Executes tools via MCP Integration
    6. Aggregates and returns results
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Orchestrator Agent."""
        self.logger = get_logger(__name__)
        
        # Load configuration
        if config is None:
            config = self._load_default_config()
        self.config = config
        
        # Initialize components
        self.logger.info("Initializing Orchestrator Agent components...")
        
        # Intent Recognition
        self.intent_agent = IntentRecognitionAgent(config.get('intent_recognition', {}))
        
        # MCP Integration
        self.mcp_integration = MCPIntegration(config)
        
        # Tool Registry
        registry_path = config.get('database', {}).get('tool_registry', 'data/registry/tools.db')
        self.tool_registry = ToolRegistry(registry_path)
        
        # State Machine
        self.state_machine = ConversationStateMachine()
        self._state_machine_initialized = False
        
        # Intent to capability mapping
        self.intent_capability_map = {
            'query.search': ['search', 'find', 'query', 'list'],
            'query.retrieve': ['read', 'get', 'fetch', 'retrieve'],
            'query.analyze': ['analyze', 'examine', 'inspect', 'evaluate'],
            'action.create': ['create', 'write', 'generate', 'make'],
            'action.modify': ['update', 'edit', 'modify', 'change'],
            'action.delete': ['delete', 'remove', 'clear', 'drop'],
            'system.configure': ['configure', 'setup', 'initialize'],
            'system.monitor': ['monitor', 'track', 'watch', 'observe']
        }
        
        # Initialize Q-Learning Engine
        q_learning_enabled = config.get('q_learning', {}).get('enable_learning', False)
        if q_learning_enabled:
            self.logger.info("Initializing Q-Learning Engine...")
            self.q_learning_engine = QLearningEngine(config)
            # Try to load existing model
            asyncio.create_task(self._load_q_learning_model())
        else:
            self.q_learning_engine = None
            self.logger.info("Q-Learning disabled in configuration")
        
        # Initialize enhanced reward calculator
        self.reward_calculator = RewardCalculator(config)
        
        # Initialize context extractor for context-aware patterns
        self.context_extractor = ContextExtractor()
        
        # Initialize database manager for failure tracking
        self.db_manager = DatabaseManager()
        asyncio.create_task(self.db_manager.initialize())
        
        # Track execution history for learning
        self.execution_history = []
        
        # Track failure metrics for enhanced state representation
        self.failure_metrics = {
            'failure_rates': {},
            'failure_types': {},
            'retry_patterns': {}
        }
        self.current_state = None
        
        # Initialize context for session tracking
        self.context = {
            'session_start': datetime.now()
        }
        
        # Track user statistics for context extraction
        self.user_stats = {
            'success_rate': 0.5,
            'query_count': 0,
            'avg_tools_used': 1.0
        }
        
        # Initialize result cache
        self.result_cache = ResultCache(config.get('result_cache', {}))
        
        # Initialize cache monitoring if enabled
        self.cache_monitor = None
        cache_monitor_config = config.get('cache_monitoring', {})
        if cache_monitor_config.get('enabled', False):
            from src.monitoring.cache_metrics_monitor import CacheMetricsMonitor
            self.cache_monitor = CacheMetricsMonitor(config)
            self.cache_monitor.attach_cache(self.result_cache)
            
            # Set pattern extractor for query pattern analysis
            self.result_cache.set_pattern_extractor(self._extract_query_pattern)
        
        self.logger.info("Orchestrator Agent initialized successfully")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from config file."""
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Fallback configuration
        return {
            'orchestration': {
                'max_tools_per_query': 3,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            }
        }
    
    async def process_user_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """
        Process a user query end-to-end.
        
        Args:
            query: Natural language query from user
            context: Optional context information
            
        Returns:
            OrchestrationResult with complete execution details
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        self.current_session_id = execution_id
        
        if context is None:
            context = {}
        
        # Store current query in context
        self.context['current_query'] = query
        
        self.logger.info(f"Processing user query: {query}")
        
        # Check cache first
        cache_context = {
            'domain': context.get('domain'),
            'user_expertise': context.get('user_expertise')
        }
        cache_key = self.result_cache.generate_key(query, cache_context)
        cached_result = self.result_cache.get(cache_key)
        
        if cached_result is not None:
            self.logger.info(f"Cache hit for query: {query}")
            # Update metrics for cached result
            cache_metrics = self.result_cache.get_metrics()
            self.logger.debug(f"Cache hit rate: {cache_metrics['hit_rate']:.2%}")
            
            # Track in monitoring if enabled
            if self.cache_monitor:
                response_time_ms = (time.time() - start_time) * 1000
                pattern = self._extract_query_pattern(cache_key)
                self.cache_monitor.track_query_pattern(query, pattern, True, response_time_ms)
            
            return cached_result
        
        try:
            # Step 0: Update state machine - receive query
            if not await self.state_machine.receive_query(query, context):
                raise ValueError("Failed to receive query in state machine")
            
            # Step 1: Recognize intent
            intent_result = await self.intent_agent.process_query(query, context)
            self.logger.info(f"Intent recognized: {intent_result.primary_intent.type} "
                           f"(confidence: {intent_result.primary_intent.confidence:.2f})")
            
            # Step 1.5: Extract user context for context-aware patterns
            self.user_stats['query_count'] += 1
            user_context = self.context_extractor.extract_context(
                query=query,
                user_stats=self.user_stats,
                intent_type=intent_result.primary_intent.type
            )
            self.logger.info(f"User context extracted - Expertise: {user_context.user_expertise}, Domain: {user_context.domain}")
            
            # Store intent confidence and context for reward calculation
            self.context['last_intent_confidence'] = intent_result.primary_intent.confidence
            self.context['user_context'] = user_context
            
            # Update state machine with intent
            intent_dict = {
                'type': intent_result.primary_intent.type,
                'keywords': intent_result.primary_intent.keywords,
                'confidence': intent_result.primary_intent.confidence
            }
            await self.state_machine.recognize_intent(intent_dict, intent_result.primary_intent.confidence)
            
            # Step 2: Discover tools based on intent
            discovered_tools = await self.discover_tools_for_intent(intent_result)
            self.logger.info(f"Discovered {len(discovered_tools)} potential tools")
            
            # Update state machine with discovered tools
            tool_ids = [tool['id'] for tool in discovered_tools]
            await self.state_machine.discover_tools(tool_ids)
            
            # Step 3: Select best tools for execution
            selected_tools = await self.select_tools(discovered_tools, intent_result)
            self.logger.info(f"Selected {len(selected_tools)} tools for execution")
            
            # Check if we're in the right state to execute
            current_state_name = self.state_machine.get_current_state().name if self.state_machine.get_current_state() else None
            if current_state_name != ConversationStates.TOOLS_DISCOVERED:
                # Handle edge cases like NO_TOOLS_FOUND or CLARIFICATION_NEEDED
                if current_state_name == ConversationStates.NO_TOOLS_FOUND:
                    return OrchestrationResult(
                        query=query,
                        intent=intent_result,
                        discovered_tools=[],
                        selected_tools=[],
                        execution_results=[],
                        total_time_ms=(time.time() - start_time) * 1000,
                        success=False,
                        summary="No tools found for the given query. Please try rephrasing or provide more details."
                    )
                elif current_state_name == ConversationStates.CLARIFICATION_NEEDED:
                    return OrchestrationResult(
                        query=query,
                        intent=intent_result,
                        discovered_tools=discovered_tools,
                        selected_tools=[],
                        execution_results=[],
                        total_time_ms=(time.time() - start_time) * 1000,
                        success=False,
                        summary="Query is ambiguous. Please provide more details or clarify your intent."
                    )
            
            # Step 4: Execute selected tools
            # Update state machine - start execution
            selected_tool_ids = [tool['id'] for tool in selected_tools]
            await self.state_machine.start_execution(selected_tool_ids)
            
            execution_results = await self.execute_tools(selected_tools, query, context)
            
            # Update state machine - complete execution
            execution_success = any(r.success for r in execution_results) if execution_results else False
            results_for_state = [
                {
                    'tool': r.tool_name,
                    'success': r.success,
                    'result': r.result,
                    'error': r.error
                } for r in execution_results
            ]
            await self.state_machine.complete_execution(results_for_state, execution_success)
            
            # Step 5: Update Q-learning if enabled
            if self.q_learning_engine and execution_results:
                await self._update_q_learning(execution_results, intent_result)
            
            # Step 6: Generate summary
            summary = self._generate_summary(intent_result, execution_results)
            
            # Calculate total time
            total_time_ms = (time.time() - start_time) * 1000
            
            # Determine overall success
            success = any(r.success for r in execution_results) if execution_results else False
            
            result = OrchestrationResult(
                query=query,
                intent=intent_result,
                discovered_tools=discovered_tools,
                selected_tools=[t['id'] for t in selected_tools],
                execution_results=execution_results,
                total_time_ms=total_time_ms,
                success=success,
                summary=summary
            )
            
            # Cache successful results
            if result.success:
                # Update cache context with intent information
                cache_context['intent_type'] = intent_result.primary_intent.type
                cache_context['intent_confidence'] = intent_result.primary_intent.confidence
                cache_context['domain'] = user_context.domain if 'user_context' in locals() else cache_context.get('domain')
                cache_context['user_expertise'] = user_context.user_expertise if 'user_context' in locals() else cache_context.get('user_expertise')
                
                # Generate cache key with full context
                cache_key = self.result_cache.generate_key(query, cache_context)
                self.result_cache.put(cache_key, result)
                self.logger.debug(f"Cached result for query: {query}")
                
                # Track successful cache storage in monitoring
                if self.cache_monitor:
                    pattern = self._extract_query_pattern(cache_key)
                    self.cache_monitor.track_query_pattern(query, pattern, False, total_time_ms)
            
            # Return to idle state for next query
            await self.state_machine.return_to_idle()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            total_time_ms = (time.time() - start_time) * 1000
            
            # Update state machine with error
            await self.state_machine.handle_error(e, {'stage': 'orchestration', 'query': query})
            
            result = OrchestrationResult(
                query=query,
                intent=intent_result if 'intent_result' in locals() else None,
                discovered_tools=[],
                selected_tools=[],
                execution_results=[],
                total_time_ms=total_time_ms,
                success=False,
                summary=f"Error processing query: {str(e)}"
            )
            
            # Try to return to idle state
            try:
                await self.state_machine.return_to_idle()
            except:
                pass
                
            return result
    
    async def discover_tools_for_intent(self, intent_result: IntentResult) -> List[Dict[str, Any]]:
        """
        Discover tools that match the identified intent.
        
        Args:
            intent_result: Result from intent recognition
            
        Returns:
            List of discovered tools with metadata
        """
        discovered_tools = []
        
        # Get capabilities associated with the intent
        primary_intent = intent_result.primary_intent.type
        required_capabilities = self.intent_capability_map.get(primary_intent, [])
        
        # Also consider keywords from the query
        query_keywords = intent_result.primary_intent.keywords
        
        # Search for tools with matching capabilities
        for capability in required_capabilities:
            tools = await self.tool_registry.search_tools(capability)
            for tool in tools:
                # Enhance tool data with relevance score
                tool['relevance_score'] = self._calculate_relevance_score(
                    tool, 
                    intent_result, 
                    required_capabilities,
                    query_keywords
                )
                if tool not in discovered_tools:
                    discovered_tools.append(tool)
        
        # Sort by relevance score
        discovered_tools.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return discovered_tools
    
    def _calculate_relevance_score(self, tool: Dict[str, Any], 
                                   intent_result: IntentResult,
                                   required_capabilities: List[str],
                                   query_keywords: List[str]) -> float:
        """Calculate relevance score for a tool based on intent match."""
        score = 0.0
        
        # Check capability match
        tool_capabilities = tool.get('capabilities', {})
        if isinstance(tool_capabilities, str):
            try:
                tool_capabilities = json.loads(tool_capabilities)
            except:
                tool_capabilities = {}
        
        # Score based on capability overlap
        if isinstance(tool_capabilities, dict):
            tool_cap_list = tool_capabilities.get('operations', [])
            if tool_cap_list:
                for cap in required_capabilities:
                    if any(cap in str(op) for op in tool_cap_list):
                        score += 0.3
        
        # Score based on keyword match
        tool_name = tool.get('name', '').lower()
        tool_type = tool.get('type', '').lower()
        
        for keyword in query_keywords:
            if keyword in tool_name or keyword in tool_type:
                score += 0.2
        
        # Boost score based on performance
        performance_score = tool.get('performance_score', 0.5)
        score += performance_score * 0.2
        
        # Boost score based on intent confidence
        score += intent_result.primary_intent.confidence * 0.3
        
        return min(score, 1.0)
    
    async def select_tools(self, discovered_tools: List[Dict[str, Any]], 
                          intent_result: IntentResult) -> List[Dict[str, Any]]:
        """
        Select the best tools for execution based on various criteria.
        
        Args:
            discovered_tools: List of discovered tools
            intent_result: Intent recognition result
            
        Returns:
            List of selected tools
        """
        if not discovered_tools:
            return []
        
        # Check if Q-learning is enabled
        if self.q_learning_engine and self.config.get('q_learning', {}).get('enable_learning', False):
            # Use Q-learning for tool selection
            return await self._select_tools_with_q_learning(discovered_tools, intent_result)
        else:
            # Use traditional selection strategy
            return await self._select_tools_traditional(discovered_tools, intent_result)
    
    async def _select_tools_traditional(self, discovered_tools: List[Dict[str, Any]], 
                                       intent_result: IntentResult) -> List[Dict[str, Any]]:
        """Traditional tool selection without Q-learning."""
        # Get max tools to select
        max_tools = self.config.get('orchestration', {}).get('max_tools_per_query', 3)
        
        # Apply selection strategy
        strategy = self.config.get('orchestration', {}).get('tool_selection_strategy', 'performance_weighted')
        
        if strategy == 'performance_weighted':
            # Weight by both relevance and performance
            for tool in discovered_tools:
                tool['selection_score'] = (
                    tool.get('relevance_score', 0) * 0.7 +
                    tool.get('performance_score', 0.5) * 0.3
                )
            discovered_tools.sort(key=lambda x: x.get('selection_score', 0), reverse=True)
        
        elif strategy == 'relevance_only':
            # Already sorted by relevance
            pass
        
        elif strategy == 'performance_only':
            discovered_tools.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
        
        # Select top tools
        selected = discovered_tools[:max_tools]
        
        # Filter out tools with very low scores
        selected = [t for t in selected if t.get('relevance_score', 0) > 0.3]
        
        return selected
    
    async def _select_tools_with_q_learning(self, discovered_tools: List[Dict[str, Any]], 
                                          intent_result: IntentResult) -> List[Dict[str, Any]]:
        """Select tools using Q-learning engine."""
        # Encode current state with user context
        user_context = self.context.get('user_context')
        context = {
            'domain': user_context.domain if user_context is not None else 'general',
            'user_expertise': user_context.user_expertise if user_context is not None else 'intermediate',
            'query_count': len(self.execution_history),
            'success_rate': self._calculate_success_rate(),
            'metrics': {
                'avg_response_time': self._calculate_avg_response_time(),
                'tools_invoked': len(discovered_tools)
            }
        }
        
        # Get recent tool history
        recent_tools = [h['tools'] for h in self.execution_history[-5:]]
        tool_history = [tool for tools in recent_tools for tool in tools][:20]
        
        # Add failure metrics to context
        context.update({
            'failure_rates': self.failure_metrics['failure_rates'],
            'failure_types': self.failure_metrics['failure_types'],
            'retry_patterns': self.failure_metrics['retry_patterns']
        })
        
        # Encode state
        state = self.q_learning_engine.state_encoder.encode_state(
            intent_result, context, tool_history
        )
        self.current_state = state
        
        # Get available tool IDs
        available_tool_ids = [tool['id'] for tool in discovered_tools]
        
        # Define constraints (tool relationships)
        constraints = await self._get_tool_constraints(available_tool_ids)
        
        # Select action (tool combination) using Q-learning with context
        selected_tool_ids = await self.q_learning_engine.select_action(
            state, available_tool_ids, constraints, context=context
        )
        
        # Map selected IDs back to tool objects
        selected_tools = []
        for tool_id in selected_tool_ids:
            tool = next((t for t in discovered_tools if t['id'] == tool_id), None)
            if tool:
                selected_tools.append(tool)
        
        self.logger.info(f"Q-learning selected tools: {[t['name'] for t in selected_tools]}")
        
        return selected_tools
    
    async def execute_tools(self, selected_tools: List[Dict[str, Any]], 
                           query: str, context: Dict[str, Any]) -> List[ToolExecutionResult]:
        """
        Execute the selected tools.
        
        Args:
            selected_tools: List of tools to execute
            query: Original user query
            context: Execution context
            
        Returns:
            List of execution results
        """
        results = []
        
        if not selected_tools:
            self.logger.warning("No tools selected for execution")
            return results
        
        # Check if parallel execution is enabled
        parallel = self.config.get('orchestration', {}).get('parallel_execution', True)
        
        if parallel and len(selected_tools) > 1:
            # Execute tools in parallel
            tasks = []
            for tool in selected_tools:
                task = self._execute_single_tool(tool, query, context)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(ToolExecutionResult(
                        tool_id=selected_tools[i]['id'],
                        tool_name=selected_tools[i]['name'],
                        success=False,
                        result=None,
                        error=str(result),
                        execution_time_ms=0.0  # Default for exceptions in parallel execution
                    ))
                else:
                    final_results.append(result)
            results = final_results
        else:
            # Execute tools sequentially
            for tool in selected_tools:
                try:
                    result = await self._execute_single_tool(tool, query, context)
                    results.append(result)
                except Exception as e:
                    results.append(ToolExecutionResult(
                        tool_id=tool['id'],
                        tool_name=tool['name'],
                        success=False,
                        result=None,
                        error=str(e),
                        execution_time_ms=0.0  # Default for exceptions in sequential execution
                    ))
        
        return results
    
    async def _execute_single_tool(self, tool: Dict[str, Any], 
                                   query: str, context: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a single tool and return enhanced result metrics."""
        start_time = time.time()
        
        tool_id = tool['id']
        tool_name = tool['name']
        
        self.logger.info(f"Executing tool: {tool_name}")
        
        # Track resource usage at start
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu_percent = process.cpu_percent(interval=0.1)
        
        retry_count = 0
        error_type = None
        partial_success = False
        completion_percentage = 0.0
        result_quality = 1.0
        
        try:
            # Prepare tool input based on tool type
            tool_input = self._prepare_tool_input(tool, query, context)
            
            # Execute via MCP Integration (includes retry logic)
            result = await self.mcp_integration.execute_tool(tool_id, tool_input)
            
            # Get retry count from MCP integration if available
            if hasattr(self.mcp_integration, 'get_last_retry_count'):
                retry_count_result = self.mcp_integration.get_last_retry_count(tool_id)
                # Handle both sync and async results (for mocks)
                if asyncio.iscoroutine(retry_count_result):
                    retry_count = await retry_count_result
                else:
                    retry_count = retry_count_result
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Calculate resource usage
            end_memory = process.memory_info().rss / 1024 / 1024
            end_cpu_percent = process.cpu_percent(interval=0.1)
            
            resource_usage = {
                'memory_mb': end_memory - start_memory,
                'cpu_percent': (start_cpu_percent + end_cpu_percent) / 2,
                'execution_time_ms': execution_time_ms
            }
            
            # Evaluate result quality
            result_quality = self._evaluate_result_quality(result, tool_type=tool.get('type'))
            
            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                partial_success=False,
                completion_percentage=1.0,
                retry_count=retry_count,
                resource_usage=resource_usage,
                result_quality=result_quality
            )
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Classify error type
            error_type = self._classify_error(e)
            
            # Check for partial success
            partial_result = self._check_partial_success(e, tool_id)
            if partial_result:
                partial_success = True
                completion_percentage = partial_result.get('completion', 0.0)
                result = partial_result.get('data')
            
            # Calculate resource usage even for failures
            end_memory = process.memory_info().rss / 1024 / 1024
            end_cpu_percent = process.cpu_percent(interval=0.1)
            
            resource_usage = {
                'memory_mb': max(0, end_memory - start_memory),
                'cpu_percent': (start_cpu_percent + end_cpu_percent) / 2,
                'execution_time_ms': execution_time_ms
            }
            
            # Record failure in database
            if self.current_session_id:
                await self.db_manager.record_failure(
                    execution_id=self.current_session_id,
                    tool_id=tool_id,
                    failure_type=error_type,
                    error_message=str(e),
                    retry_count=retry_count,
                    recovery_successful=partial_success
                )
            
            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                success=False,
                result=result if partial_success else None,
                error=str(e),
                execution_time_ms=execution_time_ms,
                partial_success=partial_success,
                completion_percentage=completion_percentage,
                error_type=error_type,
                retry_count=retry_count,
                resource_usage=resource_usage,
                result_quality=0.0
            )
    
    def _prepare_tool_input(self, tool: Dict[str, Any], query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input parameters for tool execution based on tool requirements."""
        # This is a simplified version - in practice, this would be more sophisticated
        tool_input = {
            'query': query,
            'context': context
        }
        
        # Add tool-specific parameters based on tool type
        tool_type = tool.get('type', '').lower()
        
        if 'search' in tool_type:
            tool_input['search_query'] = query
        elif 'sqlite' in tool_type:
            tool_input['sql_query'] = self._convert_to_sql(query)
        elif 'filesystem' in tool_type:
            tool_input['path'] = context.get('working_directory', '.')
        
        return tool_input
    
    def _convert_to_sql(self, query: str) -> str:
        """Convert natural language query to SQL (simplified)."""
        # This is a very basic implementation
        query_lower = query.lower()
        
        if 'find' in query_lower or 'search' in query_lower:
            return "SELECT * FROM tools WHERE name LIKE '%{}%'".format(
                query.split()[-1] if query.split() else ''
            )
        elif 'create' in query_lower:
            return "INSERT INTO tools (name) VALUES ('new_tool')"
        else:
            return "SELECT * FROM tools LIMIT 10"
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for learning purposes."""
        error_str = str(error).lower()
        error_type_name = type(error).__name__
        
        # Network-related errors
        if any(keyword in error_str for keyword in ['timeout', 'timed out', 'connection timeout']):
            return 'network_timeout'
        elif any(keyword in error_str for keyword in ['connection', 'network', 'unreachable']):
            return 'connection_error'
        
        # Permission errors
        elif any(keyword in error_str for keyword in ['permission', 'access denied', 'unauthorized']):
            return 'permission_error'
        
        # Rate limiting
        elif any(keyword in error_str for keyword in ['rate limit', 'too many requests', 'throttled']):
            return 'rate_limit'
        
        # Known retryable errors
        elif error_type_name in ['RetryableError', 'TemporaryError']:
            return 'retryable'
        
        # Non-retryable errors
        elif error_type_name in ['NonRetryableError', 'InvalidArgumentError']:
            return 'non_retryable'
        
        else:
            return 'other'
    
    def _check_partial_success(self, error: Exception, tool_id: str) -> Optional[Dict[str, Any]]:
        """Check if there was a partial success despite the error."""
        # This would be more sophisticated in practice, checking for partial results
        # in the error object or from the tool's state
        
        # Example: Some tools might return partial results in the exception
        if hasattr(error, 'partial_result'):
            return {
                'data': error.partial_result,
                'completion': getattr(error, 'completion_percentage', 0.5)
            }
        
        # Check if error message indicates partial completion
        error_str = str(error).lower()
        if 'partial' in error_str or 'incomplete' in error_str:
            return {
                'data': None,
                'completion': 0.3  # Default partial completion
            }
        
        return None
    
    def _evaluate_result_quality(self, result: Any, tool_type: str = None) -> float:
        """Evaluate the quality of a tool's result (0-1 score)."""
        if result is None:
            return 0.0
        
        # Basic heuristics for result quality
        quality_score = 0.5  # Base score
        
        # Adjust based on result type
        if isinstance(result, dict):
            # Check if dict has meaningful content
            if len(result) > 0:
                quality_score += 0.2
            if 'error' not in result and 'warning' not in result:
                quality_score += 0.1
        
        elif isinstance(result, list):
            # Lists with content are good
            if len(result) > 0:
                quality_score += 0.3
            else:
                quality_score -= 0.2
        
        elif isinstance(result, str):
            # Non-empty strings
            if len(result.strip()) > 10:
                quality_score += 0.2
        
        # Tool-specific adjustments
        if tool_type:
            if 'search' in tool_type and isinstance(result, list) and len(result) > 5:
                quality_score += 0.1  # Good search results
            elif 'database' in tool_type and isinstance(result, dict) and 'rows' in result:
                quality_score += 0.1  # Proper database response
        
        return min(quality_score, 1.0)
    
    async def _save_execution_to_database(self, execution_record: Dict[str, Any], 
                                         intent_result: IntentResult):
        """Save execution record to database for pattern mining."""
        try:
            async with self.db_manager.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO execution_history 
                    (id, user_id, session_id, query, intent, tools_used, 
                     execution_time_ms, success, reward, user_expertise, domain, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    execution_record['execution_id'],
                    None,  # user_id - not implemented yet
                    execution_record['execution_id'],  # Using execution_id as session_id
                    execution_record['query'],
                    json.dumps({
                        'type': intent_result.primary_intent.type,
                        'confidence': intent_result.primary_intent.confidence,
                        'keywords': intent_result.primary_intent.keywords
                    }),
                    json.dumps(execution_record['tools']),
                    execution_record['execution_time_ms'],
                    execution_record['success'],
                    execution_record['reward'],
                    execution_record['user_expertise'],
                    execution_record['domain'],
                    datetime.now().isoformat()
                ))
                await conn.commit()
                self.logger.debug(f"Saved execution {execution_record['execution_id']} to database with context")
        except Exception as e:
            self.logger.error(f"Failed to save execution to database: {e}")
    
    def _update_user_stats(self, execution_results: List[ToolExecutionResult]):
        """Update user statistics based on execution results."""
        # Calculate success rate with exponential moving average
        success = any(r.success for r in execution_results)
        alpha = 0.1  # Learning rate for moving average
        self.user_stats['success_rate'] = (1 - alpha) * self.user_stats['success_rate'] + alpha * (1.0 if success else 0.0)
        
        # Update average tools used
        tools_used = len(execution_results)
        self.user_stats['avg_tools_used'] = (1 - alpha) * self.user_stats['avg_tools_used'] + alpha * tools_used
        
        self.logger.debug(f"Updated user stats - Success rate: {self.user_stats['success_rate']:.2f}, "
                         f"Avg tools: {self.user_stats['avg_tools_used']:.2f}")
    
    def _generate_summary(self, intent_result: IntentResult, 
                         execution_results: List[ToolExecutionResult]) -> str:
        """Generate a human-readable summary of the execution results."""
        if not execution_results:
            return "No tools were executed for this query."
        
        successful_tools = [r for r in execution_results if r.success]
        failed_tools = [r for r in execution_results if not r.success]
        
        summary_parts = []
        
        # Intent summary
        summary_parts.append(
            f"Intent: {intent_result.primary_intent.type} "
            f"(confidence: {intent_result.primary_intent.confidence:.2f})"
        )
        
        # Execution summary
        if successful_tools:
            summary_parts.append(f"Successfully executed {len(successful_tools)} tool(s):")
            for result in successful_tools:
                summary_parts.append(f"  - {result.tool_name}: {self._summarize_result(result.result)}")
        
        if failed_tools:
            summary_parts.append(f"Failed to execute {len(failed_tools)} tool(s):")
            for result in failed_tools:
                summary_parts.append(f"  - {result.tool_name}: {result.error}")
        
        return "\n".join(summary_parts)
    
    def _summarize_result(self, result: Any) -> str:
        """Create a brief summary of a tool's result."""
        if result is None:
            return "No result"
        elif isinstance(result, dict):
            return f"{len(result)} items returned"
        elif isinstance(result, list):
            return f"{len(result)} results found"
        elif isinstance(result, str):
            return result[:100] + "..." if len(result) > 100 else result
        else:
            return str(result)[:100]
    
    def get_state_machine_summary(self) -> Dict[str, Any]:
        """Get current state machine summary."""
        if not self._state_machine_initialized:
            return {"status": "not_initialized"}
        
        return self.state_machine.get_conversation_summary()
    
    def get_current_state(self) -> Optional[str]:
        """Get current conversation state."""
        if not self._state_machine_initialized:
            return None
        
        state = self.state_machine.get_current_state()
        return state.name if state else None
    
    async def handle_user_clarification(self, clarification: str) -> bool:
        """Handle user clarification when in CLARIFICATION_NEEDED state."""
        current_state = self.state_machine.get_current_state()
        if not current_state or current_state.name != ConversationStates.CLARIFICATION_NEEDED:
            self.logger.warning("Clarification received but not in CLARIFICATION_NEEDED state")
            return False
        
        return await self.state_machine.handle_clarification(clarification)
    
    async def request_retry(self) -> bool:
        """Request retry of failed operation."""
        return await self.state_machine.request_retry()
    
    async def cancel_current_operation(self) -> bool:
        """Cancel the current operation."""
        return await self.state_machine.cancel_operation()
    
    async def initialize(self):
        """Initialize all components and connections."""
        self.logger.info("Initializing Orchestrator components...")
        
        # Initialize State Machine
        if not self._state_machine_initialized:
            register_handlers(self.state_machine)
            await self.state_machine.start()
            self._state_machine_initialized = True
            self.logger.info("State machine initialized")
        
        # Initialize MCP Integration (which handles the registry)
        await self.mcp_integration.initialize()
        
        # Warm cache from execution history if available
        if self.execution_history:
            await self.warm_cache_from_history()
        
        self.logger.info("Orchestrator initialization complete")
    
    async def _load_q_learning_model(self):
        """Load saved Q-learning model if available."""
        if self.q_learning_engine:
            try:
                await self.q_learning_engine.db_manager.initialize()
                await self.q_learning_engine.load_model()
                self.logger.info("Q-learning model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load Q-learning model: {e}")
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate from execution history."""
        if not self.execution_history:
            return 0.5  # Default
        
        successes = sum(1 for h in self.execution_history if h.get('success', False))
        return successes / len(self.execution_history)
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from execution history."""
        if not self.execution_history:
            return 1000.0  # Default 1s
        
        times = [h.get('execution_time_ms', 1000) for h in self.execution_history]
        return sum(times) / len(times)
    
    async def _get_tool_constraints(self, tool_ids: List[str]) -> Dict[str, Any]:
        """Get tool relationship constraints from registry."""
        constraints = {
            'conflicts': {},
            'requires': {},
            'max_tools': self.config.get('orchestration', {}).get('max_tools_per_query', 3)
        }
        
        # Get relationships from tool registry
        for tool_id in tool_ids:
            relationships = await self.tool_registry.get_tool_relationships(tool_id)
            
            for rel in relationships:
                # Get the relationship type from the correct field
                rel_type = rel.get('relationship_type', rel.get('type', ''))
                
                # Determine the other tool in the relationship
                other_tool = rel['tool2_id'] if rel['tool1_id'] == tool_id else rel['tool1_id']
                
                if rel_type == 'conflicts':
                    if tool_id not in constraints['conflicts']:
                        constraints['conflicts'][tool_id] = []
                    constraints['conflicts'][tool_id].append(other_tool)
                
                elif rel_type == 'requires':
                    if tool_id not in constraints['requires']:
                        constraints['requires'][tool_id] = []
                    constraints['requires'][tool_id].append(other_tool)
        
        return constraints
    
    async def _update_q_learning(self, execution_results: List[ToolExecutionResult], 
                                intent_result: IntentResult):
        """Update Q-learning based on execution results."""
        if not self.q_learning_engine or self.current_state is None:
            return
        
        # Calculate reward based on execution results
        reward = await self._calculate_reward(execution_results)
        
        # Get tools that were executed
        executed_tools = tuple(r.tool_id for r in execution_results)
        
        # Prepare next state (after execution)
        user_context = self.context.get('user_context')
        next_context = {
            'domain': user_context.domain if user_context is not None else 'general',
            'user_expertise': user_context.user_expertise if user_context is not None else 'intermediate',
            'query_count': len(self.execution_history) + 1,
            'success_rate': self._calculate_success_rate(),
            'metrics': {
                'avg_response_time': self._calculate_avg_response_time(),
                'tools_invoked': len(execution_results)
            }
        }
        
        # Update tool history
        recent_tools = [h['tools'] for h in self.execution_history[-4:]]
        tool_history = [tool for tools in recent_tools for tool in tools]
        tool_history.extend([r.tool_id for r in execution_results])
        tool_history = tool_history[:20]
        
        # Add failure metrics to next context
        next_context.update({
            'failure_rates': self.failure_metrics['failure_rates'],
            'failure_types': self.failure_metrics['failure_types'],
            'retry_patterns': self.failure_metrics['retry_patterns']
        })
        
        # Encode next state
        next_state = self.q_learning_engine.state_encoder.encode_state(
            intent_result, next_context, tool_history
        )
        
        # Get available tools for next state (empty for now as episode ends)
        next_available_tools = []
        
        # Update Q-table
        await self.q_learning_engine.learn_from_experience(
            self.current_state,
            executed_tools,
            reward,
            next_state,
            next_available_tools,
            {},  # constraints
            done=True  # Episode complete
        )
        
        # Decay exploration rate
        self.q_learning_engine.decay_exploration()
        
        # Get user context
        user_context = self.context.get('user_context')
        
        # Update execution history
        # Handle potential coroutine in execution_time_ms (for mocks)
        exec_times = []
        for r in execution_results:
            exec_time = r.execution_time_ms
            if asyncio.iscoroutine(exec_time):
                exec_times.append(await exec_time)
            else:
                exec_times.append(exec_time)
        
        total_exec_time = sum(exec_times)
            
        execution_record = {
            'execution_id': self.current_session_id,
            'query': self.context.get('current_query', ''),
            'tools': [r.tool_id for r in execution_results],
            'success': any(r.success for r in execution_results),
            'execution_time_ms': total_exec_time,
            'intent': intent_result.primary_intent.type,
            'reward': reward,
            'timestamp': datetime.now(),
            'execution_results': execution_results,  # Store full results for feedback updates
            'context': {
                'mode': 'exploration' if getattr(self.q_learning_engine, 'exploration_rate', 0.2) > 0.15 else 'production',
                'intent_confidence': intent_result.primary_intent.confidence,
                'user_initiated': True
            },
            'user_expertise': user_context.user_expertise if user_context is not None else 'intermediate',
            'domain': user_context.domain if user_context is not None else 'general'
        }
        self.execution_history.append(execution_record)
        
        # Save to database for pattern mining
        await self._save_execution_to_database(execution_record, intent_result)
        
        # Update user statistics for future context extraction
        self._update_user_stats(execution_results)
        
        # Keep history size manageable
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        # Periodically save model
        if hasattr(self, 'execution_history') and isinstance(self.execution_history, list) and len(self.execution_history) % 10 == 0:
            await self.q_learning_engine.save_model()
            self.logger.info("Q-learning model saved")
    
    async def _calculate_reward(self, execution_results: List[ToolExecutionResult]) -> float:
        """Calculate reward using enhanced reward calculator."""
        if not execution_results:
            return -0.5  # Penalty for no results
        
        # Convert ToolExecutionResult to ExecutionMetrics for reward calculator
        execution_metrics = []
        for result in execution_results:
            metric = ExecutionMetrics(
                tool_id=result.tool_id,
                success=result.success,
                partial_success=result.partial_success,
                completion_percentage=result.completion_percentage,
                execution_time_ms=result.execution_time_ms,
                error_type=result.error_type,
                retry_count=result.retry_count,
                resource_usage=result.resource_usage,
                result_quality=result.result_quality
            )
            execution_metrics.append(metric)
        
        # Get context for reward calculation
        reward_context = {
            'mode': 'exploration' if getattr(self.q_learning_engine, 'exploration_rate', 0.2) > 0.15 else 'production',
            'intent_confidence': self.context.get('last_intent_confidence', 0.7),
            'user_initiated': True,  # Assuming user-initiated for now
            'session_duration': (datetime.now() - self.context.get('session_start', datetime.now())).seconds
        }
        
        # Calculate reward with breakdown
        total_reward, breakdown = self.reward_calculator.calculate_reward(
            execution_metrics, 
            reward_context,
            user_feedback=None  # Will be added when user feedback is available
        )
        
        # Log reward breakdown for analysis
        self.logger.debug(f"Reward breakdown: {breakdown}")
        
        # Update failure metrics for state representation
        await self._update_failure_metrics(execution_results)
        
        return total_reward
    
    async def _update_failure_metrics(self, execution_results: List[ToolExecutionResult]):
        """Update failure metrics for enhanced state representation."""
        # Update failure rates per tool
        for result in execution_results:
            tool_id = result.tool_id
            if tool_id not in self.failure_metrics['failure_rates']:
                self.failure_metrics['failure_rates'][tool_id] = 0.0
            
            # Update with exponential moving average
            alpha = 0.1  # Learning rate for failure tracking
            if result.success:
                self.failure_metrics['failure_rates'][tool_id] *= (1 - alpha)
            else:
                self.failure_metrics['failure_rates'][tool_id] = (
                    self.failure_metrics['failure_rates'][tool_id] * (1 - alpha) + alpha
                )
        
        # Update failure type distribution
        for result in execution_results:
            if not result.success and result.error_type:
                error_type = result.error_type
                self.failure_metrics['failure_types'][error_type] = (
                    self.failure_metrics['failure_types'].get(error_type, 0) + 1
                )
        
        # Update retry patterns
        # Handle potential mocks in retry_count
        retry_counts = []
        for r in execution_results:
            retry_count = getattr(r, 'retry_count', 0)
            if isinstance(retry_count, (int, float)):
                retry_counts.append(retry_count)
            else:
                retry_counts.append(0)
        avg_retry_count = sum(retry_counts) / len(execution_results) if execution_results else 0
        # Handle potential mocks in retry_count
        retried_results = []
        for r in execution_results:
            retry_count = getattr(r, 'retry_count', 0)
            if isinstance(retry_count, (int, float)) and retry_count > 0:
                retried_results.append(r)
        
        retry_success_rate = sum(1 for r in retried_results if r.success) / max(len(retried_results), 1)
        
        self.failure_metrics['retry_patterns'] = {
            'avg_retry_count': avg_retry_count,
            'retry_success_rate': retry_success_rate,
            'avg_retry_delay_ms': 1000,  # Would be calculated from actual retry delays
            'circuit_breaker_triggers': 0,  # Would be tracked from circuit breaker events
            'max_consecutive_failures': max(
                self.failure_metrics['retry_patterns'].get('max_consecutive_failures', 0),
                self._count_consecutive_failures()
            )
        }
        
        # Get failure rates from database for comprehensive view
        try:
            db_failure_rates = await self.db_manager.get_tool_failure_rates(time_window_hours=24)
            # Merge with in-memory rates
            for tool_id, rates in db_failure_rates.items():
                if tool_id in self.failure_metrics['failure_rates']:
                    # Weighted average with database rates
                    self.failure_metrics['failure_rates'][tool_id] = (
                        self.failure_metrics['failure_rates'][tool_id] * 0.3 + 
                        rates['failure_rate'] * 0.7
                    )
        except Exception as e:
            self.logger.warning(f"Failed to get failure rates from database: {e}")
    
    def _count_consecutive_failures(self) -> int:
        """Count consecutive failures in execution history."""
        if not self.execution_history:
            return 0
        
        consecutive = 0
        for history in reversed(self.execution_history):
            if history.get('success', False):
                break
            consecutive += 1
        
        return consecutive
    
    async def record_user_feedback(self, execution_id: str, feedback_type: str,
                                  rating: int = None, follow_up_query: str = None):
        """Record user feedback for learning and reward adjustment."""
        # Calculate derived feedback signals
        query_reformulated = False
        follow_up_time_seconds = None
        result_used = None
        
        # Check if this is a reformulation of previous query
        if follow_up_query and self.execution_history:
            last_query = self.execution_history[-1].get('query', '')
            # Simple similarity check - in practice would use semantic similarity
            if self._is_query_reformulation(last_query, follow_up_query):
                query_reformulated = True
            
            # Calculate time between queries
            last_timestamp = self.execution_history[-1].get('timestamp')
            if last_timestamp:
                follow_up_time_seconds = (datetime.now() - last_timestamp).total_seconds()
        
        # Infer result usage from feedback type
        if feedback_type == 'positive':
            result_used = True
        elif feedback_type == 'negative':
            result_used = False
        
        # Store in database
        await self.db_manager.record_user_feedback(
            execution_id=execution_id,
            feedback_type=feedback_type,
            rating=rating,
            query_reformulated=query_reformulated,
            follow_up_query=follow_up_query,
            follow_up_time_seconds=follow_up_time_seconds,
            result_used=result_used
        )
        
        # Update Q-learning with delayed reward if available
        if self.q_learning_engine and execution_id in [h.get('execution_id') for h in self.execution_history]:
            # Find the execution in history
            for i, history in enumerate(self.execution_history):
                if history.get('execution_id') == execution_id:
                    # Create feedback dict for reward recalculation
                    user_feedback = {
                        'rating': rating,
                        'query_reformulated': query_reformulated,
                        'follow_up_query': follow_up_query,
                        'follow_up_time_seconds': follow_up_time_seconds,
                        'result_used': result_used
                    }
                    
                    # Recalculate reward with user feedback
                    execution_results = history.get('execution_results', [])
                    if execution_results:
                        # Convert back to ExecutionMetrics
                        execution_metrics = []
                        for result in execution_results:
                            metric = ExecutionMetrics(
                                tool_id=result.tool_id,
                                success=result.success,
                                partial_success=result.partial_success,
                                completion_percentage=result.completion_percentage,
                                execution_time_ms=result.execution_time_ms,
                                error_type=result.error_type,
                                retry_count=result.retry_count,
                                resource_usage=result.resource_usage,
                                result_quality=result.result_quality
                            )
                            execution_metrics.append(metric)
                        
                        # Get context
                        reward_context = history.get('context', {})
                        
                        # Recalculate reward with user feedback
                        new_reward, breakdown = self.reward_calculator.calculate_reward(
                            execution_metrics,
                            reward_context,
                            user_feedback=user_feedback
                        )
                        
                        # Update Q-value with new reward
                        # This would require storing state-action pairs in history
                        self.logger.info(f"Updated reward with user feedback: {history.get('reward')} -> {new_reward}")
                        
                        # Update history
                        self.execution_history[i]['reward'] = new_reward
                        self.execution_history[i]['user_feedback'] = user_feedback
                    
                    break
    
    def _is_query_reformulation(self, query1: str, query2: str) -> bool:
        """Check if query2 is a reformulation of query1."""
        # Simple heuristic - check for significant overlap in words
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0
        
        # If high similarity but not identical, likely a reformulation
        return 0.3 < similarity < 0.9
    
    async def get_execution_feedback_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of user feedback for recent executions."""
        # This would query the database for feedback statistics
        # For now, return a placeholder
        return {
            'total_feedback': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'average_rating': 0.0,
            'reformulation_rate': 0.0,
            'result_usage_rate': 0.0
        }
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        return self.result_cache.get_metrics()
    
    def clear_cache(self):
        """Clear the result cache."""
        self.result_cache.clear()
        self.logger.info("Result cache cleared")
    
    def invalidate_cache_for_tool(self, tool_id: str):
        """Invalidate cache entries that used a specific tool."""
        # Invalidate based on tool ID pattern in the cache
        self.result_cache.invalidate(pattern=tool_id)
        self.logger.info(f"Invalidated cache entries for tool: {tool_id}")
    
    async def warm_cache_from_history(self):
        """Warm the cache from execution history."""
        if self.execution_history:
            self.result_cache.warm_cache(self.execution_history)
            self.logger.info(f"Warmed cache from {len(self.execution_history)} execution history entries")
    
    def save_cache(self):
        """Save cache to persistent storage."""
        self.result_cache.save_cache()
    
    def _extract_query_pattern(self, query: str) -> str:
        """
        Extract a pattern from a query for pattern-based analysis.
        
        Args:
            query: The query or cache key
            
        Returns:
            Pattern string
        """
        # If it's a hash (cache key), use the first 8 chars
        if len(query) == 32 and all(c in '0123456789abcdef' for c in query):
            return f"hash_{query[:8]}"
        
        # Otherwise, extract pattern from query
        query_lower = query.lower()
        
        # Common patterns
        if 'find' in query_lower and 'file' in query_lower:
            return 'find_files'
        elif 'search' in query_lower:
            return 'search_query'
        elif 'create' in query_lower:
            return 'create_action'
        elif 'analyze' in query_lower:
            return 'analyze_data'
        elif 'list' in query_lower:
            return 'list_items'
        elif 'get' in query_lower or 'fetch' in query_lower:
            return 'retrieve_data'
        elif 'update' in query_lower or 'modify' in query_lower:
            return 'update_action'
        elif 'delete' in query_lower or 'remove' in query_lower:
            return 'delete_action'
        else:
            # Extract first verb if possible
            words = query_lower.split()
            if words:
                return f"query_{words[0][:10]}"
            return 'unknown_pattern'
    
    async def start_cache_monitoring(self):
        """Start cache monitoring if configured."""
        if self.cache_monitor and not self.cache_monitor.monitoring_active:
            await self.cache_monitor.start_monitoring()
            self.logger.info("Cache monitoring started")
    
    async def stop_cache_monitoring(self):
        """Stop cache monitoring if active."""
        if self.cache_monitor and self.cache_monitor.monitoring_active:
            await self.cache_monitor.stop_monitoring()
            self.logger.info("Cache monitoring stopped")
    
    def get_cache_monitor(self):
        """Get the cache monitor instance."""
        return self.cache_monitor
    
    async def shutdown(self):
        """Cleanup and shutdown all components."""
        self.logger.info("Shutting down Orchestrator...")
        
        # Stop cache monitoring if active
        await self.stop_cache_monitoring()
        
        # Save cache before shutdown
        self.save_cache()
        
        # Shutdown MCP Integration (which handles the registry)
        await self.mcp_integration.shutdown()
        
        self.logger.info("Orchestrator shutdown complete")


# Example usage
if __name__ == "__main__":
    async def test_orchestrator():
        """Test the Orchestrator Agent."""
        orchestrator = OrchestratorAgent()
        
        # Initialize
        await orchestrator.initialize()
        
        # Test queries
        test_queries = [
            "Find all Python files in the project",
            "Search for information about machine learning",
            "Create a new configuration file",
            "Analyze the database schema"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            result = await orchestrator.process_user_query(query)
            
            print(f"\nSummary:\n{result.summary}")
            print(f"\nTotal execution time: {result.total_time_ms:.2f}ms")
            print(f"Success: {result.success}")
        
        # Shutdown
        await orchestrator.shutdown()
    
    # Run the test
    asyncio.run(test_orchestrator())