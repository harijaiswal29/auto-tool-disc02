"""
Orchestrator Agent for Autonomous Tool Discovery System.

This agent coordinates between intent recognition, tool discovery, and tool execution
to provide end-to-end query processing capabilities.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.intent_recognition_agent import IntentRecognitionAgent, IntentResult
from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger
from src.state_machine.conversation_state_machine import ConversationStateMachine, ConversationStates
from src.state_machine.handlers import register_handlers


@dataclass
class ToolExecutionResult:
    """Result from executing a tool."""
    tool_id: str
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


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
        
        if context is None:
            context = {}
        
        self.logger.info(f"Processing user query: {query}")
        
        try:
            # Step 0: Update state machine - receive query
            if not await self.state_machine.receive_query(query, context):
                raise ValueError("Failed to receive query in state machine")
            
            # Step 1: Recognize intent
            intent_result = await self.intent_agent.process_query(query, context)
            self.logger.info(f"Intent recognized: {intent_result.primary_intent.type} "
                           f"(confidence: {intent_result.primary_intent.confidence:.2f})")
            
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
            if not self.state_machine.is_in_state(ConversationStates.TOOLS_DISCOVERED):
                # Handle edge cases like NO_TOOLS_FOUND or CLARIFICATION_NEEDED
                if self.state_machine.is_in_state(ConversationStates.NO_TOOLS_FOUND):
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
                elif self.state_machine.is_in_state(ConversationStates.CLARIFICATION_NEEDED):
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
            
            # Step 5: Generate summary
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
                        error=str(result)
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
                        error=str(e)
                    ))
        
        return results
    
    async def _execute_single_tool(self, tool: Dict[str, Any], 
                                   query: str, context: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a single tool and return the result."""
        start_time = time.time()
        
        tool_id = tool['id']
        tool_name = tool['name']
        
        self.logger.info(f"Executing tool: {tool_name}")
        
        try:
            # Prepare tool input based on tool type
            tool_input = self._prepare_tool_input(tool, query, context)
            
            # Execute via MCP Integration
            result = await self.mcp_integration.execute_tool(tool_id, tool_input)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            execution_time_ms = (time.time() - start_time) * 1000
            
            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time_ms
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
        if not self.state_machine.is_in_state(ConversationStates.CLARIFICATION_NEEDED):
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
        
        self.logger.info("Orchestrator initialization complete")
    
    async def shutdown(self):
        """Cleanup and shutdown all components."""
        self.logger.info("Shutting down Orchestrator...")
        
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