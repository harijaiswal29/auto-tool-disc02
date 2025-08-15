"""Tool execution wrapper for proper reward calculation.

This module wraps tool execution to create proper ExecutionMetrics
for the reward calculator, ensuring realistic execution times and
proper success/failure tracking.
"""

import asyncio
import random
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.learning.reward_calculator import ExecutionMetrics

logger = logging.getLogger(__name__)


class ToolExecutionWrapper:
    """Wraps tool execution to create proper ExecutionMetrics."""
    
    def __init__(self):
        """Initialize the wrapper."""
        # Realistic execution times for different tool types (in ms)
        self.execution_times = {
            'filesystem': (50, 200),      # File operations
            'search': (200, 500),          # Web search
            'database': (100, 300),        # Database queries
            'github': (300, 800),          # GitHub API
            'financial': (400, 1000),      # Financial data
            'notion': (250, 600),          # Notion API
            'weather': (150, 400),         # Weather API
            'sqlite': (50, 150),           # SQLite queries
        }
        
        # Success rates for different tool types (when appropriate)
        self.success_rates = {
            'filesystem': 0.95,   # Usually works
            'search': 0.90,       # Good success rate
            'database': 0.85,     # Some query failures
            'github': 0.80,       # API rate limits
            'financial': 0.75,    # External service
            'notion': 0.85,       # Good API
            'weather': 0.88,      # Reliable service
            'sqlite': 0.92,       # Local database
        }
        
        # Error types by tool
        self.error_types = {
            'filesystem': ['permission_error', 'file_not_found'],
            'search': ['network_timeout', 'rate_limit'],
            'database': ['connection_error', 'query_error'],
            'github': ['rate_limit', 'authentication_error'],
            'financial': ['api_error', 'data_unavailable'],
            'notion': ['permission_error', 'page_not_found'],
            'weather': ['location_not_found', 'api_error'],
            'sqlite': ['table_not_found', 'syntax_error'],
        }
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ExecutionMetrics:
        """Execute a tool and return ExecutionMetrics.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            ExecutionMetrics object with execution results
        """
        # Determine tool type
        tool_type = self._get_tool_type(tool_name)
        
        # Get execution time range
        min_time, max_time = self.execution_times.get(tool_type, (100, 500))
        execution_time = random.uniform(min_time, max_time)
        
        # Simulate execution delay
        await asyncio.sleep(execution_time / 1000.0)
        
        # Determine success based on tool type and query appropriateness
        base_success_rate = self.success_rates.get(tool_type, 0.8)
        
        # Adjust success rate based on query appropriateness
        if self._is_appropriate_tool(tool_name, parameters):
            success_rate = min(base_success_rate * 1.2, 0.95)
        else:
            success_rate = base_success_rate * 0.5
        
        # Determine if execution succeeds
        success = random.random() < success_rate
        
        # Create ExecutionMetrics
        if success:
            return ExecutionMetrics(
                tool_id=tool_name,
                success=True,
                partial_success=False,
                completion_percentage=1.0,
                execution_time_ms=execution_time,
                error_type=None,
                retry_count=0,
                resource_usage={
                    'memory_mb': random.uniform(10, 100),
                    'cpu_percent': random.uniform(5, 30)
                },
                result_quality=random.uniform(0.7, 1.0)
            )
        else:
            # Determine if partial success
            partial_success = random.random() < 0.3
            
            # Select error type
            error_types = self.error_types.get(tool_type, ['unknown'])
            error_type = random.choice(error_types)
            
            return ExecutionMetrics(
                tool_id=tool_name,
                success=False,
                partial_success=partial_success,
                completion_percentage=random.uniform(0.1, 0.5) if partial_success else 0.0,
                execution_time_ms=execution_time,
                error_type=error_type,
                retry_count=random.randint(0, 2),
                resource_usage={
                    'memory_mb': random.uniform(5, 50),
                    'cpu_percent': random.uniform(2, 15)
                },
                result_quality=0.0
            )
    
    async def execute_tools(self, tool_names: List[str], query: str) -> List[ExecutionMetrics]:
        """Execute multiple tools and return ExecutionMetrics list.
        
        Args:
            tool_names: List of tool names to execute
            query: The user query (for context)
            
        Returns:
            List of ExecutionMetrics objects
        """
        metrics = []
        
        for tool_name in tool_names:
            # Create mock parameters based on query
            parameters = {'query': query, 'context': 'evaluation'}
            
            # Execute tool
            metric = await self.execute_tool(tool_name, parameters)
            metrics.append(metric)
            
            logger.debug(f"Executed {tool_name}: success={metric.success}, "
                        f"time={metric.execution_time_ms:.0f}ms")
        
        return metrics
    
    def _get_tool_type(self, tool_name: str) -> str:
        """Extract tool type from tool name."""
        tool_name_lower = tool_name.lower()
        
        for tool_type in self.execution_times.keys():
            if tool_type in tool_name_lower:
                return tool_type
        
        # Default type
        return 'database'
    
    def _is_appropriate_tool(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """Check if tool is appropriate for the query.
        
        This is a simplified heuristic - in reality would use intent matching.
        """
        tool_type = self._get_tool_type(tool_name)
        query = str(parameters.get('query', '')).lower()
        
        # Simple keyword matching
        appropriate_keywords = {
            'filesystem': ['file', 'directory', 'folder', 'path', 'read', 'write'],
            'search': ['search', 'find', 'web', 'internet', 'browse'],
            'database': ['database', 'query', 'sql', 'table', 'data'],
            'github': ['github', 'repository', 'code', 'commit', 'pull'],
            'financial': ['stock', 'price', 'finance', 'market', 'crypto'],
            'notion': ['notion', 'page', 'workspace', 'note'],
            'weather': ['weather', 'temperature', 'forecast', 'climate'],
            'sqlite': ['sqlite', 'database', 'query', 'table'],
        }
        
        keywords = appropriate_keywords.get(tool_type, [])
        return any(keyword in query for keyword in keywords)


def create_mock_execution_metrics(tool_names: List[str], success_rate: float = 0.8) -> List[ExecutionMetrics]:
    """Create mock ExecutionMetrics for testing.
    
    Args:
        tool_names: List of tool names
        success_rate: Probability of success for each tool
        
    Returns:
        List of ExecutionMetrics objects
    """
    metrics = []
    
    for tool_name in tool_names:
        success = random.random() < success_rate
        
        metrics.append(ExecutionMetrics(
            tool_id=tool_name,
            success=success,
            partial_success=False if success else random.random() < 0.3,
            completion_percentage=1.0 if success else random.uniform(0, 0.5),
            execution_time_ms=random.uniform(50, 500),
            error_type=None if success else random.choice(['timeout', 'permission', 'not_found']),
            retry_count=0 if success else random.randint(0, 2),
            resource_usage={
                'memory_mb': random.uniform(10, 100),
                'cpu_percent': random.uniform(5, 30)
            },
            result_quality=random.uniform(0.7, 1.0) if success else 0.0
        ))
    
    return metrics


# Test the wrapper
if __name__ == "__main__":
    import asyncio
    
    async def test():
        wrapper = ToolExecutionWrapper()
        
        # Test single tool execution
        print("Testing single tool execution:")
        metric = await wrapper.execute_tool("filesystem.list_directory", {"query": "list files"})
        print(f"  Tool: {metric.tool_id}")
        print(f"  Success: {metric.success}")
        print(f"  Time: {metric.execution_time_ms:.0f}ms")
        print(f"  Error: {metric.error_type}")
        
        # Test multiple tools
        print("\nTesting multiple tool execution:")
        tools = ["filesystem.search_files", "github.search_code", "database.query"]
        metrics = await wrapper.execute_tools(tools, "search for Python files")
        
        for metric in metrics:
            print(f"  {metric.tool_id}: {'✓' if metric.success else '✗'} ({metric.execution_time_ms:.0f}ms)")
        
        # Test with inappropriate tools
        print("\nTesting with inappropriate tools:")
        metric = await wrapper.execute_tool("weather.get_forecast", {"query": "find Python files"})
        print(f"  Weather tool for file search: {'✓' if metric.success else '✗'}")
        
        # Calculate rewards
        from src.learning.reward_calculator import RewardCalculator
        import json
        
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        calc = RewardCalculator(config)
        
        # Test reward calculation
        print("\nTesting reward calculation:")
        
        # Successful execution
        success_metrics = create_mock_execution_metrics(["filesystem.list", "search.web"], success_rate=1.0)
        success_reward, _ = calc.calculate_reward(success_metrics, {'intent': 'search'})
        print(f"  Success reward: {success_reward:.2f}")
        
        # Failed execution
        fail_metrics = create_mock_execution_metrics(["filesystem.list", "search.web"], success_rate=0.0)
        fail_reward, _ = calc.calculate_reward(fail_metrics, {'intent': 'search'})
        print(f"  Failure reward: {fail_reward:.2f}")
        
        print(f"  Difference: {success_reward - fail_reward:.2f}x")
    
    asyncio.run(test())