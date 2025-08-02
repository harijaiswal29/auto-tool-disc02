"""
Test utilities for setting up tools and MCP integration for tests.

Provides helper functions for:
- Registering test tools with all required fields
- Setting up mock MCP servers
- Creating consistent test data
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_test_tools() -> List[Dict[str, Any]]:
    """Get a standard set of test tools with all required fields."""
    return [
        {
            'id': 'test_search_001',
            'name': 'Test Search Tool',
            'type': 'search',
            'server_type': 'stdio',
            'endpoint': 'search_mcp',
            'description': 'Test search tool for finding information',
            'capabilities': json.dumps({
                'operations': ['search', 'find', 'query'],
                'data_types': ['text', 'json']
            }),
            'input_schema': json.dumps({
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'limit': {'type': 'integer', 'default': 10}
                },
                'required': ['query']
            }),
            'performance_score': 0.9
        },
        {
            'id': 'test_db_001',
            'name': 'Test Database Tool',
            'type': 'database',
            'server_type': 'stdio',
            'endpoint': 'sqlite_mcp',
            'description': 'Test database tool for querying data',
            'capabilities': json.dumps({
                'operations': ['query', 'retrieve', 'analyze'],
                'data_types': ['sql', 'json']
            }),
            'input_schema': json.dumps({
                'type': 'object',
                'properties': {
                    'sql_query': {'type': 'string'},
                    'database': {'type': 'string', 'default': 'test.db'}
                },
                'required': ['sql_query']
            }),
            'performance_score': 0.85
        },
        {
            'id': 'test_fs_001',
            'name': 'Test FileSystem Tool',
            'type': 'filesystem',
            'server_type': 'stdio',
            'endpoint': 'filesystem_mcp',
            'description': 'Test filesystem tool for file operations',
            'capabilities': json.dumps({
                'operations': ['read', 'write', 'list'],
                'data_types': ['file', 'directory']
            }),
            'input_schema': json.dumps({
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'operation': {'type': 'string', 'enum': ['read', 'write', 'list']}
                },
                'required': ['path', 'operation']
            }),
            'performance_score': 0.95
        },
        {
            'id': 'test_analyze_001',
            'name': 'Test Analysis Tool',
            'type': 'analysis',
            'server_type': 'stdio',
            'endpoint': 'analysis_mcp',
            'description': 'Test analysis tool for data evaluation',
            'capabilities': json.dumps({
                'operations': ['analyze', 'examine', 'evaluate'],
                'data_types': ['text', 'json', 'metrics']
            }),
            'input_schema': json.dumps({
                'type': 'object',
                'properties': {
                    'data': {'type': 'object'},
                    'analysis_type': {'type': 'string', 'default': 'basic'}
                },
                'required': ['data']
            }),
            'performance_score': 0.8
        }
    ]


def register_test_tools(registry: ToolRegistry) -> bool:
    """
    Register standard test tools in the registry.
    
    Args:
        registry: Tool registry instance
        
    Returns:
        True if all tools registered successfully
    """
    test_tools = get_test_tools()
    
    for tool in test_tools:
        if not registry.register_tool(tool):
            logger.error(f"Failed to register tool: {tool['id']}")
            return False
        logger.info(f"Registered test tool: {tool['id']}")
    
    return True


async def setup_tool_relationships(registry: ToolRegistry):
    """Setup standard tool relationships for testing."""
    relationships = [
        ('test_search_001', 'test_analyze_001', 'complements', 0.8),
        ('test_db_001', 'test_analyze_001', 'complements', 0.9),
        ('test_search_001', 'test_fs_001', 'conflicts', 0.6),
        ('test_db_001', 'test_fs_001', 'requires', 0.7)
    ]
    
    for tool1, tool2, rel_type, strength in relationships:
        success = await registry.add_tool_relationship(tool1, tool2, rel_type, strength)
        if success:
            logger.info(f"Added relationship: {tool1} {rel_type} {tool2}")
        else:
            logger.warning(f"Failed to add relationship: {tool1} {rel_type} {tool2}")


async def mock_tool_execution(tool_id: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock tool execution with realistic responses.
    
    Args:
        tool_id: Tool identifier
        tool_input: Tool input parameters
        
    Returns:
        Mock execution result
    """
    # Simulate execution time
    await asyncio.sleep(0.05)
    
    if 'search' in tool_id:
        query = tool_input.get('query', '')
        return {
            'results': [
                {'title': f'Result 1 for {query}', 'url': 'http://example.com/1', 'relevance': 0.95},
                {'title': f'Result 2 for {query}', 'url': 'http://example.com/2', 'relevance': 0.87},
                {'title': f'Result 3 for {query}', 'url': 'http://example.com/3', 'relevance': 0.76}
            ],
            'total': 3,
            'query': query,
            'success': True
        }
    
    elif 'db' in tool_id or 'database' in tool_id:
        sql = tool_input.get('sql_query', 'SELECT * FROM test')
        return {
            'rows': [
                {'id': 1, 'name': 'Item 1', 'value': 100},
                {'id': 2, 'name': 'Item 2', 'value': 200},
                {'id': 3, 'name': 'Item 3', 'value': 300}
            ],
            'count': 3,
            'query': sql,
            'execution_time_ms': 25.5,
            'success': True
        }
    
    elif 'fs' in tool_id or 'filesystem' in tool_id:
        path = tool_input.get('path', '/test/file.txt')
        operation = tool_input.get('operation', 'read')
        
        if operation == 'read':
            return {
                'content': f'Content of {path}',
                'size': 1024,
                'path': path,
                'success': True
            }
        elif operation == 'list':
            return {
                'files': ['file1.txt', 'file2.py', 'config.json'],
                'directories': ['src', 'tests', 'docs'],
                'path': path,
                'success': True
            }
        else:
            return {
                'path': path,
                'operation': operation,
                'success': True
            }
    
    elif 'analyze' in tool_id:
        data = tool_input.get('data', {})
        return {
            'analysis': {
                'score': 0.85,
                'summary': 'Analysis complete',
                'metrics': {
                    'quality': 0.9,
                    'completeness': 0.8,
                    'accuracy': 0.85
                }
            },
            'input_size': len(str(data)),
            'success': True
        }
    
    else:
        return {
            'status': 'success',
            'data': 'Generic result',
            'tool_id': tool_id,
            'input': tool_input
        }


async def mock_tool_execution_with_errors(tool_id: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock tool execution with some failures for error testing.
    
    Args:
        tool_id: Tool identifier
        tool_input: Tool input parameters
        
    Returns:
        Mock execution result or raises exception
    """
    # Simulate execution time
    await asyncio.sleep(0.05)
    
    if 'search' in tool_id:
        raise ConnectionError("Search service unavailable")
    
    elif 'database' in tool_id:
        # Return empty result (not an error, but no data)
        return {
            'rows': [],
            'count': 0,
            'query': tool_input.get('sql_query', ''),
            'success': True
        }
    
    else:
        # Other tools succeed
        return await mock_tool_execution(tool_id, tool_input)


def create_test_config(temp_dir: str, enable_learning: bool = True) -> Dict[str, Any]:
    """
    Create a standard test configuration.
    
    Args:
        temp_dir: Temporary directory for test data
        enable_learning: Whether to enable Q-learning
        
    Returns:
        Test configuration dictionary
    """
    return {
        "orchestration": {
            "max_tools_per_query": 3,
            "tool_selection_strategy": "performance_weighted",
            "parallel_execution": True
        },
        "q_learning": {
            "enable_learning": enable_learning,
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 0.2,
            "model_path": f"{temp_dir}/q_model.pkl"
        },
        "result_cache": {
            "enabled": True,
            "max_size": 100,
            "ttl_seconds": 3600,
            "cache_successful_only": True,
            "consider_context": True,
            "storage_path": f"{temp_dir}/cache.db"
        },
        "cache_monitoring": {
            "enabled": True,
            "metrics_window_seconds": 300
        },
        "database": {
            "path": f"{temp_dir}/test.db",
            "tool_registry": f"{temp_dir}/registry.db",
            "learning_db": f"{temp_dir}/learning.db"
        },
        "intent_recognition": {
            "model": "all-MiniLM-L6-v2",
            "confidence_threshold": 0.7
        },
        "mcp": {
            "servers": {
                "search_mcp": {
                    "command": ["python", "src/tools/mock_search_mcp.py"],
                    "type": "stdio"
                },
                "sqlite_mcp": {
                    "command": ["python", "src/tools/mock_sqlite_mcp.py"],
                    "type": "stdio"
                },
                "filesystem_mcp": {
                    "command": ["python", "src/tools/mock_filesystem_mcp.py"],
                    "type": "stdio"
                },
                "analysis_mcp": {
                    "command": ["python", "src/tools/mock_mcp_servers.py"],
                    "type": "stdio"
                }
            }
        }
    }