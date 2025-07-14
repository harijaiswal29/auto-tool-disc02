"""
Unit tests for MCP Integration module.

Tests the central MCP integration functionality including:
- Server lifecycle management
- Tool discovery and registration
- Tool execution with retry logic
- Circuit breaker integration
- Connection pooling
- Intent-based tool finding
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry
from src.core.connection_pool import ConnectionPool
from src.utils.retry import RetryableError, NonRetryableError, CircuitBreaker


class TestMCPIntegration:
    """Test cases for MCPIntegration class."""
    
    @pytest.fixture
    async def mock_registry(self):
        """Create a mock tool registry."""
        registry = Mock(spec=ToolRegistry)
        registry.initialize = AsyncMock()
        registry.close = AsyncMock()
        registry.list_tools = Mock(return_value=[])
        registry.get_tool = Mock(return_value=None)
        registry.record_usage = Mock()
        return registry
    
    @pytest.fixture
    async def mock_connection_pool(self):
        """Create a mock connection pool."""
        pool = Mock(spec=ConnectionPool)
        pool.start = AsyncMock()
        pool.stop = AsyncMock()
        pool.get_statistics = Mock(return_value={})
        return pool
    
    @pytest.fixture
    def default_config(self):
        """Default configuration for testing."""
        return {
            'database': {
                'tool_registry': 'test_registry.db'
            },
            'retry_policies': {
                'default': {
                    'type': 'exponential_backoff',
                    'max_attempts': 3,
                    'base_delay': 0.1,
                    'max_delay': 1.0,
                    'jitter_factor': 0.1
                },
                'mcp_servers': {
                    'sqlite': {
                        'type': 'fixed_delay',
                        'max_attempts': 2,
                        'delay': 0.5
                    }
                },
                'no_retry_errors': ['NonRetryableError', 'AuthenticationError']
            },
            'circuit_breaker': {
                'default': {
                    'failure_threshold': 3,
                    'recovery_timeout': 5.0,
                    'half_open_test_requests': 2
                }
            },
            'connection_pool': {
                'max_connections': 10,
                'idle_timeout': 300
            }
        }
    
    @pytest.fixture
    async def mcp_integration(self, mock_registry, mock_connection_pool, default_config):
        """Create MCPIntegration instance with mocks."""
        with patch('src.core.mcp_integration.ToolRegistry', return_value=mock_registry):
            with patch('src.core.mcp_integration.ConnectionPool', return_value=mock_connection_pool):
                integration = MCPIntegration(config=default_config, registry=mock_registry)
                integration.connection_pool = mock_connection_pool
                yield integration
    
    @pytest.mark.asyncio
    async def test_initialization(self, default_config):
        """Test MCPIntegration initialization."""
        with patch('src.core.mcp_integration.ToolRegistry') as mock_registry_class:
            with patch('src.core.mcp_integration.ConnectionPool') as mock_pool_class:
                integration = MCPIntegration(config=default_config)
                
                assert integration.config == default_config
                assert integration.servers == {}
                assert integration.active_connections == {}
                assert integration.retry_manager is not None
                assert integration.connection_pool is not None
    
    @pytest.mark.asyncio
    async def test_load_default_config(self, mcp_integration):
        """Test loading default configuration."""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
                'database': {'tool_registry': 'default.db'}
            })
            
            config = mcp_integration._load_default_config()
            assert config['database']['tool_registry'] == 'default.db'
    
    @pytest.mark.asyncio
    async def test_add_sqlite_server_success(self, mcp_integration):
        """Test successfully adding a SQLite server."""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.register_tools_to_registry = Mock()
        mock_client.use_mock = False
        
        with patch('src.core.mcp_integration.SQLiteMCPClient', return_value=mock_client):
            result = await mcp_integration.add_sqlite_server(
                db_path='/test/db.sqlite',
                server_id='test_sqlite'
            )
            
            assert result is True
            assert 'test_sqlite' in mcp_integration.servers
            assert mcp_integration.servers['test_sqlite']['type'] == 'sqlite'
            assert mcp_integration.servers['test_sqlite']['status'] == 'active'
            assert 'test_sqlite' in mcp_integration.active_connections
    
    @pytest.mark.asyncio
    async def test_add_sqlite_server_failure(self, mcp_integration):
        """Test failing to add a SQLite server."""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=False)
        
        with patch('src.core.mcp_integration.SQLiteMCPClient', return_value=mock_client):
            result = await mcp_integration.add_sqlite_server(
                db_path='/test/db.sqlite',
                server_id='test_sqlite'
            )
            
            assert result is False
            assert 'test_sqlite' not in mcp_integration.servers
    
    @pytest.mark.asyncio
    async def test_add_search_server_with_mock(self, mcp_integration):
        """Test adding a search server with mock mode."""
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.register_tools_to_registry = Mock()
        mock_client.use_mock = True
        
        with patch('src.core.mcp_integration.SearchMCPClient', return_value=mock_client):
            result = await mcp_integration.add_search_server(
                config={'api_key': 'test'},
                server_id='test_search',
                use_mock=True
            )
            
            assert result is True
            assert mcp_integration.servers['test_search']['is_mock'] is True
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mcp_integration):
        """Test successful tool execution."""
        # Setup mock registry to return a tool
        mcp_integration.registry.get_tool.return_value = {
            'id': 'sqlite.query',
            'server_type': 'sqlite'
        }
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value={
            'success': True,
            'result': 'test_result'
        })
        
        # Add server and connection
        mcp_integration.servers['test_sqlite'] = {
            'type': 'sqlite',
            'status': 'active',
            'client': mock_client
        }
        mcp_integration.active_connections['test_sqlite'] = mock_client
        
        # Execute tool
        result = await mcp_integration.execute_tool(
            'sqlite.query',
            {'sql': 'SELECT 1'}
        )
        
        assert result['success'] is True
        assert result['result'] == 'test_result'
        mcp_integration.registry.record_usage.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mcp_integration):
        """Test executing a non-existent tool."""
        mcp_integration.registry.get_tool.return_value = None
        
        result = await mcp_integration.execute_tool(
            'nonexistent.tool',
            {}
        )
        
        assert 'error' in result
        assert 'Tool not found' in result['error']
    
    @pytest.mark.asyncio
    async def test_execute_tool_no_active_server(self, mcp_integration):
        """Test executing a tool when no server is active."""
        mcp_integration.registry.get_tool.return_value = {
            'id': 'sqlite.query',
            'server_type': 'sqlite'
        }
        
        result = await mcp_integration.execute_tool(
            'sqlite.query',
            {'sql': 'SELECT 1'}
        )
        
        assert 'error' in result
        assert 'No active server' in result['error']
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_retry(self, mcp_integration):
        """Test tool execution with retry on failure."""
        mcp_integration.registry.get_tool.return_value = {
            'id': 'sqlite.query',
            'server_type': 'sqlite'
        }
        
        # Mock client that fails once then succeeds
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(side_effect=[
            {'success': False, 'error': 'Temporary error'},
            {'success': True, 'result': 'success_after_retry'}
        ])
        
        mcp_integration.servers['test_sqlite'] = {
            'type': 'sqlite',
            'status': 'active',
            'client': mock_client
        }
        mcp_integration.active_connections['test_sqlite'] = mock_client
        
        result = await mcp_integration.execute_tool(
            'sqlite.query',
            {'sql': 'SELECT 1'}
        )
        
        assert result['success'] is True
        assert result['result'] == 'success_after_retry'
        assert mock_client.call_tool.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_tool_non_retryable_error(self, mcp_integration):
        """Test tool execution with non-retryable error."""
        mcp_integration.registry.get_tool.return_value = {
            'id': 'sqlite.query',
            'server_type': 'sqlite'
        }
        
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value={
            'success': False,
            'error': 'AuthenticationError: Invalid credentials'
        })
        
        mcp_integration.servers['test_sqlite'] = {
            'type': 'sqlite',
            'status': 'active',
            'client': mock_client
        }
        mcp_integration.active_connections['test_sqlite'] = mock_client
        
        result = await mcp_integration.execute_tool(
            'sqlite.query',
            {'sql': 'SELECT 1'}
        )
        
        assert 'error' in result
        assert 'AuthenticationError' in result['error']
        # Should only be called once (no retry)
        assert mock_client.call_tool.call_count == 1
    
    @pytest.mark.asyncio
    async def test_discover_all_tools(self, mcp_integration):
        """Test discovering tools from all servers."""
        mcp_integration.servers = {
            'sqlite_1': {'type': 'sqlite', 'status': 'active'},
            'search_1': {'type': 'search', 'status': 'active'},
            'inactive_1': {'type': 'sqlite', 'status': 'inactive'}
        }
        
        mcp_integration.registry.list_tools.side_effect = [
            [{'id': 'sqlite.query', 'name': 'Query'}],
            [{'id': 'search.web', 'name': 'Web Search'}]
        ]
        
        tools = await mcp_integration.discover_all_tools()
        
        assert len(tools) == 2
        assert any(t['id'] == 'sqlite.query' for t in tools)
        assert any(t['id'] == 'search.web' for t in tools)
    
    @pytest.mark.asyncio
    async def test_shutdown_server(self, mcp_integration):
        """Test shutting down a specific server."""
        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock()
        
        mcp_integration.active_connections['test_server'] = mock_client
        mcp_integration.servers['test_server'] = {
            'status': 'active'
        }
        
        result = await mcp_integration.shutdown_server('test_server')
        
        assert result is True
        assert 'test_server' not in mcp_integration.active_connections
        assert mcp_integration.servers['test_server']['status'] == 'inactive'
        mock_client.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_all(self, mcp_integration):
        """Test shutting down all servers."""
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        
        mcp_integration.active_connections = {
            'server1': mock_client1,
            'server2': mock_client2
        }
        mcp_integration.servers = {
            'server1': {'status': 'active'},
            'server2': {'status': 'active'}
        }
        
        await mcp_integration.shutdown_all()
        
        assert len(mcp_integration.active_connections) == 0
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_tools_by_intent(self, mcp_integration):
        """Test finding tools by intent type."""
        mcp_integration.registry.list_tools.return_value = [
            {
                'id': 'sqlite.query',
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'query', 'category': 'data'}
                    ]
                })
            },
            {
                'id': 'filesystem.create',
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'create_file', 'category': 'file'}
                    ]
                })
            }
        ]
        
        # Test query intent
        query_tools = await mcp_integration.find_tools_by_intent('query.search')
        assert len(query_tools) == 1
        assert query_tools[0]['id'] == 'sqlite.query'
        
        # Test create intent
        create_tools = await mcp_integration.find_tools_by_intent('action.create')
        assert len(create_tools) == 1
        assert create_tools[0]['id'] == 'filesystem.create'
    
    @pytest.mark.asyncio
    async def test_get_tools_by_capabilities(self, mcp_integration):
        """Test getting tools by specific capabilities."""
        mcp_integration.registry.list_tools.return_value = [
            {
                'id': 'tool1',
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'read_file'},
                        {'name': 'write_file'}
                    ]
                })
            },
            {
                'id': 'tool2',
                'capabilities': json.dumps({
                    'operations': [
                        {'name': 'query_database'}
                    ]
                })
            }
        ]
        
        # Test finding tools with 'read' capability
        read_tools = await mcp_integration.get_tools_by_capabilities(['read'])
        assert len(read_tools) == 1
        assert read_tools[0]['id'] == 'tool1'
        
        # Test finding tools with multiple capabilities
        file_tools = await mcp_integration.get_tools_by_capabilities(['read', 'write'])
        assert len(file_tools) == 1
        assert file_tools[0]['id'] == 'tool1'
    
    @pytest.mark.asyncio
    async def test_execute_tool_by_intent(self, mcp_integration):
        """Test executing the best tool for a given intent."""
        mcp_integration.registry.list_tools.return_value = [
            {
                'id': 'tool1',
                'performance_score': 0.7,
                'capabilities': json.dumps({
                    'operations': [{'name': 'search'}]
                })
            },
            {
                'id': 'tool2',
                'performance_score': 0.9,
                'capabilities': json.dumps({
                    'operations': [{'name': 'query'}]
                })
            }
        ]
        
        # Mock the execute_tool method
        mcp_integration.execute_tool = AsyncMock(return_value={
            'success': True,
            'result': 'best_tool_result'
        })
        
        result = await mcp_integration.execute_tool_by_intent(
            'query.search',
            {'param': 'value'}
        )
        
        assert result['success'] is True
        assert result['result'] == 'best_tool_result'
        # Should select tool2 with higher performance score
        mcp_integration.execute_tool.assert_called_with('tool2', {'param': 'value'})
    
    @pytest.mark.asyncio
    async def test_get_server_status(self, mcp_integration):
        """Test getting server status with circuit breaker state."""
        # Setup servers
        mcp_integration.servers = {
            'server1': {'type': 'sqlite', 'status': 'active'},
            'server2': {'type': 'search', 'status': 'inactive'}
        }
        
        # Mock circuit breaker
        mock_cb = Mock()
        mock_cb.state.value = 'closed'
        mock_cb.statistics = {'failures': 0, 'successes': 100}
        mcp_integration.retry_manager.circuit_breakers = {
            'server1': mock_cb
        }
        
        # Mock registry
        mcp_integration.registry.list_tools.side_effect = [
            [{'id': 'tool1'}, {'id': 'tool2'}],  # 2 tools for sqlite
            []  # 0 tools for search
        ]
        
        status = mcp_integration.get_server_status()
        
        assert status['server1']['type'] == 'sqlite'
        assert status['server1']['status'] == 'active'
        assert status['server1']['tools_count'] == 2
        assert status['server1']['circuit_breaker_state'] == 'closed'
        assert status['server1']['circuit_breaker_stats']['failures'] == 0
        
        assert status['server2']['type'] == 'search'
        assert status['server2']['status'] == 'inactive'
        assert status['server2']['circuit_breaker_state'] is None
    
    @pytest.mark.asyncio
    async def test_get_retry_statistics(self, mcp_integration):
        """Test getting retry statistics."""
        mock_stats = {
            'total_retries': 10,
            'successful_retries': 8,
            'failed_retries': 2
        }
        mcp_integration.retry_manager.get_statistics = Mock(return_value=mock_stats)
        
        stats = mcp_integration.get_retry_statistics()
        
        assert stats == mock_stats
    
    @pytest.mark.asyncio
    async def test_get_pool_statistics(self, mcp_integration):
        """Test getting connection pool statistics."""
        mock_stats = {
            'active_connections': 5,
            'idle_connections': 3,
            'total_created': 15
        }
        mcp_integration.connection_pool.get_statistics.return_value = mock_stats
        
        stats = mcp_integration.get_pool_statistics()
        
        assert stats == mock_stats
    
    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, mcp_integration):
        """Test initialization and shutdown lifecycle."""
        # Test initialize
        await mcp_integration.initialize()
        
        mcp_integration.registry.initialize.assert_called_once()
        mcp_integration.connection_pool.start.assert_called_once()
        
        # Test shutdown
        await mcp_integration.shutdown()
        
        mcp_integration.connection_pool.stop.assert_called_once()
        mcp_integration.registry.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_is_non_retryable_error(self, mcp_integration):
        """Test checking if an error is non-retryable."""
        # Test configured non-retryable errors
        assert mcp_integration._is_non_retryable_error('NonRetryableError: Something failed')
        assert mcp_integration._is_non_retryable_error('AuthenticationError occurred')
        
        # Test retryable errors
        assert not mcp_integration._is_non_retryable_error('Network timeout')
        assert not mcp_integration._is_non_retryable_error('Connection refused')
    
    @pytest.mark.asyncio
    async def test_get_retry_policy_for_server(self, mcp_integration):
        """Test getting server-specific retry policy."""
        # Get policy for configured server (sqlite)
        sqlite_policy = mcp_integration._get_retry_policy_for_server('sqlite')
        assert sqlite_policy is not None
        
        # Get policy for unconfigured server (should use default)
        default_policy = mcp_integration._get_retry_policy_for_server('unknown')
        assert default_policy is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])