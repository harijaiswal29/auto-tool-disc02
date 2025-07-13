"""
Connection pool manager for MCP server connections.
Manages connection lifecycle, health checking, and reuse.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set
from collections import defaultdict
from contextlib import asynccontextmanager

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MCPConnection:
    """Represents a single MCP connection."""
    
    def __init__(self, server_id: str, client: Any, server_type: str):
        self.server_id = server_id
        self.client = client
        self.server_type = server_type
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.usage_count = 0
        self.is_healthy = True
        self.last_health_check = None
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire the connection for use."""
        await self._lock.acquire()
        self.last_used = datetime.now()
        self.usage_count += 1
    
    def release(self):
        """Release the connection after use."""
        self._lock.release()
    
    def is_idle(self, idle_timeout: float) -> bool:
        """Check if connection has been idle for longer than timeout."""
        idle_time = (datetime.now() - self.last_used).total_seconds()
        return idle_time > idle_timeout
    
    async def health_check(self) -> bool:
        """Perform health check on the connection."""
        try:
            # Try to call a simple tool or check connection status
            if hasattr(self.client, 'is_connected'):
                self.is_healthy = await self.client.is_connected()
            else:
                # Fallback: assume healthy if no check method available
                self.is_healthy = True
            
            self.last_health_check = datetime.now()
            return self.is_healthy
        except Exception as e:
            logger.error(f"Health check failed for {self.server_id}: {e}")
            self.is_healthy = False
            return False


class ConnectionPool:
    """
    Connection pool for managing MCP server connections.
    
    Features:
    - Connection reuse
    - Health checking
    - Automatic cleanup of idle connections
    - Connection limits per server type
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize connection pool.
        
        Args:
            config: Pool configuration including:
                - max_connections: Maximum total connections
                - connection_timeout: Timeout for acquiring connection
                - idle_timeout: Time before idle connections are closed
                - health_check_interval: Interval between health checks
        """
        self.config = config or {}
        self.max_connections = self.config.get('max_connections', 10)
        self.connection_timeout = self.config.get('connection_timeout', 5.0)
        self.idle_timeout = self.config.get('idle_timeout', 300.0)
        self.health_check_interval = self.config.get('health_check_interval', 60.0)
        
        # Connection storage
        self.connections: Dict[str, MCPConnection] = {}
        self.available_connections: Dict[str, Set[str]] = defaultdict(set)
        self.in_use_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connections_closed': 0,
            'failed_acquisitions': 0,
            'health_check_failures': 0
        }
        
        # Background tasks
        self._health_check_task = None
        self._cleanup_task = None
        self._running = False
        
        logger.info(f"[POOL] Connection pool initialized with max_connections={self.max_connections}")
    
    async def start(self):
        """Start background tasks for health checking and cleanup."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("[POOL] Background tasks started")
    
    async def stop(self):
        """Stop background tasks and close all connections."""
        logger.info("[POOL] Stopping connection pool...")
        self._running = False
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Close all connections
        for conn_id in list(self.connections.keys()):
            await self._close_connection(conn_id)
        
        logger.info("[POOL] Connection pool stopped")
    
    @asynccontextmanager
    async def acquire_connection(self, server_id: str, server_type: str, 
                               create_func: Optional[Any] = None):
        """
        Acquire a connection from the pool.
        
        Args:
            server_id: Server identifier
            server_type: Type of server (sqlite, postgres, etc.)
            create_func: Async function to create new connection if needed
            
        Yields:
            MCPConnection object
        """
        connection = None
        start_time = time.time()
        
        try:
            # Try to get an available connection
            connection = await self._get_available_connection(server_id, server_type)
            
            if not connection and create_func:
                # Create new connection if under limit
                if len(self.connections) < self.max_connections:
                    connection = await self._create_connection(
                        server_id, server_type, create_func
                    )
                else:
                    # Wait for a connection to become available
                    connection = await self._wait_for_connection(
                        server_id, server_type, self.connection_timeout
                    )
            
            if not connection:
                self.stats['failed_acquisitions'] += 1
                raise TimeoutError(
                    f"Could not acquire connection for {server_id} within {self.connection_timeout}s"
                )
            
            # Mark connection as in use
            await connection.acquire()
            self.available_connections[server_type].discard(connection.server_id)
            self.in_use_connections[server_type].add(connection.server_id)
            
            acquisition_time = time.time() - start_time
            logger.debug(
                f"[POOL] Connection acquired for {server_id} in {acquisition_time:.2f}s"
            )
            
            yield connection
            
        finally:
            # Return connection to pool
            if connection:
                connection.release()
                self.in_use_connections[server_type].discard(connection.server_id)
                self.available_connections[server_type].add(connection.server_id)
    
    async def _get_available_connection(self, server_id: str, 
                                      server_type: str) -> Optional[MCPConnection]:
        """Get an available connection from the pool."""
        # First try to find exact match
        if server_id in self.available_connections[server_type]:
            if server_id in self.connections:
                conn = self.connections[server_id]
                if conn.is_healthy:
                    self.stats['connections_reused'] += 1
                    return conn
        
        # Try any available connection of the same type
        for conn_id in list(self.available_connections[server_type]):
            if conn_id in self.connections:
                conn = self.connections[conn_id]
                if conn.is_healthy and conn.server_type == server_type:
                    self.stats['connections_reused'] += 1
                    return conn
        
        return None
    
    async def _create_connection(self, server_id: str, server_type: str, 
                               create_func: Any) -> MCPConnection:
        """Create a new connection."""
        logger.info(f"[POOL] Creating new connection for {server_id}")
        
        try:
            client = await create_func()
            connection = MCPConnection(server_id, client, server_type)
            
            self.connections[server_id] = connection
            self.available_connections[server_type].add(server_id)
            self.stats['connections_created'] += 1
            
            return connection
        except Exception as e:
            logger.error(f"[POOL] Failed to create connection for {server_id}: {e}")
            raise
    
    async def _wait_for_connection(self, server_id: str, server_type: str, 
                                 timeout: float) -> Optional[MCPConnection]:
        """Wait for a connection to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if any connection became available
            conn = await self._get_available_connection(server_id, server_type)
            if conn:
                return conn
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        return None
    
    async def _close_connection(self, conn_id: str):
        """Close and remove a connection from the pool."""
        if conn_id not in self.connections:
            return
        
        conn = self.connections[conn_id]
        
        try:
            if hasattr(conn.client, 'disconnect'):
                await conn.client.disconnect()
            elif hasattr(conn.client, 'close'):
                await conn.client.close()
        except Exception as e:
            logger.error(f"[POOL] Error closing connection {conn_id}: {e}")
        
        # Remove from all sets
        self.available_connections[conn.server_type].discard(conn_id)
        self.in_use_connections[conn.server_type].discard(conn_id)
        del self.connections[conn_id]
        
        self.stats['connections_closed'] += 1
        logger.info(f"[POOL] Connection {conn_id} closed")
    
    async def _health_check_loop(self):
        """Background task to perform health checks."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check health of all available connections
                for conn_id in list(self.connections.keys()):
                    if conn_id in self.connections:
                        conn = self.connections[conn_id]
                        
                        # Only check available connections
                        if conn_id in self.available_connections[conn.server_type]:
                            is_healthy = await conn.health_check()
                            
                            if not is_healthy:
                                logger.warning(
                                    f"[POOL] Connection {conn_id} failed health check"
                                )
                                self.stats['health_check_failures'] += 1
                                await self._close_connection(conn_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[POOL] Error in health check loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task to clean up idle connections."""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Check every minute
                
                # Find and close idle connections
                for conn_id in list(self.connections.keys()):
                    if conn_id in self.connections:
                        conn = self.connections[conn_id]
                        
                        # Only cleanup available connections
                        if (conn_id in self.available_connections[conn.server_type] and 
                            conn.is_idle(self.idle_timeout)):
                            logger.info(f"[POOL] Closing idle connection {conn_id}")
                            await self._close_connection(conn_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[POOL] Error in cleanup loop: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        total_connections = len(self.connections)
        available_count = sum(len(conns) for conns in self.available_connections.values())
        in_use_count = sum(len(conns) for conns in self.in_use_connections.values())
        
        return {
            'total_connections': total_connections,
            'available_connections': available_count,
            'in_use_connections': in_use_count,
            'connections_by_type': {
                server_type: len(conns) 
                for server_type, conns in self.available_connections.items()
            },
            **self.stats
        }