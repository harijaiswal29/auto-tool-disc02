"""
Demo script showing retry logic, exponential backoff, and circuit breakers in action.
"""

import asyncio
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.utils.logger import get_logger
from src.core.mcp_integration import MCPIntegration
from src.core.tool_registry import ToolRegistry

logger = get_logger(__name__)


class UnreliableServer:
    """Simulates an unreliable server for testing retry logic."""
    
    def __init__(self, failure_rate=0.7):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Simulate tool call that might fail."""
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Randomly fail based on failure rate
        if random.random() < self.failure_rate:
            logger.warning(f"[UNRELIABLE] Call {self.call_count} failed!")
            raise Exception("Server temporarily unavailable")
        
        logger.info(f"[UNRELIABLE] Call {self.call_count} succeeded!")
        return {"success": True, "result": f"Result from call {self.call_count}"}


async def demo_retry_logic():
    """Demonstrate retry logic with exponential backoff."""
    logger.info("=" * 60)
    logger.info("RETRY LOGIC DEMONSTRATION")
    logger.info("=" * 60)
    
    # Initialize components
    registry = ToolRegistry("data/demo_retry.db")
    await registry.initialize()
    
    integration = MCPIntegration(registry=registry)
    await integration.initialize()
    
    try:
        # Register a test tool
        tool_info = {
            'id': 'demo.unreliable_tool',
            'name': 'Unreliable Demo Tool',
            'server_type': 'demo',
            'endpoint': 'demo://localhost',
            'description': 'A tool that fails often to demonstrate retry logic',
            'capabilities': {'operations': ['test']},
            'input_schema': {}
        }
        registry.register_tool(tool_info)
        
        logger.info("\n1. Demonstrating retry with eventual success:")
        logger.info("-" * 40)
        
        # Execute tool multiple times to see retry behavior
        for i in range(3):
            logger.info(f"\n[ATTEMPT {i+1}] Executing unreliable tool...")
            result = await integration.execute_tool(
                'demo.unreliable_tool',
                {'operation': 'test'}
            )
            
            if result.get('success'):
                logger.info(f"✓ Success: {result}")
            else:
                logger.error(f"✗ Failed: {result}")
            
            # Show retry statistics
            retry_stats = integration.get_retry_statistics()
            logger.info(f"Retry Statistics: {retry_stats}")
            
            await asyncio.sleep(1)
        
        # Check circuit breaker status
        logger.info("\n2. Circuit Breaker Status:")
        logger.info("-" * 40)
        
        server_status = integration.get_server_status()
        for server_id, status in server_status.items():
            logger.info(f"Server: {server_id}")
            logger.info(f"  Status: {status.get('status')}")
            logger.info(f"  Circuit Breaker: {status.get('circuit_breaker_state', 'N/A')}")
            if status.get('circuit_breaker_stats'):
                stats = status['circuit_breaker_stats']
                logger.info(f"  Total Requests: {stats.get('total_requests', 0)}")
                logger.info(f"  Failed Requests: {stats.get('failed_requests', 0)}")
                logger.info(f"  Rejected Requests: {stats.get('rejected_requests', 0)}")
        
        # Show tool performance metrics
        logger.info("\n3. Tool Performance Metrics:")
        logger.info("-" * 40)
        
        perf_metrics = registry.get_tool_performance('demo.unreliable_tool')
        logger.info(f"Performance Score: {perf_metrics.get('overall_score', 0):.2%}")
        logger.info(f"Total Uses: {perf_metrics.get('total_uses', 0)}")
        logger.info(f"Success Rate: {perf_metrics.get('total_successes', 0) / max(perf_metrics.get('total_uses', 1), 1):.2%}")
        
        # Show failure metrics
        logger.info("\n4. Failure Analysis:")
        logger.info("-" * 40)
        
        failure_metrics = registry.get_failure_metrics('demo.unreliable_tool')
        logger.info(f"Failure Count: {failure_metrics.get('failure_count', 0)}")
        logger.info(f"Consecutive Failures: {failure_metrics.get('consecutive_failures', 0)}")
        logger.info(f"Circuit Breaker State: {failure_metrics.get('circuit_breaker_state', 'closed')}")
        
        if failure_metrics.get('retry_stats'):
            retry_stats = failure_metrics['retry_stats']
            logger.info(f"Retry Attempts (24h): {retry_stats.get('total_retries_24h', 0)}")
            logger.info(f"Average Retry Delay: {retry_stats.get('avg_delay_ms', 0):.0f}ms")
        
        # Get system health report
        logger.info("\n5. System Health Report:")
        logger.info("-" * 40)
        
        health_report = registry.get_health_report()
        logger.info(f"Health Status: {health_report.get('health_status', 'unknown')}")
        
        if health_report.get('overall_stats'):
            stats = health_report['overall_stats']
            logger.info(f"Total Tools: {stats.get('total_tools', 0)}")
            logger.info(f"Total Uses: {stats.get('total_uses', 0)}")
            logger.info(f"Average Performance: {stats.get('avg_performance', 0):.2%}")
        
        if health_report.get('open_circuit_breakers'):
            logger.warning(f"Open Circuit Breakers: {len(health_report['open_circuit_breakers'])}")
            for cb in health_report['open_circuit_breakers']:
                logger.warning(f"  - {cb['id']}: {cb.get('consecutive_failures', 0)} failures")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await integration.shutdown()
        await registry.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)


async def demo_connection_pool():
    """Demonstrate connection pooling."""
    logger.info("\n" + "=" * 60)
    logger.info("CONNECTION POOL DEMONSTRATION")
    logger.info("=" * 60)
    
    from src.core.connection_pool import ConnectionPool
    
    pool = ConnectionPool({
        'max_connections': 3,
        'connection_timeout': 2.0,
        'idle_timeout': 10.0
    })
    
    await pool.start()
    
    try:
        # Create mock connections
        async def create_mock_connection():
            logger.info("[POOL] Creating new connection...")
            await asyncio.sleep(0.5)  # Simulate connection time
            return {"id": f"conn_{random.randint(1000, 9999)}"}
        
        # Test connection reuse
        logger.info("\n1. Testing connection reuse:")
        logger.info("-" * 40)
        
        # First acquisition
        async with pool.acquire_connection('server1', 'type1', create_mock_connection) as conn:
            logger.info(f"Acquired connection: {conn.client}")
        
        # Second acquisition - should reuse
        async with pool.acquire_connection('server1', 'type1', create_mock_connection) as conn:
            logger.info(f"Acquired connection (reused): {conn.client}")
        
        # Show statistics
        stats = pool.get_statistics()
        logger.info(f"\nPool Statistics:")
        logger.info(f"  Connections created: {stats['connections_created']}")
        logger.info(f"  Connections reused: {stats['connections_reused']}")
        logger.info(f"  Total connections: {stats['total_connections']}")
        
    finally:
        await pool.stop()


async def main():
    """Run all demos."""
    await demo_retry_logic()
    await demo_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())