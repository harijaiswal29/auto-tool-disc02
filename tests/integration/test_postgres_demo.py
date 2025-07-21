import asyncio
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.postgres_mcp import PostgresMCPClient
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def demonstrate_postgres_mcp():
    """Comprehensive demonstration of PostgreSQL MCP functionality."""
    
    print("=" * 80)
    print("PostgreSQL MCP Comprehensive Demonstration")
    print("=" * 80)
    
    # Test configurations
    real_connection = "postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"
    test_connection = "postgresql://testuser:testpass@localhost:5432/auto_tool_disc"
    
    # Part 1: Test with Real Database
    print("\n📊 PART 1: Testing with Real PostgreSQL Database")
    print("-" * 60)
    
    client = PostgresMCPClient(real_connection)
    connected = await client.connect(use_mock=False)
    
    if connected:
        print("✅ Connected to real PostgreSQL MCP server!")
        
        # Show discovered tools
        print(f"\n🔧 Discovered Tools: {len(client.tools)}")
        for tool in client.tools:
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        
        # Execute various queries
        queries = [
            ("Database Version", "SELECT version()"),
            ("List Tables", "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"),
            ("Tools Count", "SELECT COUNT(*) as count, type FROM tools GROUP BY type"),
            ("Recent Executions", "SELECT id, query, success, execution_time_ms FROM execution_history ORDER BY created_at DESC LIMIT 5"),
            ("Tool Relationships", "SELECT t1.name as tool1, relationship_type, t2.name as tool2 FROM tool_relationships tr JOIN tools t1 ON tr.tool1_id = t1.id JOIN tools t2 ON tr.tool2_id = t2.id")
        ]
        
        for title, query in queries:
            print(f"\n📋 {title}:")
            result = await client.execute_query(query)
            if result['success']:
                # Parse and display results
                if 'content' in result['result']:
                    data = json.loads(result['result']['content'][0]['text'])
                    print(json.dumps(data, indent=2))
                print(f"⏱️  Execution time: {result['execution_time']:.3f}s")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        await client.disconnect()
    else:
        print("❌ Could not connect to real PostgreSQL server")
    
    # Part 2: Test with Mock Server
    print("\n\n🎭 PART 2: Testing with Mock PostgreSQL Server")
    print("-" * 60)
    
    mock_client = PostgresMCPClient(test_connection)
    connected = await mock_client.connect(use_mock=True)
    
    if connected:
        print("✅ Connected to mock PostgreSQL MCP server!")
        
        # Test mock operations
        print("\n📊 Mock Database Operations:")
        
        # List tables
        tables_result = await mock_client.list_tables()
        print(f"\nTables found: {len(tables_result.get('tables', []))}")
        
        # Get schema
        schema_result = await mock_client.get_schema("users")
        print(f"\nSchema for 'users' table: {json.dumps(schema_result, indent=2)}")
        
        # Test query
        query_result = await mock_client.execute_query("SELECT * FROM tools WHERE type = 'mcp'")
        print(f"\nMCP tools in mock database: {json.dumps(query_result, indent=2)}")
        
        await mock_client.disconnect()
    
    # Part 3: Tool Registry Integration
    print("\n\n🗄️ PART 3: Tool Registry Integration")
    print("-" * 60)
    
    registry = ToolRegistry("tests/data/temp/postgres_mcp/test_registry.db")
    
    # Register tools from real client
    client = PostgresMCPClient(real_connection)
    if await client.connect(use_mock=True):  # Use mock for consistent results
        client.register_tools_to_registry(registry)
        
        # List registered PostgreSQL tools
        pg_tools = registry.list_tools("postgres")
        print(f"\n✅ Registered {len(pg_tools)} PostgreSQL tools:")
        for tool in pg_tools:
            print(f"  - {tool['id']}: Performance score = {tool.get('performance_score', 'N/A')}")
        
        await client.disconnect()
    
    print("\n" + "=" * 80)
    print("✅ PostgreSQL MCP Demonstration Complete!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(demonstrate_postgres_mcp())