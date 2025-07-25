#!/usr/bin/env python3
"""
PostgreSQL MCP Demonstration Script

Shows various PostgreSQL MCP capabilities including:
- Real server connection (if available)
- Mock server fallback
- Query execution
- Schema inspection
- Tool registry integration
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.postgres_mcp import PostgresMCPClient
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


async def demonstrate_postgres_mcp():
    """Comprehensive demonstration of PostgreSQL MCP functionality."""
    
    print_section("PostgreSQL MCP Demonstration")
    print(f"Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Database connection string
    connection = "postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"
    
    # Part 1: Connection Testing
    print_section("Part 1: Connection Testing")
    
    client = PostgresMCPClient(connection)
    
    # Try real server first
    print("\n🔌 Attempting to connect to real PostgreSQL MCP server...")
    connected = await client.connect(use_mock=False)
    
    if connected:
        print("✅ Successfully connected to real PostgreSQL MCP server!")
        server_type = "real"
    else:
        print("❌ Could not connect to real server")
        print("🎭 Falling back to mock server...")
        connected = await client.connect(use_mock=True)
        if connected:
            print("✅ Connected to mock PostgreSQL MCP server!")
            server_type = "mock"
        else:
            print("❌ Could not connect to any server")
            return
    
    # Part 2: Tool Discovery
    print_section("Part 2: Tool Discovery")
    
    print(f"\n🔧 Discovered {len(client.tools)} tools:")
    for tool in client.tools:
        print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        if 'inputSchema' in tool:
            required = tool['inputSchema'].get('required', [])
            if required:
                print(f"    Required params: {', '.join(required)}")
    
    # Part 3: Query Execution
    print_section("Part 3: Query Execution")
    
    # Test queries that work with both real and mock servers
    test_queries = [
        ("Database Version", "SELECT version()"),
        ("Current Timestamp", "SELECT current_timestamp"),
    ]
    
    if server_type == "mock":
        # Add mock-specific queries
        test_queries.extend([
            ("List Users", "SELECT * FROM users"),
            ("List Tools", "SELECT * FROM tools WHERE type = 'mcp'"),
            ("Count Tables", "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        ])
    else:
        # Add real server queries
        test_queries.extend([
            ("List Tables", "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"),
            ("Database Size", "SELECT pg_database_size(current_database()) as size_bytes"),
        ])
    
    for title, query in test_queries:
        print(f"\n📋 {title}:")
        print(f"   Query: {query}")
        
        result = await client.execute_query(query)
        
        if result['success']:
            print(f"   ✅ Success!")
            if server_type == "mock":
                # Mock server returns different format
                print(f"   Result: {json.dumps(result['result'], indent=6)}")
            else:
                # Real server returns JSON-RPC format
                if 'content' in result['result']:
                    data = json.loads(result['result']['content'][0]['text'])
                    print(f"   Result: {json.dumps(data, indent=6)}")
            print(f"   ⏱️  Execution time: {result['execution_time']:.3f}s")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    
    # Part 4: Schema Operations
    print_section("Part 4: Schema Operations")
    
    print("\n📊 Listing database tables...")
    tables_result = await client.list_tables()
    
    if tables_result['success']:
        tables = tables_result.get('tables', [])
        print(f"   Found {len(tables)} tables:")
        for table in tables[:5]:  # Show first 5 tables
            print(f"   - {table['table_name']} ({table['table_type']})")
        if len(tables) > 5:
            print(f"   ... and {len(tables) - 5} more")
    
    # Get schema for a specific table (if available)
    if tables_result['success'] and tables_result.get('tables'):
        first_table = tables_result['tables'][0]['table_name']
        print(f"\n📋 Getting schema for table '{first_table}'...")
        
        schema_result = await client.get_schema(first_table)
        
        if schema_result['success']:
            columns = schema_result.get('columns', [])
            print(f"   Table '{first_table}' has {len(columns)} columns:")
            for col in columns[:5]:  # Show first 5 columns
                nullable = "NULL" if col['is_nullable'] == "YES" else "NOT NULL"
                print(f"   - {col['column_name']} ({col['data_type']}) {nullable}")
            if len(columns) > 5:
                print(f"   ... and {len(columns) - 5} more columns")
    
    # Part 5: Tool Registry Integration
    print_section("Part 5: Tool Registry Integration")
    
    # Create temporary registry
    registry_path = Path("tests/data/temp/postgres_demo_registry.db")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    registry = ToolRegistry(str(registry_path))
    
    print("\n📝 Registering PostgreSQL tools to registry...")
    client.register_tools_to_registry(registry)
    
    # List registered tools
    pg_tools = registry.list_tools("postgres")
    print(f"✅ Registered {len(pg_tools)} PostgreSQL tools:")
    for tool in pg_tools:
        print(f"  - {tool['id']}")
    
    # Disconnect
    await client.disconnect()
    print("\n🔌 Disconnected from PostgreSQL MCP server")
    
    print_section("Demo Complete!")
    print(f"Server used: {server_type}")
    print(f"Tools discovered: {len(client.tools)}")
    print(f"Registry path: {registry_path}")


if __name__ == "__main__":
    asyncio.run(demonstrate_postgres_mcp())