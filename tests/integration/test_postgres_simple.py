#!/usr/bin/env python3
"""Simple test to verify PostgreSQL MCP is working with real database."""

import asyncio
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.postgres_mcp import PostgresMCPClient

async def test_postgres():
    print("=== PostgreSQL MCP Simple Test ===\n")
    
    # Use the correct database credentials
    connection = "postgresql://auto_tool_user:auto_tool_pass@localhost:5432/auto_tool_disc"
    
    print(f"1. Connecting to: {connection}")
    client = PostgresMCPClient(connection)
    
    # Try real server first
    print("\n2. Attempting connection to real PostgreSQL MCP server...")
    connected = await client.connect(use_mock=False)
    
    if connected:
        print("   ✅ SUCCESS: Connected to real PostgreSQL MCP server!")
        
        # Execute a simple query
        print("\n3. Executing test query...")
        result = await client.execute_query("SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public'")
        
        if result['success']:
            print("   ✅ Query executed successfully!")
            # Parse the result
            data = json.loads(result['result']['content'][0]['text'])
            print(f"   📊 Number of tables in database: {data[0]['table_count']}")
            print(f"   ⏱️  Query execution time: {result['execution_time']:.3f}s")
        else:
            print(f"   ❌ Query failed: {result.get('error')}")
            
        await client.disconnect()
        print("\n4. Disconnected from server.")
    else:
        print("   ❌ FAILED: Could not connect to real server")
        print("\n   Falling back to mock server...")
        
        connected = await client.connect(use_mock=True)
        if connected:
            print("   ✅ Connected to mock server instead")
            
            # Test with mock
            result = await client.execute_query("SELECT version()")
            if result['success']:
                print(f"   📊 Mock server response: {result['result']}")
                
            await client.disconnect()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_postgres())