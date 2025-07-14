"""
Quick integration test to verify the system works.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.agents import OrchestratorAgent


async def test_integration():
    """Test the integrated system."""
    print("Starting integration test...")
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent()
    await orchestrator.initialize()
    
    # Add mock servers for testing
    mcp = orchestrator.mcp_integration
    await mcp.add_sqlite_server("data/test.db", "test_sqlite", use_mock=True)
    await mcp.add_search_server(server_id="test_search", use_mock=True)
    
    # Test queries
    test_queries = [
        "Find all Python files",
        "Search for information about AI",
        "Create a new config file"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        # Process query
        result = await orchestrator.process_user_query(query)
        
        # Display results
        print(f"Intent: {result.intent.primary_intent.type} "
              f"(confidence: {result.intent.primary_intent.confidence:.2f})")
        print(f"Discovered tools: {len(result.discovered_tools)}")
        print(f"Selected tools: {result.selected_tools}")
        
        if result.execution_results:
            print("Execution results:")
            for exec_result in result.execution_results:
                status = "Success" if exec_result.success else "Failed"
                print(f"  - {exec_result.tool_name}: {status}")
        
        print(f"Total time: {result.total_time_ms:.2f}ms")
    
    # Cleanup
    await orchestrator.shutdown()
    print("\nIntegration test complete!")


if __name__ == "__main__":
    asyncio.run(test_integration())