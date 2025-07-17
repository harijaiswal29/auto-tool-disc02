#!/usr/bin/env python3
"""
Notion MCP Demo Script

This script demonstrates the capabilities of the Notion MCP integration,
showing how to create pages, manage databases, and perform various
operations in a Notion workspace.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tools.notion_mcp import NotionMCPClient
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def demo_basic_operations(client: NotionMCPClient):
    """Demonstrate basic page operations."""
    print("\n" + "="*60)
    print("DEMO: Basic Page Operations")
    print("="*60)
    
    # Create a page
    print("\n1. Creating a new page...")
    page = await client.create_page(
        title="Demo Page - Auto Tool Discovery",
        content="""# Auto Tool Discovery Project

## Overview
This page was created automatically using the Notion MCP integration.

## Features Demonstrated
- Page creation
- Markdown content support
- Property management
- Block manipulation

## Project Details
The Auto Tool Discovery system uses AI agents to discover and integrate tools through the Model Context Protocol (MCP).

### Key Components
1. Intent Recognition
2. Tool Discovery
3. Tool Selection & Learning
4. Execution & Monitoring
5. Learning & Adaptation
""",
        properties={
            "Tags": ["demo", "mcp", "auto-tool-discovery"],
            "Status": "Active",
            "Created By": "Notion MCP Demo"
        }
    )
    
    page_id = page['id']
    print(f"✅ Created page with ID: {page_id}")
    print(f"   URL: {page.get('url', 'N/A')}")
    
    # Get the page
    print("\n2. Retrieving page content...")
    retrieved_page = await client.get_page(page_id)
    print(f"✅ Retrieved page: {retrieved_page['title']}")
    print(f"   Content preview: {retrieved_page['content'][:100]}...")
    
    # Update the page
    print("\n3. Updating page content...")
    await client.update_page(
        page_id,
        content=retrieved_page['content'] + "\n\n### Update Note\nThis content was added via update operation at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        properties={"Status": "Updated"}
    )
    print("✅ Page updated successfully")
    
    # Append blocks
    print("\n4. Appending various block types...")
    
    # Add a heading
    await client.append_block(page_id, "heading_2", "Additional Information")
    print("✅ Added heading block")
    
    # Add a paragraph
    await client.append_block(
        page_id, 
        "paragraph", 
        "This section demonstrates how to append different types of blocks to a Notion page."
    )
    print("✅ Added paragraph block")
    
    # Add bulleted list items
    await client.append_block(page_id, "bulleted_list_item", "First bullet point")
    await client.append_block(page_id, "bulleted_list_item", "Second bullet point")
    await client.append_block(page_id, "bulleted_list_item", "Third bullet point")
    print("✅ Added bulleted list")
    
    # Add a quote
    await client.append_block(
        page_id,
        "quote",
        "The best way to predict the future is to invent it. - Alan Kay"
    )
    print("✅ Added quote block")
    
    return page_id


async def demo_database_operations(client: NotionMCPClient):
    """Demonstrate database operations."""
    print("\n" + "="*60)
    print("DEMO: Database Operations")
    print("="*60)
    
    # Create a database
    print("\n1. Creating a project tracking database...")
    db = await client.create_database(
        title="MCP Integration Tasks",
        properties={
            "Task": {"type": "title"},
            "Status": {
                "type": "select",
                "options": ["To Do", "In Progress", "Testing", "Done"]
            },
            "Priority": {
                "type": "select",
                "options": ["High", "Medium", "Low"]
            },
            "Component": {
                "type": "select",
                "options": ["Client", "Server", "Tests", "Documentation"]
            },
            "Due Date": {"type": "date"},
            "Assignee": {"type": "rich_text"}
        }
    )
    
    db_id = db['id']
    print(f"✅ Created database with ID: {db_id}")
    
    # Add records to the database
    print("\n2. Adding tasks to the database...")
    
    tasks = [
        {
            "Task": "Implement Notion MCP client",
            "Status": "Done",
            "Priority": "High",
            "Component": "Client",
            "Assignee": "AI Agent"
        },
        {
            "Task": "Create mock server for testing",
            "Status": "Done",
            "Priority": "High",
            "Component": "Server",
            "Assignee": "AI Agent"
        },
        {
            "Task": "Write integration tests",
            "Status": "Done",
            "Priority": "Medium",
            "Component": "Tests",
            "Assignee": "AI Agent"
        },
        {
            "Task": "Update documentation",
            "Status": "In Progress",
            "Priority": "Medium",
            "Component": "Documentation",
            "Assignee": "AI Agent"
        },
        {
            "Task": "Performance optimization",
            "Status": "To Do",
            "Priority": "Low",
            "Component": "Client",
            "Due Date": "2024-12-31",
            "Assignee": "Future Developer"
        }
    ]
    
    for task in tasks:
        record = await client.create_database_record(db_id, properties=task)
        print(f"✅ Added task: {task['Task']}")
    
    # Query the database
    print("\n3. Querying database for high priority tasks...")
    # Note: The mock server doesn't implement complex filtering,
    # but this shows how it would work with the real API
    results = await client.query_database(db_id, limit=10)
    
    print(f"✅ Found {len(results['results'])} tasks in database")
    
    return db_id


async def demo_search_operations(client: NotionMCPClient):
    """Demonstrate search functionality."""
    print("\n" + "="*60)
    print("DEMO: Search Operations")
    print("="*60)
    
    # Search for pages
    print("\n1. Searching for pages containing 'MCP'...")
    search_results = await client.search_pages("MCP", limit=5)
    
    print(f"✅ Found {len(search_results['results'])} pages:")
    for result in search_results['results']:
        print(f"   - {result['title']} (ID: {result['id']})")
    
    print("\n2. Searching for pages containing 'demo'...")
    demo_results = await client.search_pages("demo", limit=5)
    
    print(f"✅ Found {len(demo_results['results'])} pages:")
    for result in demo_results['results']:
        print(f"   - {result['title']}")


async def demo_workspace_operations(client: NotionMCPClient):
    """Demonstrate workspace-level operations."""
    print("\n" + "="*60)
    print("DEMO: Workspace Operations")
    print("="*60)
    
    # List workspace pages
    print("\n1. Listing workspace pages...")
    pages = await client.list_workspace_pages(limit=10)
    
    print(f"✅ Found {len(pages['pages'])} pages in workspace:")
    for page in pages['pages'][:5]:  # Show first 5
        print(f"   - {page['title']}")
        print(f"     Last edited: {page['last_edited_time']}")


async def demo_with_registry(client: NotionMCPClient):
    """Demonstrate integration with tool registry."""
    print("\n" + "="*60)
    print("DEMO: Tool Registry Integration")
    print("="*60)
    
    # Create a temporary registry
    registry = ToolRegistry("data/demo_notion_registry.db")
    
    # Register Notion tools
    print("\n1. Registering Notion tools...")
    client.register_tools_to_registry(registry)
    
    # List registered tools
    notion_tools = [
        tool for tool in registry.list_tools() 
        if tool['id'].startswith('notion_')
    ]
    
    print(f"✅ Registered {len(notion_tools)} Notion tools:")
    for tool in notion_tools[:5]:  # Show first 5
        print(f"   - {tool['id']}: {tool['capabilities']['description']}")


async def main():
    """Run the Notion MCP demo."""
    print("\n" + "="*60)
    print("NOTION MCP INTEGRATION DEMO")
    print("="*60)
    
    # Initialize client
    client = NotionMCPClient()
    
    # Check for real token
    use_mock = True
    if os.environ.get('NOTION_INTEGRATION_TOKEN'):
        print("\n🔑 Notion integration token detected.")
        print("Would you like to use the real Notion API? (y/n): ", end="")
        choice = input().strip().lower()
        use_mock = choice != 'y'
    else:
        print("\n⚠️  No NOTION_INTEGRATION_TOKEN found in environment.")
        print("Using mock server for demonstration.")
    
    try:
        # Connect to server
        print(f"\n📡 Connecting to {'mock' if use_mock else 'real'} Notion server...")
        connected = await client.connect(use_mock=use_mock)
        
        if not connected:
            print("❌ Failed to connect to Notion server")
            return
        
        print("✅ Successfully connected!")
        print(f"📋 Available tools: {len(client.tools)}")
        
        # Run demos
        page_id = await demo_basic_operations(client)
        db_id = await demo_database_operations(client)
        await demo_search_operations(client)
        await demo_workspace_operations(client)
        await demo_with_registry(client)
        
        # Cache demonstration
        print("\n" + "="*60)
        print("DEMO: Caching Behavior")
        print("="*60)
        
        print("\n1. First read (not cached)...")
        await client.get_page(page_id)
        
        print("2. Second read (should be cached)...")
        await client.get_page(page_id)
        
        print("3. Clearing cache...")
        client.clear_cache()
        
        print("4. Third read (not cached after clear)...")
        await client.get_page(page_id)
        
        print("\n✅ Caching demonstration complete")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect
        print("\n🔌 Disconnecting from Notion server...")
        await client.disconnect()
        print("✅ Disconnected")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    
    if use_mock:
        print("\nℹ️  This demo used a mock server.")
        print("To use the real Notion API:")
        print("1. Create an integration at https://www.notion.so/profile/integrations")
        print("2. Set NOTION_INTEGRATION_TOKEN environment variable")
        print("3. Share pages/databases with your integration")
        print("4. Run the demo again")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())