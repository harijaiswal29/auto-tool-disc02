"""
Setup script to create test data with tool relationships.

This script sets up a realistic tool registry with various tools and relationships
for testing the Tool Discovery Agent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def setup_test_tools(registry_path: str = None):
    """Set up test tools and relationships in the registry."""
    
    if registry_path is None:
        registry_path = "data/registry/test_tools.db"
    
    # Initialize registry
    registry = ToolRegistry(registry_path)
    await registry.initialize()
    
    logger.info("Setting up test tools and relationships...")
    
    # Define comprehensive set of tools
    tools = [
        # Filesystem tools
        {
            'id': 'filesystem.read',
            'name': 'File Reader',
            'server_type': 'filesystem',
            'endpoint': 'npx @modelcontextprotocol/server-filesystem',
            'description': 'Read files from the filesystem',
            'capabilities': {
                'operations': [
                    {'name': 'read_file', 'category': 'read'},
                    {'name': 'list_directory', 'category': 'list'},
                    {'name': 'get_file_info', 'category': 'info'}
                ]
            }
        },
        {
            'id': 'filesystem.write',
            'name': 'File Writer',
            'server_type': 'filesystem',
            'endpoint': 'npx @modelcontextprotocol/server-filesystem',
            'description': 'Write and modify files',
            'capabilities': {
                'operations': [
                    {'name': 'write_file', 'category': 'write'},
                    {'name': 'create_directory', 'category': 'create'},
                    {'name': 'delete_file', 'category': 'delete'}
                ]
            }
        },
        
        # Database tools
        {
            'id': 'sqlite.query',
            'name': 'SQLite Query Tool',
            'server_type': 'sqlite',
            'endpoint': 'python src/tools/sqlite_mcp.py',
            'description': 'Query SQLite databases',
            'capabilities': {
                'operations': [
                    {'name': 'execute_query', 'category': 'query'},
                    {'name': 'list_tables', 'category': 'list'},
                    {'name': 'describe_table', 'category': 'info'}
                ]
            }
        },
        {
            'id': 'postgres.query',
            'name': 'PostgreSQL Query Tool',
            'server_type': 'postgres',
            'endpoint': 'npx @modelcontextprotocol/server-postgres',
            'description': 'Query PostgreSQL databases',
            'capabilities': {
                'operations': [
                    {'name': 'execute_query', 'category': 'query'},
                    {'name': 'list_databases', 'category': 'list'},
                    {'name': 'describe_schema', 'category': 'info'}
                ]
            }
        },
        {
            'id': 'database.export',
            'name': 'Database Export Tool',
            'server_type': 'database',
            'endpoint': 'custom://db-export',
            'description': 'Export data from databases to various formats',
            'capabilities': {
                'operations': [
                    {'name': 'export_csv', 'category': 'export'},
                    {'name': 'export_json', 'category': 'export'},
                    {'name': 'export_excel', 'category': 'export'}
                ]
            }
        },
        
        # Search tools
        {
            'id': 'search.web',
            'name': 'Web Search Tool',
            'server_type': 'search',
            'endpoint': 'npx brave-search-mcp',
            'description': 'Search the web using Brave Search API',
            'capabilities': {
                'operations': [
                    {'name': 'search_web', 'category': 'search'},
                    {'name': 'search_images', 'category': 'search'},
                    {'name': 'search_news', 'category': 'search'}
                ]
            }
        },
        {
            'id': 'search.code',
            'name': 'Code Search Tool',
            'server_type': 'search',
            'endpoint': 'custom://code-search',
            'description': 'Search through code repositories',
            'capabilities': {
                'operations': [
                    {'name': 'search_code', 'category': 'search'},
                    {'name': 'find_functions', 'category': 'find'},
                    {'name': 'find_imports', 'category': 'find'}
                ]
            }
        },
        
        # API/Integration tools
        {
            'id': 'github.api',
            'name': 'GitHub API Tool',
            'server_type': 'github',
            'endpoint': 'npx @modelcontextprotocol/server-github',
            'description': 'Interact with GitHub repositories',
            'capabilities': {
                'operations': [
                    {'name': 'list_repos', 'category': 'list'},
                    {'name': 'create_issue', 'category': 'create'},
                    {'name': 'get_pr_info', 'category': 'info'}
                ]
            }
        },
        {
            'id': 'notion.api',
            'name': 'Notion API Tool',
            'server_type': 'notion',
            'endpoint': 'python src/tools/notion_mcp.py',
            'description': 'Manage Notion workspace',
            'capabilities': {
                'operations': [
                    {'name': 'create_page', 'category': 'create'},
                    {'name': 'update_page', 'category': 'update'},
                    {'name': 'search_pages', 'category': 'search'}
                ]
            }
        },
        
        # Analysis tools
        {
            'id': 'ml.analyze',
            'name': 'ML Analysis Tool',
            'server_type': 'ml',
            'endpoint': 'custom://ml-analysis',
            'description': 'Analyze data using machine learning',
            'capabilities': {
                'operations': [
                    {'name': 'analyze_sentiment', 'category': 'analyze'},
                    {'name': 'classify_text', 'category': 'analyze'},
                    {'name': 'predict_values', 'category': 'predict'}
                ]
            }
        },
        {
            'id': 'data.visualize',
            'name': 'Data Visualization Tool',
            'server_type': 'visualization',
            'endpoint': 'custom://data-viz',
            'description': 'Create visualizations from data',
            'capabilities': {
                'operations': [
                    {'name': 'create_chart', 'category': 'create'},
                    {'name': 'generate_report', 'category': 'generate'},
                    {'name': 'export_image', 'category': 'export'}
                ]
            }
        }
    ]
    
    # Register all tools
    for tool in tools:
        success = registry.register_tool(tool)
        if success:
            logger.info(f"Registered tool: {tool['id']}")
        else:
            logger.error(f"Failed to register tool: {tool['id']}")
    
    # Define tool relationships
    relationships = [
        # Complementary relationships (tools that work well together)
        ('filesystem.read', 'filesystem.write', 'complements', 0.9),
        ('sqlite.query', 'database.export', 'complements', 0.85),
        ('postgres.query', 'database.export', 'complements', 0.85),
        ('database.export', 'filesystem.write', 'complements', 0.8),
        ('search.web', 'ml.analyze', 'complements', 0.7),
        ('search.code', 'github.api', 'complements', 0.8),
        ('ml.analyze', 'data.visualize', 'complements', 0.9),
        ('notion.api', 'filesystem.read', 'complements', 0.6),
        
        # Requires relationships (one tool needs another)
        ('database.export', 'filesystem.write', 'requires', 0.9),
        ('data.visualize', 'ml.analyze', 'requires', 0.7),
        ('ml.analyze', 'sqlite.query', 'requires', 0.6),
        
        # Alternative relationships (tools that can substitute each other)
        ('sqlite.query', 'postgres.query', 'alternative', 0.8),
        ('search.web', 'search.code', 'alternative', 0.5),
    ]
    
    # Add all relationships
    for tool1, tool2, rel_type, strength in relationships:
        success = await registry.add_tool_relationship(tool1, tool2, rel_type, strength)
        if success:
            logger.info(f"Added relationship: {tool1} {rel_type} {tool2} (strength: {strength})")
        else:
            logger.error(f"Failed to add relationship: {tool1} {rel_type} {tool2}")
    
    # Add some usage history for performance scores
    usage_data = [
        ('filesystem.read', True, 0.05),
        ('filesystem.read', True, 0.04),
        ('filesystem.write', True, 0.08),
        ('sqlite.query', True, 0.15),
        ('sqlite.query', True, 0.12),
        ('postgres.query', True, 0.20),
        ('database.export', True, 0.25),
        ('search.web', True, 0.10),
        ('search.web', False, 0.50),  # One failure
        ('ml.analyze', True, 1.20),
        ('ml.analyze', True, 1.15),
        ('data.visualize', True, 0.30),
    ]
    
    for tool_id, success, exec_time in usage_data:
        registry.record_usage(tool_id, success, exec_time, task_type="test")
    
    logger.info(f"Setup complete! Registered {len(tools)} tools with {len(relationships)} relationships")
    
    # Print summary
    all_tools = await registry.get_all_tools()
    all_relationships = await registry.get_tool_relationships()
    
    print(f"\nRegistry Summary:")
    print(f"- Total tools: {len(all_tools)}")
    print(f"- Total relationships: {len(all_relationships)}")
    
    # Group relationships by type
    rel_types = {}
    for rel in all_relationships:
        rel_type = rel['relationship_type']
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    print("\nRelationship types:")
    for rel_type, count in rel_types.items():
        print(f"  - {rel_type}: {count}")
    
    await registry.close()
    
    return registry_path


async def verify_setup(registry_path: str):
    """Verify the setup was successful."""
    registry = ToolRegistry(registry_path)
    await registry.initialize()
    
    # Get all tools
    tools = await registry.get_all_tools()
    print(f"\nVerifying setup...")
    print(f"Found {len(tools)} tools in registry")
    
    # Get all relationships
    relationships = await registry.get_tool_relationships()
    print(f"Found {len(relationships)} relationships")
    
    # Sample some tools
    print("\nSample tools:")
    for tool in tools[:3]:
        print(f"  - {tool['name']} ({tool['id']})")
        
    await registry.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup test tools and relationships")
    parser.add_argument("--path", default="data/registry/test_tools.db", 
                       help="Path to registry database")
    parser.add_argument("--verify", action="store_true",
                       help="Verify setup after creation")
    
    args = parser.parse_args()
    
    # Run setup
    path = asyncio.run(setup_test_tools(args.path))
    
    # Optionally verify
    if args.verify:
        asyncio.run(verify_setup(path))