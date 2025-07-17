"""
Integration tests for Notion MCP client.

Tests the Notion MCP client's interaction with both mock and real servers.
"""

import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.notion_mcp import NotionMCPClient
from src.tools.mock_notion_mcp import MockNotionMCPServer
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestNotionMCPIntegration:
    """Integration tests for Notion MCP client."""
    
    @pytest.fixture
    async def mock_server(self):
        """Create a mock Notion server."""
        server = MockNotionMCPServer()
        return server
    
    @pytest.fixture
    async def client(self):
        """Create a Notion MCP client."""
        client = NotionMCPClient()
        yield client
        # Cleanup
        if client.connected:
            await client.disconnect()
    
    @pytest.fixture
    def tool_registry(self, tmp_path):
        """Create a temporary tool registry."""
        db_path = tmp_path / "test_registry.db"
        return ToolRegistry(str(db_path))
    
    @pytest.mark.asyncio
    async def test_connect_to_mock_server(self, client):
        """Test connecting to mock server."""
        # Connect to mock server
        connected = await client.connect(use_mock=True)
        
        assert connected is True
        assert client.connected is True
        assert client.use_mock is True
        assert client.mock_server is not None
        assert len(client.tools) > 0
    
    @pytest.mark.asyncio
    async def test_discover_tools(self, client):
        """Test tool discovery from mock server."""
        # Connect first
        await client.connect(use_mock=True)
        
        # Tools should be discovered during connection
        assert len(client.tools) > 0
        
        # Verify tool names
        tool_names = [tool['name'] for tool in client.tools]
        assert 'create_page' in tool_names
        assert 'get_page' in tool_names
        assert 'update_page' in tool_names
        assert 'delete_page' in tool_names
        assert 'search_pages' in tool_names
        assert 'create_database' in tool_names
        assert 'query_database' in tool_names
    
    @pytest.mark.asyncio
    async def test_create_and_get_page(self, client):
        """Test creating and retrieving a page."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Create a page
        page_result = await client.create_page(
            title="Test Page",
            content="# Test Page\n\nThis is test content.",
            properties={"Tags": ["test", "integration"]}
        )
        
        assert 'id' in page_result
        assert page_result['title'] == "Test Page"
        page_id = page_result['id']
        
        # Get the page
        page_content = await client.get_page(page_id)
        
        assert page_content['id'] == page_id
        assert page_content['title'] == "Test Page"
        assert "# Test Page" in page_content['content']
        assert page_content['properties']['Tags'] == ["test", "integration"]
    
    @pytest.mark.asyncio
    async def test_update_page(self, client):
        """Test updating page content."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Create a page first
        page_result = await client.create_page(
            title="Original Title",
            content="Original content"
        )
        page_id = page_result['id']
        
        # Update the page
        update_result = await client.update_page(
            page_id,
            title="Updated Title",
            content="Updated content\n\nWith more lines",
            properties={"Status": "Updated"}
        )
        
        assert update_result['success'] is True
        
        # Verify update
        updated_page = await client.get_page(page_id)
        assert updated_page['title'] == "Updated Title"
        assert "Updated content" in updated_page['content']
        assert updated_page['properties']['Status'] == "Updated"
    
    @pytest.mark.asyncio
    async def test_delete_page(self, client):
        """Test deleting (archiving) a page."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Create a page
        page_result = await client.create_page(title="To Delete")
        page_id = page_result['id']
        
        # Delete the page
        delete_result = await client.delete_page(page_id)
        
        assert delete_result['success'] is True
        assert delete_result['archived'] is True
    
    @pytest.mark.asyncio
    async def test_search_pages(self, client):
        """Test searching for pages."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Create some pages
        await client.create_page(title="Python Tutorial", content="Learn Python programming")
        await client.create_page(title="JavaScript Guide", content="JavaScript basics")
        await client.create_page(title="Python Advanced", content="Advanced Python concepts")
        
        # Search for Python pages
        search_result = await client.search_pages("Python")
        
        assert 'results' in search_result
        assert len(search_result['results']) >= 2
        
        # Verify search results contain Python
        for result in search_result['results']:
            assert 'python' in result['title'].lower()
    
    @pytest.mark.asyncio
    async def test_database_operations(self, client):
        """Test database creation and querying."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Create a database
        db_result = await client.create_database(
            title="Task Tracker",
            properties={
                "Name": {"type": "title"},
                "Status": {"type": "select", "options": ["Todo", "In Progress", "Done"]},
                "Priority": {"type": "select", "options": ["High", "Medium", "Low"]},
                "Due Date": {"type": "date"}
            }
        )
        
        assert 'id' in db_result
        db_id = db_result['id']
        
        # Create a record
        record_result = await client.create_database_record(
            db_id,
            properties={
                "Name": "Complete integration tests",
                "Status": "In Progress",
                "Priority": "High",
                "Due Date": "2024-12-31"
            }
        )
        
        assert 'id' in record_result
        assert record_result['properties']['Name'] == "Complete integration tests"
        
        # Query the database
        query_result = await client.query_database(db_id)
        
        assert 'results' in query_result
        assert len(query_result['results']) > 0
    
    @pytest.mark.asyncio
    async def test_append_block(self, client):
        """Test appending blocks to a page."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Create a page
        page_result = await client.create_page(
            title="Block Test",
            content="Initial content"
        )
        page_id = page_result['id']
        
        # Append various block types
        block_results = []
        
        # Heading
        result = await client.append_block(page_id, "heading_2", "New Section")
        block_results.append(result)
        
        # Paragraph
        result = await client.append_block(page_id, "paragraph", "This is a new paragraph.")
        block_results.append(result)
        
        # Bulleted list
        result = await client.append_block(page_id, "bulleted_list_item", "First item")
        block_results.append(result)
        
        # Verify all blocks were added
        for result in block_results:
            assert result['success'] is True
            assert 'id' in result
        
        # Get updated page content
        updated_page = await client.get_page(page_id)
        assert "New Section" in updated_page['content']
        assert "This is a new paragraph" in updated_page['content']
        assert "First item" in updated_page['content']
    
    @pytest.mark.asyncio
    async def test_list_workspace_pages(self, client):
        """Test listing workspace pages."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Create some pages
        for i in range(5):
            await client.create_page(title=f"Page {i+1}")
        
        # List pages
        pages_result = await client.list_workspace_pages(limit=10)
        
        assert 'pages' in pages_result
        assert len(pages_result['pages']) >= 5
        
        # Verify page structure
        for page in pages_result['pages']:
            assert 'id' in page
            assert 'title' in page
            assert 'url' in page
            assert 'last_edited_time' in page
    
    @pytest.mark.asyncio
    async def test_caching(self, client):
        """Test that read operations are cached."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Create a page
        page_result = await client.create_page(title="Cache Test")
        page_id = page_result['id']
        
        # Clear cache first
        client.clear_cache()
        
        # First read - should not be cached
        page1 = await client.get_page(page_id)
        
        # Second read - should be cached
        page2 = await client.get_page(page_id)
        
        # Results should be identical
        assert page1 == page2
        
        # Clear cache and read again
        client.clear_cache()
        page3 = await client.get_page(page_id)
        
        # Content should still be the same
        assert page3['title'] == page1['title']
    
    @pytest.mark.asyncio
    async def test_tool_registry_integration(self, client, tool_registry):
        """Test registering Notion tools with the tool registry."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Register tools
        client.register_tools_to_registry(tool_registry)
        
        # Verify tools are registered
        notion_tools = tool_registry.list_tools(server_type="mcp")
        notion_tool_names = [tool['name'] for tool in notion_tools if tool['id'].startswith('notion_')]
        
        assert len(notion_tool_names) > 0
        assert any('create_page' in name for name in notion_tool_names)
        assert any('query_database' in name for name in notion_tool_names)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling for invalid operations."""
        # Connect to mock server
        await client.connect(use_mock=True)
        
        # Try to get non-existent page
        with pytest.raises(Exception) as exc_info:
            await client.get_page("non-existent-id")
        
        assert "Page not found" in str(exc_info.value)
        
        # Try to update non-existent page
        with pytest.raises(Exception) as exc_info:
            await client.update_page("non-existent-id", title="New Title")
        
        assert "Page not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, client):
        """Test handling of connection failures."""
        # Try to connect without token (should fail for real server)
        client.integration_token = None
        connected = await client.connect(use_mock=False)
        
        assert connected is False
        assert not client.connected
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get('NOTION_INTEGRATION_TOKEN'), 
                        reason="Requires NOTION_INTEGRATION_TOKEN environment variable")
    async def test_real_server_connection(self, client):
        """Test connecting to real Notion server (requires token)."""
        # This test only runs if a real token is provided
        connected = await client.connect(use_mock=False)
        
        assert connected is True
        assert client.connected is True
        assert len(client.tools) > 0
        
        # Disconnect
        await client.disconnect()


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v", "-k", "test_create_and_get_page"])