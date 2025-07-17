"""
Mock Notion MCP Server

This module provides a mock implementation of the Notion MCP server
for development and testing when the real server is not available.
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockNotionMCPServer:
    """Mock Notion MCP server for testing."""
    
    def __init__(self):
        self.name = "notion-mock"
        self.description = "Mock Notion workspace operations"
        
        # In-memory storage for mock data
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.databases: Dict[str, Dict[str, Any]] = {}
        self.blocks: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with some sample data
        self._init_sample_data()
        
        self.tools = [
            {
                "name": "create_page",
                "description": "Create a new page in Notion",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Page title"
                        },
                        "parent_id": {
                            "type": "string",
                            "description": "Parent page or database ID (optional)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Page content in Markdown format"
                        },
                        "properties": {
                            "type": "object",
                            "description": "Page properties (optional)"
                        }
                    },
                    "required": ["title"]
                }
            },
            {
                "name": "get_page",
                "description": "Get page content and properties",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page ID"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "json"],
                            "description": "Output format (default: markdown)"
                        }
                    },
                    "required": ["page_id"]
                }
            },
            {
                "name": "update_page",
                "description": "Update page content or properties",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page ID"
                        },
                        "title": {
                            "type": "string",
                            "description": "New title (optional)"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content in Markdown format (optional)"
                        },
                        "properties": {
                            "type": "object",
                            "description": "Updated properties (optional)"
                        }
                    },
                    "required": ["page_id"]
                }
            },
            {
                "name": "delete_page",
                "description": "Delete a page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page ID to delete"
                        }
                    },
                    "required": ["page_id"]
                }
            },
            {
                "name": "search_pages",
                "description": "Search for pages in the workspace",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 10)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "create_database",
                "description": "Create a new database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Database title"
                        },
                        "parent_id": {
                            "type": "string",
                            "description": "Parent page ID (optional)"
                        },
                        "properties": {
                            "type": "object",
                            "description": "Database schema properties"
                        }
                    },
                    "required": ["title", "properties"]
                }
            },
            {
                "name": "query_database",
                "description": "Query a database with filters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database_id": {
                            "type": "string",
                            "description": "Database ID"
                        },
                        "filter": {
                            "type": "object",
                            "description": "Filter conditions (optional)"
                        },
                        "sorts": {
                            "type": "array",
                            "description": "Sort order (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 10)"
                        }
                    },
                    "required": ["database_id"]
                }
            },
            {
                "name": "create_database_record",
                "description": "Create a new record in a database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database_id": {
                            "type": "string",
                            "description": "Database ID"
                        },
                        "properties": {
                            "type": "object",
                            "description": "Record properties"
                        }
                    },
                    "required": ["database_id", "properties"]
                }
            },
            {
                "name": "append_block",
                "description": "Append a block to a page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page ID"
                        },
                        "block_type": {
                            "type": "string",
                            "enum": ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "code", "quote"],
                            "description": "Block type"
                        },
                        "content": {
                            "type": "string",
                            "description": "Block content"
                        }
                    },
                    "required": ["page_id", "block_type", "content"]
                }
            },
            {
                "name": "list_workspace_pages",
                "description": "List all pages in the workspace",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 20)"
                        }
                    }
                }
            }
        ]
        
        logger.info(f"[INIT] Mock Notion MCP server initialized")
    
    def _init_sample_data(self):
        """Initialize with some sample data."""
        # Sample page
        page_id = str(uuid.uuid4())
        self.pages[page_id] = {
            "id": page_id,
            "title": "Welcome to Notion",
            "content": "# Welcome to Notion\n\nThis is a sample page in your mock Notion workspace.\n\n## Features\n- Create pages\n- Organize with databases\n- Rich text editing\n- Collaboration",
            "properties": {
                "Tags": ["documentation", "getting-started"],
                "Status": "Published"
            },
            "created_time": datetime.now(timezone.utc).isoformat(),
            "last_edited_time": datetime.now(timezone.utc).isoformat(),
            "archived": False
        }
        
        # Sample blocks for the page
        self.blocks[page_id] = [
            {
                "id": str(uuid.uuid4()),
                "type": "heading_1",
                "content": "Welcome to Notion"
            },
            {
                "id": str(uuid.uuid4()),
                "type": "paragraph",
                "content": "This is a sample page in your mock Notion workspace."
            }
        ]
        
        # Sample database
        db_id = str(uuid.uuid4())
        self.databases[db_id] = {
            "id": db_id,
            "title": "Task Database",
            "properties": {
                "Name": {"type": "title"},
                "Status": {"type": "select", "options": ["To Do", "In Progress", "Done"]},
                "Priority": {"type": "select", "options": ["High", "Medium", "Low"]},
                "Due Date": {"type": "date"}
            },
            "records": [
                {
                    "id": str(uuid.uuid4()),
                    "properties": {
                        "Name": "Complete project documentation",
                        "Status": "In Progress",
                        "Priority": "High",
                        "Due Date": "2024-12-31"
                    }
                }
            ]
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC requests."""
        method = request.get("method", "")
        request_id = request.get("id", 0)
        
        logger.debug(f"[REQUEST] Method: {method}, ID: {request_id}")
        
        if method == "initialize":
            return self.handle_initialize(request_id)
        elif method == "tools/list":
            return self.handle_tools_list(request_id)
        elif method == "tools/call":
            return await self.handle_tool_call(request, request_id)
        else:
            return self.error_response(request_id, -32601, f"Method not found: {method}")
    
    def handle_initialize(self, request_id: int) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": self.name,
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": True,
                    "notion": {
                        "pages": True,
                        "databases": True,
                        "blocks": True,
                        "search": True,
                        "markdown_export": True
                    }
                }
            }
        }
    
    def handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """List available tools."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self.tools
            }
        }
    
    async def handle_tool_call(self, request: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Execute Notion operations."""
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        logger.info(f"[TOOL_CALL] Tool: {tool_name}, Args: {arguments}")
        
        try:
            if tool_name == "create_page":
                return await self.create_page(arguments, request_id)
            elif tool_name == "get_page":
                return await self.get_page(arguments, request_id)
            elif tool_name == "update_page":
                return await self.update_page(arguments, request_id)
            elif tool_name == "delete_page":
                return await self.delete_page(arguments, request_id)
            elif tool_name == "search_pages":
                return await self.search_pages(arguments, request_id)
            elif tool_name == "create_database":
                return await self.create_database(arguments, request_id)
            elif tool_name == "query_database":
                return await self.query_database(arguments, request_id)
            elif tool_name == "create_database_record":
                return await self.create_database_record(arguments, request_id)
            elif tool_name == "append_block":
                return await self.append_block(arguments, request_id)
            elif tool_name == "list_workspace_pages":
                return await self.list_workspace_pages(arguments, request_id)
            else:
                return self.error_response(request_id, -32602, f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"[ERROR] Tool execution failed: {str(e)}")
            return self.error_response(request_id, -32603, str(e))
    
    async def create_page(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Create a new page."""
        title = arguments.get("title", "")
        parent_id = arguments.get("parent_id")
        content = arguments.get("content", "")
        properties = arguments.get("properties", {})
        
        page_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        self.pages[page_id] = {
            "id": page_id,
            "title": title,
            "content": content,
            "properties": properties,
            "parent_id": parent_id,
            "created_time": now,
            "last_edited_time": now,
            "archived": False
        }
        
        # Parse content into blocks
        if content:
            self.blocks[page_id] = self._parse_markdown_to_blocks(content)
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": page_id,
                "url": f"https://notion.so/{page_id}",
                "title": title,
                "created_time": now
            }
        }
    
    async def get_page(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Get page content."""
        page_id = arguments.get("page_id", "")
        format_type = arguments.get("format", "markdown")
        
        if page_id not in self.pages:
            return self.error_response(request_id, -32602, f"Page not found: {page_id}")
        
        page = self.pages[page_id]
        
        if format_type == "json":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": page
            }
        else:
            # Return as markdown
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "id": page_id,
                    "title": page["title"],
                    "content": page["content"],
                    "properties": page["properties"],
                    "last_edited_time": page["last_edited_time"]
                }
            }
    
    async def update_page(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Update page content or properties."""
        page_id = arguments.get("page_id", "")
        
        if page_id not in self.pages:
            return self.error_response(request_id, -32602, f"Page not found: {page_id}")
        
        page = self.pages[page_id]
        
        # Update fields if provided
        if "title" in arguments:
            page["title"] = arguments["title"]
        if "content" in arguments:
            page["content"] = arguments["content"]
            self.blocks[page_id] = self._parse_markdown_to_blocks(arguments["content"])
        if "properties" in arguments:
            page["properties"].update(arguments["properties"])
        
        page["last_edited_time"] = datetime.now(timezone.utc).isoformat()
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": page_id,
                "success": True,
                "last_edited_time": page["last_edited_time"]
            }
        }
    
    async def delete_page(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Delete a page."""
        page_id = arguments.get("page_id", "")
        
        if page_id not in self.pages:
            return self.error_response(request_id, -32602, f"Page not found: {page_id}")
        
        # Archive instead of delete
        self.pages[page_id]["archived"] = True
        self.pages[page_id]["last_edited_time"] = datetime.now(timezone.utc).isoformat()
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": page_id,
                "success": True,
                "archived": True
            }
        }
    
    async def search_pages(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Search for pages."""
        query = arguments.get("query", "").lower()
        limit = arguments.get("limit", 10)
        
        results = []
        for page_id, page in self.pages.items():
            if page["archived"]:
                continue
                
            # Search in title and content
            if (query in page["title"].lower() or 
                query in page.get("content", "").lower()):
                results.append({
                    "id": page_id,
                    "title": page["title"],
                    "url": f"https://notion.so/{page_id}",
                    "last_edited_time": page["last_edited_time"]
                })
                
                if len(results) >= limit:
                    break
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "results": results,
                "has_more": len(results) == limit
            }
        }
    
    async def create_database(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Create a new database."""
        title = arguments.get("title", "")
        parent_id = arguments.get("parent_id")
        properties = arguments.get("properties", {})
        
        db_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        self.databases[db_id] = {
            "id": db_id,
            "title": title,
            "parent_id": parent_id,
            "properties": properties,
            "records": [],
            "created_time": now,
            "last_edited_time": now
        }
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": db_id,
                "url": f"https://notion.so/{db_id}",
                "title": title
            }
        }
    
    async def query_database(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Query a database."""
        database_id = arguments.get("database_id", "")
        filter_obj = arguments.get("filter", {})
        sorts = arguments.get("sorts", [])
        limit = arguments.get("limit", 10)
        
        if database_id not in self.databases:
            return self.error_response(request_id, -32602, f"Database not found: {database_id}")
        
        database = self.databases[database_id]
        records = database.get("records", [])
        
        # Simple filtering (in real implementation would be more complex)
        if filter_obj:
            # This is a simplified filter - real Notion has complex filter syntax
            filtered_records = records
        else:
            filtered_records = records[:limit]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "results": filtered_records,
                "has_more": len(records) > limit
            }
        }
    
    async def create_database_record(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Create a record in a database."""
        database_id = arguments.get("database_id", "")
        properties = arguments.get("properties", {})
        
        if database_id not in self.databases:
            return self.error_response(request_id, -32602, f"Database not found: {database_id}")
        
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "properties": properties,
            "created_time": datetime.now(timezone.utc).isoformat()
        }
        
        self.databases[database_id]["records"].append(record)
        self.databases[database_id]["last_edited_time"] = datetime.now(timezone.utc).isoformat()
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": record_id,
                "url": f"https://notion.so/{database_id}#{record_id}",
                "properties": properties
            }
        }
    
    async def append_block(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Append a block to a page."""
        page_id = arguments.get("page_id", "")
        block_type = arguments.get("block_type", "paragraph")
        content = arguments.get("content", "")
        
        if page_id not in self.pages:
            return self.error_response(request_id, -32602, f"Page not found: {page_id}")
        
        block_id = str(uuid.uuid4())
        block = {
            "id": block_id,
            "type": block_type,
            "content": content
        }
        
        if page_id not in self.blocks:
            self.blocks[page_id] = []
        
        self.blocks[page_id].append(block)
        self.pages[page_id]["last_edited_time"] = datetime.now(timezone.utc).isoformat()
        
        # Update page content
        self.pages[page_id]["content"] = self._blocks_to_markdown(self.blocks[page_id])
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": block_id,
                "type": block_type,
                "success": True
            }
        }
    
    async def list_workspace_pages(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """List all pages in workspace."""
        limit = arguments.get("limit", 20)
        
        pages = []
        for page_id, page in self.pages.items():
            if not page["archived"]:
                pages.append({
                    "id": page_id,
                    "title": page["title"],
                    "url": f"https://notion.so/{page_id}",
                    "last_edited_time": page["last_edited_time"]
                })
                
                if len(pages) >= limit:
                    break
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "pages": pages,
                "has_more": len(pages) == limit
            }
        }
    
    def _parse_markdown_to_blocks(self, markdown: str) -> List[Dict[str, Any]]:
        """Simple markdown to blocks parser."""
        blocks = []
        lines = markdown.split('\n')
        
        for line in lines:
            if line.strip():
                block_id = str(uuid.uuid4())
                
                if line.startswith('# '):
                    blocks.append({
                        "id": block_id,
                        "type": "heading_1",
                        "content": line[2:]
                    })
                elif line.startswith('## '):
                    blocks.append({
                        "id": block_id,
                        "type": "heading_2",
                        "content": line[3:]
                    })
                elif line.startswith('### '):
                    blocks.append({
                        "id": block_id,
                        "type": "heading_3",
                        "content": line[4:]
                    })
                elif line.startswith('- '):
                    blocks.append({
                        "id": block_id,
                        "type": "bulleted_list_item",
                        "content": line[2:]
                    })
                elif line.startswith('> '):
                    blocks.append({
                        "id": block_id,
                        "type": "quote",
                        "content": line[2:]
                    })
                else:
                    blocks.append({
                        "id": block_id,
                        "type": "paragraph",
                        "content": line
                    })
        
        return blocks
    
    def _blocks_to_markdown(self, blocks: List[Dict[str, Any]]) -> str:
        """Convert blocks back to markdown."""
        lines = []
        
        for block in blocks:
            content = block.get("content", "")
            block_type = block.get("type", "paragraph")
            
            if block_type == "heading_1":
                lines.append(f"# {content}")
            elif block_type == "heading_2":
                lines.append(f"## {content}")
            elif block_type == "heading_3":
                lines.append(f"### {content}")
            elif block_type == "bulleted_list_item":
                lines.append(f"- {content}")
            elif block_type == "numbered_list_item":
                lines.append(f"1. {content}")
            elif block_type == "quote":
                lines.append(f"> {content}")
            elif block_type == "code":
                lines.append(f"```\n{content}\n```")
            else:
                lines.append(content)
        
        return '\n'.join(lines)
    
    def error_response(self, request_id: int, code: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }


# Test script
async def test_mock_notion_server():
    """Test the mock Notion MCP server."""
    logger.info("[TEST] Starting mock Notion MCP server tests")
    
    # Initialize mock server
    server = MockNotionMCPServer()
    
    # Test initialize
    init_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1
    })
    logger.info(f"[INIT] Response: {init_response['result']['serverInfo']['name']}")
    
    # Test tools list
    tools_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    })
    logger.info(f"[TOOLS] Available tools: {len(tools_response['result']['tools'])}")
    
    # Test create page
    create_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "create_page",
            "arguments": {
                "title": "Test Page",
                "content": "# Test Page\n\nThis is a test page created via mock server.\n\n## Features\n- Easy to use\n- Fast\n- Reliable"
            }
        },
        "id": 3
    })
    logger.info(f"[CREATE] Response: {create_response}")
    page_id = create_response['result']['id']
    
    # Test get page
    get_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "get_page",
            "arguments": {
                "page_id": page_id
            }
        },
        "id": 4
    })
    logger.info(f"[GET] Page content: {get_response['result']['content'][:100]}...")
    
    # Test search
    search_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_pages",
            "arguments": {
                "query": "test"
            }
        },
        "id": 5
    })
    logger.info(f"[SEARCH] Found {len(search_response['result']['results'])} pages")
    
    # Test create database
    db_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "create_database",
            "arguments": {
                "title": "Test Tasks",
                "properties": {
                    "Name": {"type": "title"},
                    "Status": {"type": "select", "options": ["Todo", "Done"]},
                    "Priority": {"type": "select", "options": ["High", "Low"]}
                }
            }
        },
        "id": 6
    })
    logger.info(f"[DATABASE] Created: {db_response['result']['id']}")
    
    logger.info("[TEST] Mock Notion server tests complete!")


if __name__ == "__main__":
    asyncio.run(test_mock_notion_server())