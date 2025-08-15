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
        
        # Official Notion MCP server tools - matching github.com/makenotion/notion-mcp-server
        self.tools = [
            {
                "name": "search_content",
                "description": "Search content in Notion workspace",
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
                "name": "create_comments",
                "description": "Create comments on a Notion page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page ID to comment on"
                        },
                        "comment": {
                            "type": "string",
                            "description": "Comment text"
                        }
                    },
                    "required": ["page_id", "comment"]
                }
            },
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
                "name": "retrieve_page_content",
                "description": "Retrieve page content by ID or name",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page ID or name"
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
                "name": "access_page",
                "description": "Access pages by name or ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "identifier": {
                            "type": "string",
                            "description": "Page name or ID"
                        }
                    },
                    "required": ["identifier"]
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
            if tool_name == "search_content":
                return await self.search_content(arguments, request_id)
            elif tool_name == "create_comments":
                return await self.create_comments(arguments, request_id)
            elif tool_name == "create_page":
                return await self.create_page(arguments, request_id)
            elif tool_name == "retrieve_page_content":
                return await self.retrieve_page_content(arguments, request_id)
            elif tool_name == "access_page":
                return await self.access_page(arguments, request_id)
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
    
    async def search_content(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Search content in Notion workspace."""
        query = arguments.get("query", "").lower()
        limit = arguments.get("limit", 10)
        
        results = []
        for page_id, page in self.pages.items():
            if page.get("archived", False):
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
    
    async def create_comments(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Create comments on a Notion page."""
        page_id = arguments.get("page_id", "")
        comment = arguments.get("comment", "")
        
        if page_id not in self.pages:
            return self.error_response(request_id, -32602, f"Page not found: {page_id}")
        
        # Store comment as a special block
        comment_id = str(uuid.uuid4())
        comment_block = {
            "id": comment_id,
            "type": "comment",
            "content": comment,
            "created_time": datetime.now(timezone.utc).isoformat()
        }
        
        if page_id not in self.blocks:
            self.blocks[page_id] = []
        self.blocks[page_id].append(comment_block)
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": comment_id,
                "page_id": page_id,
                "comment": comment,
                "created_time": comment_block["created_time"]
            }
        }
    
    async def retrieve_page_content(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Retrieve page content by ID or name."""
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
    
    async def access_page(self, arguments: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Access pages by name or ID."""
        identifier = arguments.get("identifier", "")
        
        # Try to find by ID first
        if identifier in self.pages:
            page = self.pages[identifier]
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "id": identifier,
                    "title": page["title"],
                    "url": f"https://notion.so/{identifier}",
                    "last_edited_time": page["last_edited_time"]
                }
            }
        
        # Try to find by name
        for page_id, page in self.pages.items():
            if page["title"].lower() == identifier.lower():
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "id": page_id,
                        "title": page["title"],
                        "url": f"https://notion.so/{page_id}",
                        "last_edited_time": page["last_edited_time"]
                    }
                }
        
        return self.error_response(request_id, -32602, f"Page not found: {identifier}")
    
    
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