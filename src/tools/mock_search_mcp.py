"""
Mock Search MCP Server

A mock implementation of Search MCP server for testing without requiring
external APIs or the actual MCP server to be installed.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockSearchMCPServer:
    """
    Mock Search MCP server that simulates various search capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.initialized = False
        self.tools = self._define_tools()
        
        # Mock data for different search types
        self.mock_data = self._initialize_mock_data()
        
        logger.info(f"[MOCK] Mock Search MCP Server initialized")
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available search tools."""
        return [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 10
                        },
                        "language": {
                            "type": "string",
                            "description": "Language filter (e.g., 'en', 'es')"
                        },
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string", "format": "date"},
                                "to": {"type": "string", "format": "date"}
                            }
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "code_search",
                "description": "Search code repositories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Code search query"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language"
                        },
                        "repository": {
                            "type": "string",
                            "description": "Specific repository to search"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "doc_search",
                "description": "Search technical documentation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Documentation search query"
                        },
                        "source": {
                            "type": "string",
                            "description": "Documentation source (e.g., 'python', 'javascript')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "news_search",
                "description": "Search news articles",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "News search query"
                        },
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string", "format": "date"},
                                "to": {"type": "string", "format": "date"}
                            }
                        },
                        "category": {
                            "type": "string",
                            "description": "News category"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "scholarly_search",
                "description": "Search academic papers and research",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Academic search query"
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Academic fields to filter by"
                        },
                        "year_range": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "integer"},
                                "to": {"type": "integer"}
                            }
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def _initialize_mock_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize mock search results for different search types."""
        return {
            "web": [
                {
                    "title": "Model Context Protocol - Official Documentation",
                    "url": "https://modelcontextprotocol.io",
                    "snippet": "The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs.",
                    "relevance_score": 0.95
                },
                {
                    "title": "Understanding MCP Architecture",
                    "url": "https://example.com/mcp-architecture",
                    "snippet": "Deep dive into MCP architecture and implementation patterns for AI applications.",
                    "relevance_score": 0.87
                },
                {
                    "title": "Building MCP Servers Tutorial",
                    "url": "https://example.com/mcp-tutorial",
                    "snippet": "Step-by-step guide to building your own MCP servers for various use cases.",
                    "relevance_score": 0.82
                }
            ],
            "code": [
                {
                    "repository": "modelcontextprotocol/servers",
                    "file": "src/server.py",
                    "line": 42,
                    "code": "async def connect(self, use_mock: bool = False) -> bool:",
                    "language": "python",
                    "stars": 1523
                },
                {
                    "repository": "example/mcp-client",
                    "file": "client/main.py",
                    "line": 156,
                    "code": "async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:",
                    "language": "python",
                    "stars": 342
                }
            ],
            "docs": [
                {
                    "title": "asyncio — Asynchronous I/O",
                    "source": "Python Documentation",
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "content": "This module provides infrastructure for writing single-threaded concurrent code using coroutines.",
                    "version": "3.11"
                },
                {
                    "title": "Getting Started with Async/Await",
                    "source": "Python Tutorial",
                    "url": "https://docs.python.org/3/library/asyncio-task.html",
                    "content": "Coroutines declared with the async/await syntax is the preferred way of writing asyncio applications.",
                    "version": "3.11"
                }
            ],
            "news": [
                {
                    "title": "AI Models Get Smarter with New Context Protocol",
                    "source": "Tech News Daily",
                    "date": "2024-11-15",
                    "url": "https://technews.example.com/ai-context-protocol",
                    "summary": "Major breakthrough in AI communication as new protocol enables better context understanding.",
                    "category": "technology"
                },
                {
                    "title": "Open Source Community Embraces MCP",
                    "source": "Developer Weekly",
                    "date": "2024-11-10",
                    "url": "https://devweekly.example.com/mcp-adoption",
                    "summary": "Developers worldwide are rapidly adopting the Model Context Protocol for AI applications.",
                    "category": "technology"
                }
            ],
            "scholarly": [
                {
                    "title": "Advances in Context-Aware Machine Learning Systems",
                    "authors": ["Smith, J.", "Doe, A.", "Johnson, K."],
                    "journal": "Journal of AI Research",
                    "year": 2024,
                    "doi": "10.1234/jair.2024.123",
                    "abstract": "We present a novel approach to context-aware machine learning using standardized protocols.",
                    "citations": 45
                },
                {
                    "title": "Protocol-Based Integration of Large Language Models",
                    "authors": ["Chen, L.", "Williams, R."],
                    "conference": "International Conference on AI Systems",
                    "year": 2024,
                    "doi": "10.1234/icais.2024.456",
                    "abstract": "This paper explores the benefits of protocol-based integration for LLM applications.",
                    "citations": 23
                }
            ]
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.debug(f"[MOCK] Handling request: {method}")
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return await self._handle_tool_call(request_id, params)
        else:
            return self._error_response(request_id, f"Unknown method: {method}")
    
    def _handle_initialize(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        self.initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": "MockSearchMCPServer",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": True,
                    "resources": False
                }
            }
        }
    
    def _handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """Handle tools list request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self.tools
            }
        }
    
    async def _handle_tool_call(self, request_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "web_search":
                result = await self._web_search(arguments)
            elif tool_name == "code_search":
                result = await self._code_search(arguments)
            elif tool_name == "doc_search":
                result = await self._doc_search(arguments)
            elif tool_name == "news_search":
                result = await self._news_search(arguments)
            elif tool_name == "scholarly_search":
                result = await self._scholarly_search(arguments)
            else:
                return self._error_response(request_id, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self._error_response(request_id, str(e))
    
    async def _web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate web search."""
        query = args.get("query", "")
        num_results = args.get("num_results", 10)
        
        # Filter and score mock results based on query
        results = []
        for item in self.mock_data["web"]:
            # Simple relevance scoring based on query presence
            if query.lower() in item["title"].lower() or query.lower() in item["snippet"].lower():
                score = item["relevance_score"] * (0.8 + random.random() * 0.2)
                results.append({**item, "relevance_score": score})
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        results = results[:num_results]
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }
    
    async def _code_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate code search."""
        query = args.get("query", "")
        language = args.get("language")
        
        results = []
        for item in self.mock_data["code"]:
            # Filter by language if specified
            if language and item["language"] != language:
                continue
            
            # Simple matching
            if query.lower() in item["code"].lower():
                results.append(item)
        
        return {
            "query": query,
            "language": language,
            "total_results": len(results),
            "results": results
        }
    
    async def _doc_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate documentation search."""
        query = args.get("query", "")
        source = args.get("source")
        
        results = []
        for item in self.mock_data["docs"]:
            # Filter by source if specified
            if source and source.lower() not in item["source"].lower():
                continue
            
            # Simple matching
            if query.lower() in item["title"].lower() or query.lower() in item["content"].lower():
                results.append(item)
        
        return {
            "query": query,
            "source": source,
            "total_results": len(results),
            "results": results
        }
    
    async def _news_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate news search."""
        query = args.get("query", "")
        date_range = args.get("date_range", {})
        
        results = []
        for item in self.mock_data["news"]:
            # Simple matching
            if query.lower() in item["title"].lower() or query.lower() in item["summary"].lower():
                # Add date filtering logic here if needed
                results.append(item)
        
        return {
            "query": query,
            "date_range": date_range,
            "total_results": len(results),
            "results": results
        }
    
    async def _scholarly_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate scholarly/academic search."""
        query = args.get("query", "")
        fields = args.get("fields", [])
        
        results = []
        for item in self.mock_data["scholarly"]:
            # Simple matching
            if query.lower() in item["title"].lower() or query.lower() in item["abstract"].lower():
                results.append(item)
        
        # Sort by citations
        results.sort(key=lambda x: x["citations"], reverse=True)
        
        return {
            "query": query,
            "fields": fields,
            "total_results": len(results),
            "results": results
        }
    
    def _error_response(self, request_id: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": message
            }
        }


async def test_mock_server():
    """Test the mock Search MCP server."""
    logger.info("=" * 60)
    logger.info("[TEST] Testing Mock Search MCP Server")
    logger.info("=" * 60)
    
    # Create mock server
    server = MockSearchMCPServer({"api_key": "test_key"})
    
    # Test initialization
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0",
            "clientInfo": {"name": "TestClient", "version": "0.1.0"}
        },
        "id": 1
    }
    init_response = await server.handle_request(init_request)
    logger.info(f"[INIT] Response: {init_response}")
    
    # Test tools list
    list_request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2
    }
    list_response = await server.handle_request(list_request)
    logger.info(f"[TOOLS] Available tools: {len(list_response['result']['tools'])}")
    
    # Test web search
    web_search_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {
                "query": "Model Context Protocol",
                "num_results": 3
            }
        },
        "id": 3
    }
    web_response = await server.handle_request(web_search_request)
    logger.info(f"[WEB_SEARCH] Results: {web_response['result']['total_results']} found")
    
    # Test code search
    code_search_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "code_search",
            "arguments": {
                "query": "async def",
                "language": "python"
            }
        },
        "id": 4
    }
    code_response = await server.handle_request(code_search_request)
    logger.info(f"[CODE_SEARCH] Results: {code_response['result']['total_results']} found")
    
    # Test scholarly search
    scholarly_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "scholarly_search",
            "arguments": {
                "query": "machine learning",
                "fields": ["computer science"]
            }
        },
        "id": 5
    }
    scholarly_response = await server.handle_request(scholarly_request)
    logger.info(f"[SCHOLARLY_SEARCH] Results: {scholarly_response['result']['total_results']} found")
    
    logger.info("[TEST] Mock server test complete!")


if __name__ == "__main__":
    asyncio.run(test_mock_server())