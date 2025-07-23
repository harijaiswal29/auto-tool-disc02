#!/usr/bin/env python3
"""
Search MCP Integration Tests

Tests the Search MCP tool implementation with integration scenarios.
"""

import asyncio
import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.search_mcp import SearchMCP
from src.tools.mock_search_mcp import MockSearchMCPServer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestSearchMCPIntegration:
    """Test Search MCP integration functionality."""
    
    @pytest.fixture
    async def mock_server(self):
        """Create a mock Search MCP server."""
        server = MockSearchMCPServer()
        await server.start()
        yield server
        await server.stop()
    
    @pytest.fixture
    def search_mcp(self):
        """Create a Search MCP instance."""
        # Use a test API key or mock
        return SearchMCP(api_key="test_api_key")
    
    @pytest.mark.asyncio
    async def test_search_mcp_initialization(self, search_mcp):
        """Test Search MCP initialization."""
        assert search_mcp.name == "search_mcp"
        assert search_mcp.description == "Web search capabilities via Brave Search API"
        assert search_mcp.api_key == "test_api_key"
    
    @pytest.mark.asyncio
    async def test_web_search_mock(self, search_mcp):
        """Test web search with mocked response."""
        mock_response = {
            "web": {
                "results": [
                    {
                        "title": "Python Programming",
                        "url": "https://python.org",
                        "description": "Official Python website",
                        "age": "2 days ago"
                    },
                    {
                        "title": "Learn Python",
                        "url": "https://learnpython.org",
                        "description": "Interactive Python tutorial",
                        "age": "1 week ago"
                    }
                ]
            },
            "query": {
                "original": "python programming"
            }
        }
        
        with patch.object(search_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            result = await search_mcp.execute({
                'action': 'search',
                'query': 'python programming'
            })
            
            assert result['success'] is True
            assert len(result['data']['results']) == 2
            assert result['data']['results'][0]['title'] == 'Python Programming'
            assert result['data']['query'] == 'python programming'
    
    @pytest.mark.asyncio
    async def test_search_with_options(self, search_mcp):
        """Test search with additional options."""
        mock_response = {
            "web": {
                "results": [
                    {
                        "title": "Recent Python News",
                        "url": "https://python.org/news",
                        "description": "Latest Python updates",
                        "age": "1 hour ago"
                    }
                ]
            },
            "news": {
                "results": [
                    {
                        "title": "Python 3.12 Released",
                        "url": "https://python.org/downloads",
                        "description": "New features in Python 3.12",
                        "age": "1 day ago"
                    }
                ]
            }
        }
        
        with patch.object(search_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            result = await search_mcp.execute({
                'action': 'search',
                'query': 'python news',
                'count': 5,
                'freshness': 'day',
                'search_lang': 'en',
                'country': 'us'
            })
            
            assert result['success'] is True
            assert 'results' in result['data']
            assert 'news' in result['data']
    
    @pytest.mark.asyncio
    async def test_news_search(self, search_mcp):
        """Test news-specific search."""
        mock_response = {
            "news": {
                "results": [
                    {
                        "title": "Tech Industry Updates",
                        "url": "https://technews.com/article1",
                        "description": "Latest in technology",
                        "age": "3 hours ago",
                        "source": {
                            "name": "Tech News",
                            "url": "https://technews.com"
                        }
                    }
                ]
            }
        }
        
        with patch.object(search_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            result = await search_mcp.execute({
                'action': 'news',
                'query': 'technology updates'
            })
            
            assert result['success'] is True
            assert len(result['data']['results']) == 1
            assert result['data']['results'][0]['source']['name'] == 'Tech News'
    
    @pytest.mark.asyncio
    async def test_image_search(self, search_mcp):
        """Test image search."""
        mock_response = {
            "images": {
                "results": [
                    {
                        "title": "Python Logo",
                        "url": "https://python.org/static/img/python-logo.png",
                        "thumbnail": "https://python.org/static/img/python-logo-thumb.png",
                        "width": 200,
                        "height": 200
                    }
                ]
            }
        }
        
        with patch.object(search_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            result = await search_mcp.execute({
                'action': 'images',
                'query': 'python logo'
            })
            
            assert result['success'] is True
            assert len(result['data']['results']) == 1
            assert 'thumbnail' in result['data']['results'][0]
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, search_mcp):
        """Test handling empty query."""
        result = await search_mcp.execute({
            'action': 'search',
            'query': ''
        })
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Query is required' in result['error']
    
    @pytest.mark.asyncio
    async def test_invalid_action(self, search_mcp):
        """Test handling invalid action."""
        result = await search_mcp.execute({
            'action': 'invalid_action',
            'query': 'test'
        })
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Unsupported action' in result['error']
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, search_mcp):
        """Test handling API errors."""
        with patch.object(search_mcp, '_make_api_request',
                         side_effect=Exception("API Error")):
            result = await search_mcp.execute({
                'action': 'search',
                'query': 'test query'
            })
            
            assert result['success'] is False
            assert 'error' in result
            assert 'API Error' in result['error']
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, search_mcp):
        """Test handling rate limit errors."""
        mock_error_response = {
            "error": {
                "code": 429,
                "message": "Rate limit exceeded"
            }
        }
        
        with patch.object(search_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_error_response)()):
            result = await search_mcp.execute({
                'action': 'search',
                'query': 'test query'
            })
            
            assert result['success'] is False
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_mock_server_search(self, mock_server):
        """Test search through mock server."""
        response = mock_server.handle_tool_call(
            'web_search',
            {'query': 'Python programming'}
        )
        
        assert response['result'] is not None
        assert 'results' in response['result']
        results = response['result']['results']
        assert len(results) > 0
        assert all('title' in r and 'url' in r for r in results)
    
    @pytest.mark.asyncio
    async def test_capabilities(self, search_mcp):
        """Test getting tool capabilities."""
        capabilities = search_mcp.get_capabilities()
        
        assert capabilities['name'] == 'search_mcp'
        assert 'operations' in capabilities
        
        operations = capabilities['operations']
        assert any(op['name'] == 'search' for op in operations)
        assert any(op['name'] == 'news' for op in operations)
        assert any(op['name'] == 'images' for op in operations)
        
        # Check search operation details
        search_op = next(op for op in operations if op['name'] == 'search')
        assert 'parameters' in search_op
        assert 'query' in search_op['parameters']
    
    @pytest.mark.asyncio
    async def test_search_result_filtering(self, search_mcp):
        """Test filtering search results."""
        mock_response = {
            "web": {
                "results": [
                    {
                        "title": "Python Tutorial",
                        "url": "https://example1.com",
                        "description": "Learn Python basics",
                        "age": "1 day ago"
                    },
                    {
                        "title": "Advanced Python",
                        "url": "https://example2.com",
                        "description": "Advanced Python concepts",
                        "age": "1 week ago"
                    },
                    {
                        "title": "Python for Data Science",
                        "url": "https://example3.com",
                        "description": "Python in data science",
                        "age": "1 month ago"
                    }
                ]
            }
        }
        
        with patch.object(search_mcp, '_make_api_request',
                         return_value=asyncio.coroutine(lambda: mock_response)()):
            # Test with count limit
            result = await search_mcp.execute({
                'action': 'search',
                'query': 'python',
                'count': 2
            })
            
            # Note: actual filtering would be done by API, 
            # but we can test the parameter passing
            assert result['success'] is True
            assert 'results' in result['data']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])