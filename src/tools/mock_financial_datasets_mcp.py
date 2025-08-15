"""
Mock Financial Datasets MCP Server

This module provides a mock implementation of the Financial Datasets MCP server
for testing purposes without requiring API keys or network access.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random

from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockFinancialDatasetsMCPServer:
    """
    Mock implementation of Financial Datasets MCP server for testing.
    """
    
    def __init__(self):
        """Initialize mock server with predefined responses."""
        self.server_name = "mock_financial_datasets"
        self.tools = self._define_tools()
        self.mock_data = self._initialize_mock_data()
        
        logger.info("[MOCK] Mock Financial Datasets MCP Server initialized")
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available financial data tools matching official github.com/financial-datasets/mcp-server."""
        return [
            {
                "name": "get_income_statements",
                "description": "Get income statements for a company",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "period": {
                            "type": "string",
                            "description": "Period: latest, annual, quarterly",
                            "default": "latest"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_balance_sheets",
                "description": "Get balance sheets for a company",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "period": {
                            "type": "string",
                            "description": "Period: latest, annual, quarterly",
                            "default": "latest"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_cash_flow_statements",
                "description": "Get cash flow statements for a company",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "period": {
                            "type": "string",
                            "description": "Period: latest, annual, quarterly",
                            "default": "latest"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_current_stock_price",
                "description": "Get the current / latest price of a company",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_historical_stock_prices",
                "description": "Gets historical stock prices for a company",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_company_news",
                "description": "Get news for a company",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of news items to return",
                            "default": 10
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_available_crypto_tickers",
                "description": "Gets all available crypto tickers",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_crypto_prices",
                "description": "Gets historical prices for a crypto currency",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Crypto symbol (e.g., BTC, ETH)"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_historical_crypto_prices",
                "description": "Gets historical prices for a crypto currency",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Crypto symbol (e.g., BTC, ETH)"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_current_crypto_price",
                "description": "Get the current / latest price of a crypto currency",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Crypto symbol (e.g., BTC, ETH)"
                        },
                        "currency": {
                            "type": "string",
                            "description": "Target currency",
                            "default": "USD"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        ]
    
    def _initialize_mock_data(self) -> Dict[str, Any]:
        """Initialize mock financial data."""
        return {
            "stock_prices": {
                "AAPL": {"price": 182.52, "change": 2.34, "change_percent": 1.3, "volume": 52346789},
                "MSFT": {"price": 429.87, "change": -1.23, "change_percent": -0.29, "volume": 23456789},
                "GOOGL": {"price": 141.23, "change": 0.87, "change_percent": 0.62, "volume": 18234567},
                "TSLA": {"price": 238.45, "change": 5.67, "change_percent": 2.44, "volume": 98765432},
                "AMZN": {"price": 178.34, "change": -2.15, "change_percent": -1.19, "volume": 34567890}
            },
            "income_statements": {
                "AAPL": {
                    "revenue": 383285000000,
                    "gross_profit": 169148000000,
                    "operating_income": 114301000000,
                    "net_income": 96995000000,
                    "eps": 6.16,
                    "period": "2023",
                    "currency": "USD"
                },
                "MSFT": {
                    "revenue": 211915000000,
                    "gross_profit": 146052000000,
                    "operating_income": 88523000000,
                    "net_income": 72361000000,
                    "eps": 9.72,
                    "period": "2023",
                    "currency": "USD"
                }
            },
            "balance_sheets": {
                "AAPL": {
                    "total_assets": 352755000000,
                    "total_liabilities": 290437000000,
                    "total_equity": 62318000000,
                    "cash": 29965000000,
                    "debt": 111110000000,
                    "period": "2023",
                    "currency": "USD"
                },
                "MSFT": {
                    "total_assets": 411976000000,
                    "total_liabilities": 205753000000,
                    "total_equity": 206223000000,
                    "cash": 111262000000,
                    "debt": 78405000000,
                    "period": "2023",
                    "currency": "USD"
                }
            },
            "cash_flows": {
                "AAPL": {
                    "operating_cash_flow": 110543000000,
                    "investing_cash_flow": -7375000000,
                    "financing_cash_flow": -103629000000,
                    "free_cash_flow": 99802000000,
                    "period": "2023",
                    "currency": "USD"
                },
                "MSFT": {
                    "operating_cash_flow": 87582000000,
                    "investing_cash_flow": -23950000000,
                    "financing_cash_flow": -55979000000,
                    "free_cash_flow": 59046000000,
                    "period": "2023",
                    "currency": "USD"
                }
            },
            "crypto_prices": {
                "BTC": {"USD": 68234.56, "EUR": 63145.23, "change_24h": 3.45},
                "ETH": {"USD": 3456.78, "EUR": 3198.45, "change_24h": -1.23},
                "BNB": {"USD": 623.45, "EUR": 576.89, "change_24h": 2.11}
            },
            "companies": [
                {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
                {"symbol": "TSLA", "name": "Tesla, Inc.", "exchange": "NASDAQ"},
                {"symbol": "AMZN", "name": "Amazon.com, Inc.", "exchange": "NASDAQ"}
            ]
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle JSON-RPC request.
        
        Args:
            request: JSON-RPC request
            
        Returns:
            JSON-RPC response
        """
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.debug(f"[MOCK] Handling request: {method}")
        
        try:
            if method == "initialize":
                return self._handle_initialize(request_id)
            elif method == "tools/list":
                return self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tool_call(params, request_id)
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")
                
        except Exception as e:
            logger.error(f"[MOCK] Error handling request: {e}")
            return self._error_response(request_id, -32603, str(e))
    
    def _handle_initialize(self, request_id: int) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": "Mock Financial Datasets MCP Server",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": True,
                    "resources": False,
                    "prompts": False
                }
            },
            "id": request_id
        }
    
    def _handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """Handle tools list request."""
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": self.tools
            },
            "id": request_id
        }
    
    async def _handle_tool_call(self, params: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Handle tool call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        if tool_name == "get_income_statements":
            result = self._get_income_statements(arguments)
        elif tool_name == "get_balance_sheets":
            result = self._get_balance_sheets(arguments)
        elif tool_name == "get_cash_flow_statements":
            result = self._get_cash_flow_statements(arguments)
        elif tool_name == "get_current_stock_price":
            result = self._get_current_stock_price(arguments)
        elif tool_name == "get_historical_stock_prices":
            result = self._get_historical_stock_prices(arguments)
        elif tool_name == "get_company_news":
            result = self._get_company_news(arguments)
        elif tool_name == "get_available_crypto_tickers":
            result = self._get_available_crypto_tickers(arguments)
        elif tool_name == "get_crypto_prices":
            result = self._get_crypto_prices(arguments)
        elif tool_name == "get_historical_crypto_prices":
            result = self._get_historical_crypto_prices(arguments)
        elif tool_name == "get_current_crypto_price":
            result = self._get_current_crypto_price(arguments)
        else:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")
        
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
    
    def _get_current_stock_price(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock stock price data."""
        symbol = args.get("symbol", "").upper()
        
        if symbol in self.mock_data["stock_prices"]:
            data = self.mock_data["stock_prices"][symbol].copy()
            data["symbol"] = symbol
            data["timestamp"] = datetime.now().isoformat()
            return data
        else:
            # Generate random data for unknown symbols
            return {
                "symbol": symbol,
                "price": round(random.uniform(10, 500), 2),
                "change": round(random.uniform(-10, 10), 2),
                "change_percent": round(random.uniform(-5, 5), 2),
                "volume": random.randint(1000000, 100000000),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_income_statements(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock income statement data."""
        symbol = args.get("symbol", "").upper()
        period = args.get("period", "latest")
        
        if symbol in self.mock_data["income_statements"]:
            data = self.mock_data["income_statements"][symbol].copy()
            data["symbol"] = symbol
            return data
        else:
            # Generate mock data
            return {
                "symbol": symbol,
                "revenue": random.randint(1000000000, 100000000000),
                "gross_profit": random.randint(500000000, 50000000000),
                "operating_income": random.randint(100000000, 30000000000),
                "net_income": random.randint(50000000, 20000000000),
                "eps": round(random.uniform(0.5, 20), 2),
                "period": period,
                "currency": "USD"
            }
    
    def _get_balance_sheets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock balance sheet data."""
        symbol = args.get("symbol", "").upper()
        
        if symbol in self.mock_data["balance_sheets"]:
            data = self.mock_data["balance_sheets"][symbol].copy()
            data["symbol"] = symbol
            return data
        else:
            return {
                "symbol": symbol,
                "total_assets": random.randint(10000000000, 500000000000),
                "total_liabilities": random.randint(5000000000, 250000000000),
                "total_equity": random.randint(5000000000, 250000000000),
                "cash": random.randint(1000000000, 50000000000),
                "debt": random.randint(1000000000, 100000000000),
                "period": "latest",
                "currency": "USD"
            }
    
    def _get_cash_flow_statements(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock cash flow data."""
        symbol = args.get("symbol", "").upper()
        
        if symbol in self.mock_data["cash_flows"]:
            data = self.mock_data["cash_flows"][symbol].copy()
            data["symbol"] = symbol
            return data
        else:
            return {
                "symbol": symbol,
                "operating_cash_flow": random.randint(1000000000, 100000000000),
                "investing_cash_flow": random.randint(-50000000000, -1000000000),
                "financing_cash_flow": random.randint(-50000000000, -1000000000),
                "free_cash_flow": random.randint(1000000000, 50000000000),
                "period": "latest",
                "currency": "USD"
            }
    
    def _get_company_news(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock company news."""
        symbol = args.get("symbol", "").upper()
        limit = min(args.get("limit", 10), 20)
        
        news_items = []
        base_date = datetime.now()
        
        for i in range(limit):
            news_items.append({
                "title": f"Breaking: {symbol} announces Q{(i % 4) + 1} earnings",
                "summary": f"Company reports strong growth in key markets...",
                "source": random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ"]),
                "timestamp": (base_date - timedelta(hours=i*3)).isoformat(),
                "url": f"https://example.com/news/{symbol}/{i}",
                "sentiment": random.choice(["positive", "neutral", "negative"])
            })
        
        return {"symbol": symbol, "news": news_items}
    
    def _get_current_crypto_price(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock crypto price data."""
        symbol = args.get("symbol", "").upper()
        currency = args.get("currency", "USD").upper()
        
        if symbol in self.mock_data["crypto_prices"]:
            prices = self.mock_data["crypto_prices"][symbol]
            price = prices.get(currency, prices.get("USD", 0) * 0.93)  # Rough conversion
            return {
                "symbol": symbol,
                "currency": currency,
                "price": price,
                "change_24h": prices.get("change_24h", 0),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "symbol": symbol,
                "currency": currency,
                "price": round(random.uniform(0.01, 10000), 2),
                "change_24h": round(random.uniform(-20, 20), 2),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_historical_stock_prices(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock historical stock prices."""
        symbol = args.get("symbol", "AAPL")
        start_date = args.get("start_date", "2024-01-01")
        end_date = args.get("end_date", "2024-01-31")
        
        # Generate mock historical data
        prices = []
        base_price = self.mock_data["stock_prices"].get(symbol, {"price": 100})["price"]
        
        # Generate 30 days of mock data
        for i in range(30):
            date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)
            if date > datetime.strptime(end_date, "%Y-%m-%d"):
                break
            
            prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": base_price + random.uniform(-5, 5),
                "high": base_price + random.uniform(0, 10),
                "low": base_price + random.uniform(-10, 0),
                "close": base_price + random.uniform(-5, 5),
                "volume": random.randint(10000000, 100000000)
            })
        
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "prices": prices
        }
    
    def _get_available_crypto_tickers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get available crypto tickers."""
        return {
            "tickers": ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "DOGE", "AVAX", "MATIC"]
        }
    
    def _get_crypto_prices(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical crypto prices."""
        symbol = args.get("symbol", "BTC")
        start_date = args.get("start_date", "2024-01-01")
        end_date = args.get("end_date", "2024-01-31")
        
        # Generate mock historical crypto data
        prices = []
        base_price = self.mock_data["crypto_prices"].get(symbol, {"USD": 50000})["USD"]
        
        for i in range(30):
            date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)
            if date > datetime.strptime(end_date, "%Y-%m-%d"):
                break
            
            prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": base_price + random.uniform(-1000, 1000),
                "high": base_price + random.uniform(0, 2000),
                "low": base_price + random.uniform(-2000, 0),
                "close": base_price + random.uniform(-1000, 1000),
                "volume": random.uniform(1000000, 10000000)
            })
        
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "prices": prices
        }
    
    def _get_historical_crypto_prices(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical crypto prices - same as get_crypto_prices."""
        return self._get_crypto_prices(args)
    
    
    def _error_response(self, request_id: int, code: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "id": request_id
        }