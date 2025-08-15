"""
Mock Zerodha MCP Server

This module provides a mock implementation of the Zerodha Kite MCP server
for testing purposes without requiring API keys or network access.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import hashlib

from src.utils.logger import get_logger

logger = get_logger(__name__)

class MockZerodhaMCPServer:
    """
    Mock implementation of Zerodha Kite MCP server for testing.
    Simulates trading operations, portfolio management, and market data.
    """
    
    def __init__(self):
        """Initialize mock server with predefined responses."""
        self.server_name = "mock_zerodha"
        self.tools = self._define_tools()
        self.mock_data = self._initialize_mock_data()
        self.order_counter = 1000
        
        logger.info("[MOCK] Mock Zerodha MCP Server initialized")
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available Zerodha trading tools matching official github.com/zerodha/kite-mcp-server."""
        return [
            {
                "name": "login",
                "description": "Login to Kite API and generate authorization link",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "Kite API key"
                        },
                        "api_secret": {
                            "type": "string",
                            "description": "Kite API secret"
                        }
                    },
                    "required": ["api_key", "api_secret"]
                }
            },
            {
                "name": "get_holdings",
                "description": "Get user's equity holdings",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_positions",
                "description": "Get user's open positions (intraday and overnight)",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "place_order",
                "description": "Place a buy/sell order",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "exchange": {
                            "type": "string",
                            "description": "Exchange (NSE, BSE, NFO, etc.)"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "transaction_type": {
                            "type": "string",
                            "description": "BUY or SELL",
                            "enum": ["BUY", "SELL"]
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Quantity to trade"
                        },
                        "product": {
                            "type": "string",
                            "description": "Product type (CNC, MIS, NRML)",
                            "enum": ["CNC", "MIS", "NRML"]
                        },
                        "order_type": {
                            "type": "string",
                            "description": "Order type (MARKET, LIMIT, SL, SL-M)",
                            "enum": ["MARKET", "LIMIT", "SL", "SL-M"]
                        },
                        "price": {
                            "type": "number",
                            "description": "Limit price (for LIMIT orders)",
                            "optional": True
                        },
                        "trigger_price": {
                            "type": "number",
                            "description": "Trigger price (for SL orders)",
                            "optional": True
                        }
                    },
                    "required": ["exchange", "symbol", "transaction_type", "quantity", "product", "order_type"]
                }
            },
            {
                "name": "modify_order",
                "description": "Modify an existing order",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Order ID to modify"
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "New quantity",
                            "optional": True
                        },
                        "price": {
                            "type": "number",
                            "description": "New limit price",
                            "optional": True
                        },
                        "trigger_price": {
                            "type": "number",
                            "description": "New trigger price",
                            "optional": True
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "cancel_order",
                "description": "Cancel an existing order",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Order ID to cancel"
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "get_orders",
                "description": "Get list of all orders for the day",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_trades",
                "description": "Get list of all executed trades",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_quotes",
                "description": "Get market quote for instruments",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "exchange": {
                            "type": "string",
                            "description": "Exchange"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        }
                    },
                    "required": ["exchange", "symbol"]
                }
            },
            {
                "name": "get_ltp",
                "description": "Get last traded price for instruments",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruments": {
                            "type": "array",
                            "description": "List of exchange:symbol pairs",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["instruments"]
                }
            },
            {
                "name": "get_historical_data",
                "description": "Get historical candle data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instrument_token": {
                            "type": "string",
                            "description": "Instrument token"
                        },
                        "from_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        "to_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)"
                        },
                        "interval": {
                            "type": "string",
                            "description": "Candle interval",
                            "enum": ["minute", "5minute", "15minute", "day"]
                        }
                    },
                    "required": ["instrument_token", "from_date", "to_date", "interval"]
                }
            },
            {
                "name": "get_margins",
                "description": "Get account margins and funds",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "segment": {
                            "type": "string",
                            "description": "Trading segment",
                            "enum": ["equity", "commodity"],
                            "optional": True
                        }
                    }
                }
            },
            {
                "name": "search_instruments",
                "description": "Get list of tradeable instruments",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "exchange": {
                            "type": "string",
                            "description": "Exchange to filter by",
                            "optional": True
                        }
                    }
                }
            },
            {
                "name": "get_ohlc",
                "description": "Get OHLC data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruments": {
                            "type": "array",
                            "description": "List of exchange:symbol pairs",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["instruments"]
                }
            },
            {
                "name": "get_profile",
                "description": "User profile information",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_mf_holdings",
                "description": "Mutual fund holdings",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_order_history",
                "description": "Order execution history",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Order ID"
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "get_order_trades",
                "description": "Get trades for a specific order",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Order ID"
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "get_gtts",
                "description": "List GTT orders",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "place_gtt_order",
                "description": "Create GTT orders",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "trigger_type": {
                            "type": "string",
                            "description": "Trigger type (single, two-leg)",
                            "enum": ["single", "two-leg"]
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "exchange": {
                            "type": "string",
                            "description": "Exchange"
                        },
                        "trigger_values": {
                            "type": "array",
                            "description": "Trigger price values",
                            "items": {
                                "type": "number"
                            }
                        },
                        "orders": {
                            "type": "array",
                            "description": "Order details to execute on trigger"
                        }
                    },
                    "required": ["trigger_type", "symbol", "exchange", "trigger_values", "orders"]
                }
            },
            {
                "name": "modify_gtt_order",
                "description": "Modify GTT orders",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "trigger_id": {
                            "type": "string",
                            "description": "GTT trigger ID"
                        },
                        "trigger_values": {
                            "type": "array",
                            "description": "New trigger values",
                            "items": {
                                "type": "number"
                            }
                        }
                    },
                    "required": ["trigger_id"]
                }
            },
            {
                "name": "delete_gtt_order",
                "description": "Delete GTT orders",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "trigger_id": {
                            "type": "string",
                            "description": "GTT trigger ID to delete"
                        }
                    },
                    "required": ["trigger_id"]
                }
            }
        ]
    
    def _initialize_mock_data(self) -> Dict[str, Any]:
        """Initialize mock trading data."""
        return {
            "holdings": [
                {
                    "tradingsymbol": "RELIANCE",
                    "exchange": "NSE",
                    "quantity": 10,
                    "average_price": 2450.50,
                    "last_price": 2520.75,
                    "pnl": 702.50,
                    "product": "CNC",
                    "collateral_quantity": 0,
                    "collateral_type": None
                },
                {
                    "tradingsymbol": "INFY",
                    "exchange": "NSE",
                    "quantity": 25,
                    "average_price": 1420.00,
                    "last_price": 1455.30,
                    "pnl": 882.50,
                    "product": "CNC",
                    "collateral_quantity": 0,
                    "collateral_type": None
                },
                {
                    "tradingsymbol": "TCS",
                    "exchange": "NSE",
                    "quantity": 5,
                    "average_price": 3200.00,
                    "last_price": 3180.50,
                    "pnl": -97.50,
                    "product": "CNC",
                    "collateral_quantity": 0,
                    "collateral_type": None
                }
            ],
            "positions": {
                "net": [
                    {
                        "tradingsymbol": "NIFTY23DEC20000CE",
                        "exchange": "NFO",
                        "quantity": 50,
                        "average_price": 125.50,
                        "last_price": 142.30,
                        "pnl": 840.00,
                        "product": "NRML",
                        "buy_quantity": 50,
                        "sell_quantity": 0,
                        "buy_price": 125.50,
                        "sell_price": 0
                    }
                ],
                "day": [
                    {
                        "tradingsymbol": "SBIN",
                        "exchange": "NSE",
                        "quantity": -100,
                        "average_price": 625.75,
                        "last_price": 623.45,
                        "pnl": 230.00,
                        "product": "MIS",
                        "buy_quantity": 0,
                        "sell_quantity": 100,
                        "buy_price": 0,
                        "sell_price": 625.75
                    }
                ]
            },
            "orders": [],
            "trades": [],
            "quotes": {
                "NSE:RELIANCE": {
                    "instrument_token": "738561",
                    "timestamp": datetime.now().isoformat(),
                    "last_price": 2520.75,
                    "last_quantity": 15,
                    "buy_quantity": 45231,
                    "sell_quantity": 38956,
                    "volume": 2341567,
                    "average_price": 2515.23,
                    "ohlc": {
                        "open": 2505.00,
                        "high": 2535.50,
                        "low": 2495.25,
                        "close": 2498.90
                    },
                    "change": 21.85,
                    "last_trade_time": datetime.now().isoformat(),
                    "oi": 0,
                    "oi_day_high": 0,
                    "oi_day_low": 0,
                    "depth": {
                        "buy": [
                            {"quantity": 50, "price": 2520.50, "orders": 1},
                            {"quantity": 100, "price": 2520.25, "orders": 2},
                            {"quantity": 250, "price": 2520.00, "orders": 5},
                            {"quantity": 500, "price": 2519.75, "orders": 3},
                            {"quantity": 1000, "price": 2519.50, "orders": 8}
                        ],
                        "sell": [
                            {"quantity": 75, "price": 2521.00, "orders": 1},
                            {"quantity": 150, "price": 2521.25, "orders": 3},
                            {"quantity": 300, "price": 2521.50, "orders": 4},
                            {"quantity": 450, "price": 2521.75, "orders": 2},
                            {"quantity": 800, "price": 2522.00, "orders": 6}
                        ]
                    }
                }
            },
            "margins": {
                "equity": {
                    "available": {
                        "cash": 145678.50,
                        "intraday_payin": 0,
                        "opening_balance": 150000.00,
                        "collateral": 85420.00,
                        "live_balance": 145678.50
                    },
                    "used": {
                        "debits": 4321.50,
                        "exposure": 62540.00,
                        "m2m_realised": 0,
                        "m2m_unrealised": 1825.00,
                        "option_premium": 6275.00,
                        "payout": 0,
                        "span": 18750.00,
                        "holding_sales": 0,
                        "turnover": 0
                    },
                    "net": 231098.50
                }
            },
            "instruments": [
                {
                    "instrument_token": "738561",
                    "exchange_token": "2885",
                    "tradingsymbol": "RELIANCE",
                    "name": "RELIANCE INDUSTRIES LTD",
                    "exchange": "NSE",
                    "segment": "NSE",
                    "instrument_type": "EQ",
                    "tick_size": 0.05,
                    "lot_size": 1
                },
                {
                    "instrument_token": "408065",
                    "exchange_token": "1594",
                    "tradingsymbol": "INFY",
                    "name": "INFOSYS LIMITED",
                    "exchange": "NSE",
                    "segment": "NSE",
                    "instrument_type": "EQ",
                    "tick_size": 0.05,
                    "lot_size": 1
                }
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
                    "name": "Mock Zerodha MCP Server",
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
        
        handlers = {
            "login": self._login,
            "get_holdings": self._get_holdings,
            "get_positions": self._get_positions,
            "place_order": self._place_order,
            "modify_order": self._modify_order,
            "cancel_order": self._cancel_order,
            "get_orders": self._get_orders,
            "get_trades": self._get_trades,
            "get_quotes": self._get_quotes,
            "get_ltp": self._get_ltp,
            "get_historical_data": self._get_historical_data,
            "get_margins": self._get_margins,
            "search_instruments": self._search_instruments,
            "get_ohlc": self._get_ohlc,
            "get_profile": self._get_profile,
            "get_mf_holdings": self._get_mf_holdings,
            "get_order_history": self._get_order_history,
            "get_order_trades": self._get_order_trades,
            "get_gtts": self._get_gtts,
            "place_gtt_order": self._place_gtt_order,
            "modify_gtt_order": self._modify_gtt_order,
            "delete_gtt_order": self._delete_gtt_order
        }
        
        handler = handlers.get(tool_name)
        if handler:
            result = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        else:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")
    
    def _get_holdings(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock holdings data."""
        # Update last prices with small random changes
        for holding in self.mock_data["holdings"]:
            change = random.uniform(-0.5, 0.5) * holding["last_price"] / 100
            holding["last_price"] = round(holding["last_price"] + change, 2)
            holding["pnl"] = round((holding["last_price"] - holding["average_price"]) * holding["quantity"], 2)
        
        return self.mock_data["holdings"]
    
    def _get_positions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock positions data."""
        # Update last prices
        for position in self.mock_data["positions"]["net"] + self.mock_data["positions"]["day"]:
            change = random.uniform(-1, 1) * position["last_price"] / 100
            position["last_price"] = round(position["last_price"] + change, 2)
            
            # Calculate P&L
            if position["quantity"] > 0:
                position["pnl"] = round((position["last_price"] - position["average_price"]) * position["quantity"], 2)
            else:
                position["pnl"] = round((position["average_price"] - position["last_price"]) * abs(position["quantity"]), 2)
        
        return self.mock_data["positions"]
    
    def _place_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock order placement."""
        self.order_counter += 1
        order_id = f"23122900{self.order_counter}"
        
        order = {
            "order_id": order_id,
            "exchange": args["exchange"],
            "tradingsymbol": args["symbol"],
            "transaction_type": args["transaction_type"],
            "quantity": args["quantity"],
            "product": args["product"],
            "order_type": args["order_type"],
            "price": args.get("price", 0),
            "trigger_price": args.get("trigger_price", 0),
            "status": "OPEN",
            "status_message": None,
            "placed_by": "AA0001",
            "order_timestamp": datetime.now().isoformat(),
            "exchange_order_id": None,
            "exchange_timestamp": None,
            "variety": "regular",
            "validity": "DAY",
            "tag": None,
            "tags": [],
            "pending_quantity": args["quantity"],
            "filled_quantity": 0,
            "average_price": 0
        }
        
        self.mock_data["orders"].append(order)
        
        return {"order_id": order_id}
    
    def _modify_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock order modification."""
        order_id = args["order_id"]
        
        # Find the order
        for order in self.mock_data["orders"]:
            if order["order_id"] == order_id:
                if "quantity" in args:
                    order["quantity"] = args["quantity"]
                    order["pending_quantity"] = args["quantity"] - order["filled_quantity"]
                if "price" in args:
                    order["price"] = args["price"]
                if "trigger_price" in args:
                    order["trigger_price"] = args["trigger_price"]
                
                return {"order_id": order_id}
        
        raise Exception(f"Order not found: {order_id}")
    
    def _cancel_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock order cancellation."""
        order_id = args["order_id"]
        
        # Find and cancel the order
        for order in self.mock_data["orders"]:
            if order["order_id"] == order_id:
                order["status"] = "CANCELLED"
                order["status_message"] = "Order cancelled by user"
                return {"order_id": order_id}
        
        raise Exception(f"Order not found: {order_id}")
    
    def _get_orders(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock orders list."""
        return self.mock_data["orders"]
    
    def _get_trades(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock trades list."""
        trades = []
        
        # Generate some trades from completed orders
        for order in self.mock_data["orders"]:
            if order["filled_quantity"] > 0:
                trades.append({
                    "trade_id": f"T{order['order_id'][-6:]}",
                    "order_id": order["order_id"],
                    "exchange": order["exchange"],
                    "tradingsymbol": order["tradingsymbol"],
                    "transaction_type": order["transaction_type"],
                    "quantity": order["filled_quantity"],
                    "price": order["average_price"],
                    "product": order["product"],
                    "fill_timestamp": order["exchange_timestamp"] or datetime.now().isoformat(),
                    "exchange_order_id": order["exchange_order_id"],
                    "exchange_timestamp": order["exchange_timestamp"]
                })
        
        return trades
    
    def _get_quotes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock market quote."""
        key = f"{args['exchange']}:{args['symbol']}"
        
        if key in self.mock_data["quotes"]:
            quote = self.mock_data["quotes"][key].copy()
            # Update with random changes
            change = random.uniform(-0.5, 0.5) * quote["last_price"] / 100
            quote["last_price"] = round(quote["last_price"] + change, 2)
            quote["timestamp"] = datetime.now().isoformat()
            quote["last_trade_time"] = datetime.now().isoformat()
            return quote
        else:
            # Generate mock quote for unknown symbols
            return {
                "instrument_token": str(random.randint(100000, 999999)),
                "timestamp": datetime.now().isoformat(),
                "last_price": round(random.uniform(100, 5000), 2),
                "last_quantity": random.randint(1, 1000),
                "buy_quantity": random.randint(10000, 100000),
                "sell_quantity": random.randint(10000, 100000),
                "volume": random.randint(100000, 10000000),
                "average_price": round(random.uniform(100, 5000), 2),
                "ohlc": {
                    "open": round(random.uniform(100, 5000), 2),
                    "high": round(random.uniform(100, 5000), 2),
                    "low": round(random.uniform(100, 5000), 2),
                    "close": round(random.uniform(100, 5000), 2)
                }
            }
    
    def _get_ltp(self, args: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Mock last traded prices."""
        ltps = {}
        
        for instrument in args["instruments"]:
            if instrument in self.mock_data["quotes"]:
                ltps[instrument] = {
                    "last_price": self.mock_data["quotes"][instrument]["last_price"]
                }
            else:
                ltps[instrument] = {
                    "last_price": round(random.uniform(100, 5000), 2)
                }
        
        return ltps
    
    def _get_historical_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock historical data."""
        # Generate random candles
        candles = []
        base_price = 1000 + random.uniform(0, 4000)
        
        from_date = datetime.fromisoformat(args["from_date"])
        to_date = datetime.fromisoformat(args["to_date"])
        interval = args["interval"]
        
        # Determine time delta based on interval
        if interval == "minute":
            delta = timedelta(minutes=1)
        elif interval == "5minute":
            delta = timedelta(minutes=5)
        elif interval == "15minute":
            delta = timedelta(minutes=15)
        else:  # day
            delta = timedelta(days=1)
        
        current = from_date
        while current <= to_date:
            # Generate OHLC data
            open_price = base_price + random.uniform(-10, 10)
            high = open_price + random.uniform(0, 20)
            low = open_price - random.uniform(0, 20)
            close = random.uniform(low, high)
            volume = random.randint(10000, 1000000)
            
            candles.append({
                "date": current.isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume
            })
            
            base_price = close
            current += delta
        
        return {
            "candles": candles,
            "instrument_token": args["instrument_token"]
        }
    
    def _get_margins(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock margin data."""
        segment = args.get("segment", "equity")
        
        if segment in self.mock_data["margins"]:
            # Update with small random changes
            margins = self.mock_data["margins"][segment].copy()
            margins["available"]["live_balance"] += random.uniform(-1000, 1000)
            return margins
        else:
            return self.mock_data["margins"]["equity"]
    
    def _search_instruments(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock instruments list."""
        exchange = args.get("exchange")
        
        if exchange:
            return [inst for inst in self.mock_data["instruments"] if inst["exchange"] == exchange]
        else:
            return self.mock_data["instruments"]
    
    def _login(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock login response."""
        api_key = args.get("api_key")
        api_secret = args.get("api_secret")
        
        return {
            "status": "success",
            "auth_url": f"https://kite.trade/connect/login?api_key={api_key}",
            "message": "Please visit the URL to complete authentication"
        }
    
    def _get_ohlc(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock OHLC data."""
        instruments = args.get("instruments", [])
        ohlc_data = {}
        
        for instrument in instruments:
            base_price = 1000 + random.uniform(0, 4000)
            ohlc_data[instrument] = {
                "instrument_token": str(random.randint(100000, 999999)),
                "last_price": round(base_price, 2),
                "ohlc": {
                    "open": round(base_price - random.uniform(0, 50), 2),
                    "high": round(base_price + random.uniform(0, 100), 2),
                    "low": round(base_price - random.uniform(0, 100), 2),
                    "close": round(base_price + random.uniform(-20, 20), 2)
                }
            }
        
        return ohlc_data
    
    def _get_profile(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock user profile."""
        return {
            "user_id": "AA0001",
            "user_name": "Test User",
            "user_shortname": "TEST",
            "email": "test@example.com",
            "user_type": "individual",
            "broker": "ZERODHA",
            "exchanges": ["NSE", "BSE", "NFO", "CDS", "MCX"],
            "products": ["CNC", "MIS", "NRML"],
            "order_types": ["MARKET", "LIMIT", "SL", "SL-M"]
        }
    
    def _get_mf_holdings(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock mutual fund holdings."""
        return [
            {
                "folio": "123456/78",
                "fund": "Axis Bluechip Fund",
                "tradingsymbol": "INF846K01EW2",
                "quantity": 150.234,
                "average_price": 45.67,
                "last_price": 48.23,
                "pnl": 384.45
            },
            {
                "folio": "987654/32",
                "fund": "HDFC Mid-Cap Opportunities",
                "tradingsymbol": "INF179K01AX8",
                "quantity": 200.567,
                "average_price": 89.12,
                "last_price": 92.45,
                "pnl": 667.89
            }
        ]
    
    def _get_order_history(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock order history."""
        order_id = args.get("order_id")
        
        return [
            {
                "order_id": order_id,
                "status": "OPEN",
                "status_message": "Order placed",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
            },
            {
                "order_id": order_id,
                "status": "COMPLETE",
                "status_message": "Order executed",
                "timestamp": datetime.now().isoformat()
            }
        ]
    
    def _get_order_trades(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock order trades."""
        order_id = args.get("order_id")
        
        return [
            {
                "trade_id": f"T{order_id[-6:]}",
                "order_id": order_id,
                "exchange": "NSE",
                "tradingsymbol": "RELIANCE",
                "quantity": 10,
                "price": 2520.50,
                "product": "CNC",
                "fill_timestamp": datetime.now().isoformat()
            }
        ]
    
    def _get_gtts(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock GTT orders list."""
        return [
            {
                "id": "GTT123",
                "trigger_type": "single",
                "tradingsymbol": "RELIANCE",
                "exchange": "NSE",
                "trigger_values": [2500],
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=365)).isoformat(),
                "meta": {},
                "orders": [
                    {
                        "transaction_type": "BUY",
                        "quantity": 10,
                        "product": "CNC",
                        "order_type": "LIMIT",
                        "price": 2500
                    }
                ]
            }
        ]
    
    def _place_gtt_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock GTT order placement."""
        trigger_id = f"GTT{random.randint(1000, 9999)}"
        
        return {
            "trigger_id": trigger_id,
            "status": "success",
            "message": "GTT order placed successfully"
        }
    
    def _modify_gtt_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock GTT order modification."""
        trigger_id = args.get("trigger_id")
        
        return {
            "trigger_id": trigger_id,
            "status": "success",
            "message": "GTT order modified successfully"
        }
    
    def _delete_gtt_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock GTT order deletion."""
        trigger_id = args.get("trigger_id")
        
        return {
            "trigger_id": trigger_id,
            "status": "success",
            "message": "GTT order deleted successfully"
        }
    
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