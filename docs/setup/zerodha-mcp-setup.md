# Zerodha MCP Setup Guide

## Overview

The Zerodha MCP integration enables trading and portfolio management operations through the Model Context Protocol. This guide covers setup, configuration, and usage of the Zerodha MCP client.

## Features

The Zerodha MCP client provides access to:

### Portfolio Management
- **Holdings**: View equity holdings with P&L
- **Positions**: Track intraday and overnight positions

### Order Management
- **Place Orders**: Buy/sell orders with various types (MARKET, LIMIT, SL, SL-M)
- **Modify Orders**: Change price, quantity, or trigger price
- **Cancel Orders**: Cancel pending orders
- **Order Book**: View all orders for the day
- **Trade Book**: View executed trades

### Market Data
- **Quotes**: Full market depth with OHLC data
- **LTP**: Last traded prices for multiple instruments
- **Historical Data**: Candle data for various intervals

### Account Information
- **Margins**: Available funds and margins
- **Instruments**: List of tradeable instruments

## Prerequisites

1. **Zerodha Account**: Active trading account with Kite Connect API access
2. **API Credentials**:
   - API Key from Kite Connect dashboard
   - API Secret for authentication
   - Access Token (obtained via OAuth flow)

## Setup Instructions

### 1. Obtain API Credentials

1. Login to [Kite Connect Dashboard](https://developers.kite.trade/)
2. Create a new app or use existing one
3. Note down your API Key and API Secret
4. Generate access token via login flow

### 2. Environment Configuration

Set the following environment variables:

```bash
# Required credentials
export ZERODHA_API_KEY="your_api_key"
export ZERODHA_API_SECRET="your_api_secret"
export ZERODHA_ACCESS_TOKEN="your_access_token"

# Optional: Use sandbox for testing
export ZERODHA_USE_SANDBOX=false
export ZERODHA_SANDBOX_ENDPOINT="https://sandbox.kite.trade/mcp/v1"
```

### 3. Configuration File

The Zerodha MCP is configured in `config/config.json`:

```json
{
  "mcp_servers": {
    "zerodha": {
      "type": "remote",
      "endpoint": "https://api.kite.trade/mcp/v1",
      "auth_type": "oauth",
      "enabled": true,
      "env": {
        "ZERODHA_API_KEY": "${ZERODHA_API_KEY}",
        "ZERODHA_API_SECRET": "${ZERODHA_API_SECRET}",
        "ZERODHA_ACCESS_TOKEN": "${ZERODHA_ACCESS_TOKEN}"
      },
      "config": {
        "use_sandbox": false,
        "sandbox_endpoint": "https://sandbox.kite.trade/mcp/v1",
        "cache_ttl": 60,
        "session_expiry": "daily"
      }
    }
  }
}
```

### 4. Access Token Generation

Access tokens expire daily at 8:00 AM IST. To generate a new token:

```python
import requests
from urllib.parse import urlencode

# Step 1: Get request token via login
login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={API_KEY}"
print(f"Login URL: {login_url}")
# User logs in and you get request_token

# Step 2: Exchange request token for access token
data = {
    "api_key": API_KEY,
    "request_token": REQUEST_TOKEN,
    "checksum": generate_checksum(API_KEY, REQUEST_TOKEN, API_SECRET)
}

response = requests.post("https://api.kite.trade/session/token", data=data)
access_token = response.json()["data"]["access_token"]
```

## Usage Examples

### Basic Usage

```python
from src.tools.zerodha_mcp import ZerodhaMCPClient

# Initialize client
client = ZerodhaMCPClient()

# Connect to server
await client.connect(use_mock=False)  # Use real server

# Get holdings
holdings = await client.get_holdings()
for holding in holdings:
    print(f"{holding['tradingsymbol']}: {holding['quantity']} @ {holding['average_price']}")

# Place an order
order = await client.place_order(
    exchange="NSE",
    symbol="RELIANCE",
    transaction_type="BUY",
    quantity=10,
    product="CNC",  # Cash & Carry for delivery
    order_type="LIMIT",
    price=2500.00
)
print(f"Order placed: {order['order_id']}")

# Get quote
quote = await client.get_quote("NSE", "RELIANCE")
print(f"LTP: {quote['last_price']}, Volume: {quote['volume']}")
```

### Using Mock Server

For development and testing without API credentials:

```python
# Connect to mock server
await client.connect(use_mock=True)

# All operations work the same way
holdings = await client.get_holdings()
```

## Product Types

- **CNC**: Cash & Carry (for equity delivery)
- **MIS**: Margin Intraday Square-off
- **NRML**: Normal (for F&O overnight positions)

## Order Types

- **MARKET**: Market order at current price
- **LIMIT**: Limit order at specified price
- **SL**: Stop-loss order (requires trigger_price)
- **SL-M**: Stop-loss market order (requires trigger_price)

## Testing

Run integration tests:

```bash
# Run all Zerodha tests
pytest tests/integration/test_zerodha_mcp.py -v

# Run specific test
pytest tests/integration/test_zerodha_mcp.py::TestZerodhaMCPIntegration::test_place_order -v

# Run with coverage
pytest tests/integration/test_zerodha_mcp.py --cov=src.tools.zerodha_mcp
```

Run demo script:

```bash
# Test with mock data
python src/tools/zerodha_mcp.py
```

## Rate Limits

Zerodha API has the following rate limits:
- Order placement: 10 per second
- Other requests: 10 per second
- Historical data: 3 per second

The client implements caching to minimize API calls:
- Market data cached for 60 seconds
- Margins cached for 60 seconds

## Error Handling

Common errors and solutions:

1. **Token Expired**: Access tokens expire daily. Generate a new token.
2. **Invalid Symbol**: Ensure trading symbol is in correct format (e.g., "RELIANCE" not "RELIANCE.NS")
3. **Insufficient Funds**: Check margins before placing orders
4. **Market Closed**: Orders can only be placed during market hours

## Security Notes

1. **Never commit credentials**: Use environment variables
2. **Token rotation**: Implement daily token refresh
3. **Secure storage**: Store API secret securely
4. **IP whitelisting**: Configure allowed IPs in Kite Connect dashboard

## Troubleshooting

### Connection Issues
```python
# Enable debug logging
import logging
logging.getLogger('src.tools.zerodha_mcp').setLevel(logging.DEBUG)
```

### Mock Server Testing
```python
# Test specific scenarios with mock
client = ZerodhaMCPClient()
await client.connect(use_mock=True)

# Mock server provides realistic test data
holdings = await client.get_holdings()
assert len(holdings) > 0
```

## Integration with Tool Registry

The Zerodha MCP automatically registers its tools:

```python
from src.core.tool_registry import ToolRegistry

registry = ToolRegistry()
client = ZerodhaMCPClient()
await client.connect()

# Register tools
client.register_tools_to_registry(registry)

# Tools are now available for discovery
tools = registry.search_by_domain("finance")
```

## Advanced Configuration

### Custom Endpoints

For enterprise or custom deployments:

```python
client = ZerodhaMCPClient(
    endpoint="https://custom.broker.com/mcp/v1",
    api_key="custom_key",
    access_token="custom_token"
)
```

### Session Management

Implement automatic token refresh:

```python
import asyncio
from datetime import datetime, time

async def refresh_token_daily():
    while True:
        now = datetime.now()
        # Refresh at 8:30 AM IST
        target = now.replace(hour=8, minute=30, second=0)
        if now >= target:
            target += timedelta(days=1)
        
        wait_seconds = (target - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        
        # Refresh token logic here
        new_token = await generate_new_access_token()
        client.access_token = new_token
```

## Support

For issues related to:
- **Zerodha API**: Contact Kite Connect support
- **MCP Integration**: Check project documentation
- **Mock Server**: File issues in project repository