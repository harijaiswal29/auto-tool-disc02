# RESTful API Specification

## Overview

The Auto Tool Discovery API provides RESTful endpoints for query processing, tool management, learning feedback, and system metrics.

## Base URL

```
http://localhost:8000/api/v1
```

## Core Endpoints

### Query Processing

#### POST /api/v1/query
Process a natural language query and execute appropriate tools.

**Request Body:**
```json
{
  "query": "Find all Python files modified today",
  "context": {
    "domain": "engineering",
    "session_id": "session_123"
  }
}
```

**Response:**
```json
{
  "query_id": "query_456",
  "intent": {
    "type": "query.search",
    "confidence": 0.85
  },
  "tools_executed": ["filesystem_mcp", "git_mcp"],
  "results": {
    "files": ["main.py", "utils.py"],
    "count": 2
  },
  "execution_time_ms": 250
}
```

### Tool Management

#### GET /api/v1/tools
List available tools with optional filtering.

**Query Parameters:**
- `category` (optional): Filter by tool category
- `domain` (optional): Filter by domain
- `active` (optional): Filter by active status

**Response:**
```json
{
  "tools": [
    {
      "id": "filesystem_mcp",
      "name": "Filesystem MCP",
      "type": "mcp",
      "capabilities": ["read", "write", "search"],
      "performance_score": 0.95,
      "available": true
    }
  ],
  "total": 15
}
```

#### POST /api/v1/tools
Register a new tool.

**Request Body:**
```json
{
  "name": "Custom Tool",
  "type": "api",
  "endpoint": "http://localhost:9000",
  "capabilities": {
    "operations": ["process", "analyze"]
  }
}
```

#### POST /api/v1/tools/{tool_id}/execute
Execute a specific tool directly.

**Request Body:**
```json
{
  "parameters": {
    "path": "/home/user/project"
  },
  "context": {
    "user_id": "user_123"
  }
}
```

### Learning Feedback

#### POST /api/v1/learning/feedback
Submit feedback for learning system improvement.

**Request Body:**
```json
{
  "execution_id": "exec_789",
  "feedback_type": "positive",
  "details": {
    "helpful": true,
    "accurate": true,
    "comment": "Found exactly what I needed"
  }
}
```

### System Metrics

#### GET /api/v1/metrics
Retrieve system performance metrics.

**Query Parameters:**
- `start_time`: ISO 8601 timestamp
- `end_time`: ISO 8601 timestamp
- `metric_type`: Type of metrics (performance, learning, usage)

**Response:**
```json
{
  "metrics": {
    "avg_response_time_ms": 180,
    "success_rate": 0.94,
    "tool_usage": {
      "filesystem_mcp": 450,
      "git_mcp": 320
    },
    "learning_progress": {
      "q_table_size": 1250,
      "patterns_discovered": 42
    }
  },
  "period": {
    "start": "2024-01-15T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  }
}
```

## Authentication

### API Key Authentication
Include API key in header:
```
X-API-Key: your-api-key-here
```

### JWT Authentication
Include JWT token in Authorization header:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

## Rate Limiting

- Default: 100 requests/minute
- Premium: 1000 requests/minute
- Tool execution: 50 executions/minute

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642291200
```

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "TOOL_NOT_FOUND",
    "message": "The requested tool 'unknown_tool' was not found",
    "details": {
      "tool_id": "unknown_tool",
      "available_tools": ["tool1", "tool2"]
    },
    "trace_id": "abc123-def456-ghi789",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
- `503` - Service Unavailable

## Pagination

For endpoints returning lists:
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "total_pages": 5
  },
  "links": {
    "first": "/api/v1/tools?page=1",
    "last": "/api/v1/tools?page=5",
    "next": "/api/v1/tools?page=2",
    "prev": null
  }
}
```

## Versioning

- Current version: v1
- Version in URL path: `/api/v1/`
- Deprecation notice in header: `X-API-Deprecated: true`
- Sunset date in header: `X-API-Sunset: 2024-12-31`

## CORS Configuration

Allowed origins:
- `http://localhost:*`
- `https://*.yourdomain.com`

Allowed methods:
- GET, POST, PUT, DELETE, OPTIONS

Allowed headers:
- Content-Type, Authorization, X-API-Key