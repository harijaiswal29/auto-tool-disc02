# WebSocket API Specification

## Overview

The WebSocket API provides real-time communication for streaming results, execution updates, and system notifications.

## Connection

**Endpoint:** `ws://localhost:8000/ws`

**Authentication:** Include token in connection URL:
```
ws://localhost:8000/ws?token=your-jwt-token
```

## Message Format

All messages use JSON format with a `type` field indicating the message type.

## Client → Server Messages

### Subscribe to Channels
```json
{
  "type": "subscribe",
  "channels": ["executions", "metrics", "alerts"],
  "id": "sub-123"
}
```

### Streaming Query
```json
{
  "type": "query",
  "id": "query-123",
  "data": {
    "query": "Find all Python files modified today",
    "stream": true
  }
}
```

### Unsubscribe
```json
{
  "type": "unsubscribe",
  "channels": ["metrics"],
  "id": "unsub-456"
}
```

### Ping/Keep-Alive
```json
{
  "type": "ping",
  "timestamp": 1642291200
}
```

## Server → Client Messages

### Execution Updates
```json
{
  "type": "execution_update",
  "execution_id": "exec-456",
  "status": "in_progress",
  "progress": 0.75,
  "current_tool": "filesystem_mcp",
  "message": "Scanning directories..."
}
```

### Result Chunks
```json
{
  "type": "result_chunk",
  "query_id": "query-123",
  "chunk_index": 1,
  "total_chunks": 3,
  "chunk": {
    "tool": "filesystem_mcp",
    "data": ["file1.py", "file2.py"]
  }
}
```

### Tool Status Updates
```json
{
  "type": "tool_status",
  "tool_id": "git_mcp",
  "status": "available",
  "performance": {
    "avg_response_time_ms": 120,
    "success_rate": 0.98
  }
}
```

### Metric Updates
```json
{
  "type": "metrics",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "active_executions": 5,
    "queue_length": 12,
    "avg_wait_time_ms": 50
  }
}
```

### Alerts
```json
{
  "type": "alert",
  "level": "warning",
  "alert_id": "alert-789",
  "message": "High memory usage detected",
  "details": {
    "memory_percent": 85,
    "threshold": 80
  }
}
```

### Error Messages
```json
{
  "type": "error",
  "error_code": "INVALID_MESSAGE",
  "message": "Invalid message format",
  "request_id": "query-123"
}
```

### Acknowledgments
```json
{
  "type": "ack",
  "request_id": "sub-123",
  "status": "success"
}
```

### Pong Response
```json
{
  "type": "pong",
  "timestamp": 1642291201
}
```

## Channel Descriptions

### executions
Real-time updates about tool executions:
- Execution start/complete
- Progress updates
- Performance metrics

### metrics
System performance metrics:
- Resource usage
- Queue statistics
- Tool performance

### alerts
System alerts and notifications:
- Error conditions
- Performance warnings
- System status changes

### results
Streaming query results:
- Partial results as available
- Tool-by-tool results
- Final aggregated results

## Connection Management

### Heartbeat
- Client should send ping every 30 seconds
- Server responds with pong
- Connection closed after 60 seconds of inactivity

### Reconnection
- Client should implement exponential backoff
- Server maintains 5-minute session cache
- Resume subscriptions after reconnection

### Rate Limiting
- Max 100 messages/minute per connection
- Exceeded limit results in warning
- Persistent violations result in disconnection

## Error Handling

### Connection Errors
- `1000` - Normal closure
- `1001` - Going away
- `1002` - Protocol error
- `1003` - Unsupported data
- `1006` - Abnormal closure
- `1008` - Policy violation
- `1011` - Internal server error

### Message Errors
- Invalid JSON format
- Missing required fields
- Unknown message type
- Authentication failure
- Rate limit exceeded

## Best Practices

1. **Message IDs**: Include unique IDs for request tracking
2. **Buffering**: Implement client-side message buffering
3. **Compression**: Enable WebSocket compression
4. **Error Recovery**: Implement automatic reconnection
5. **State Management**: Track subscription state locally

## Example Session

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8000/ws?token=jwt-token');

// Subscribe to channels
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['executions', 'metrics'],
  id: 'sub-001'
}));

// Send streaming query
ws.send(JSON.stringify({
  type: 'query',
  id: 'query-001',
  data: {
    query: 'Analyze Python code quality',
    stream: true
  }
}));

// Handle messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.type) {
    case 'execution_update':
      updateProgress(message);
      break;
    case 'result_chunk':
      appendResult(message);
      break;
    case 'error':
      handleError(message);
      break;
  }
};
```