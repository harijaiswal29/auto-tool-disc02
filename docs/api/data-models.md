# Data Models & Schemas

## Core Data Models

### User Model
```json
{
  "id": "user_123",
  "username": "john_doe",
  "email": "john@example.com",
  "role": "developer",
  "preferences": {
    "domain": "engineering",
    "exploration_level": "medium",
    "feedback_frequency": "always"
  },
  "created_at": "2024-01-01T00:00:00Z",
  "last_active": "2024-01-15T10:00:00Z"
}
```

### Execution Model
```json
{
  "id": "exec_456",
  "user_id": "user_123",
  "session_id": "session_789",
  "query": {
    "raw": "Find and analyze Python files",
    "normalized": "find analyze python files",
    "timestamp": "2024-01-15T10:00:00Z"
  },
  "intent": {
    "type": "query.search",
    "confidence": 0.85,
    "entities": ["Python", "files"]
  },
  "tools_used": [
    {
      "tool_id": "filesystem_mcp",
      "start_time": "2024-01-15T10:00:01Z",
      "end_time": "2024-01-15T10:00:03Z",
      "status": "success",
      "result_summary": "Found 42 Python files"
    }
  ],
  "learning_data": {
    "state_vector": [0.1, 0.2, ...],
    "action_taken": ["filesystem_mcp", "code_analyzer"],
    "reward": 0.8
  },
  "metrics": {
    "total_time_ms": 3000,
    "tools_invoked": 2,
    "success_rate": 1.0
  }
}
```

### Pattern Model
```json
{
  "id": "pattern_001",
  "type": "sequential",
  "tools": ["git_mcp", "filesystem_mcp", "code_analyzer"],
  "support": 0.15,
  "confidence": 0.85,
  "lift": 2.3,
  "contexts": ["code_review", "bug_fix"],
  "discovered_at": "2024-01-10T00:00:00Z",
  "usage_count": 45
}
```

### Tool Model
```json
{
  "id": "filesystem_mcp",
  "name": "Filesystem MCP",
  "type": "mcp",
  "endpoint": "stdio://mcp-server-filesystem",
  "version": "1.0.0",
  "capabilities": {
    "operations": [
      {
        "name": "read_file",
        "category": "file_io",
        "parameters": {
          "path": { "type": "string", "required": true }
        }
      }
    ],
    "constraints": {
      "max_file_size_mb": 100
    },
    "semantic_tags": ["filesystem", "io"]
  },
  "performance": {
    "avg_response_time_ms": 50,
    "success_rate": 0.98,
    "usage_count": 1250
  },
  "status": {
    "available": true,
    "last_health_check": "2024-01-15T10:00:00Z"
  }
}
```

## Event Schemas

### Tool Discovery Event
```json
{
  "event_type": "tool_discovery",
  "timestamp": "2024-01-15T10:00:00Z",
  "data": {
    "intent_id": "intent_123",
    "discovered_tools": ["tool1", "tool2"],
    "discovery_method": "semantic_search",
    "discovery_time_ms": 150
  }
}
```

### Learning Update Event
```json
{
  "event_type": "learning_update",
  "timestamp": "2024-01-15T10:05:00Z",
  "data": {
    "execution_id": "exec_456",
    "q_table_update": {
      "state": "encoded_state_vector",
      "action": "tool_combination",
      "old_q_value": 0.5,
      "new_q_value": 0.7,
      "reward": 0.8
    },
    "pattern_mined": "pattern_001"
  }
}
```

### System Event
```json
{
  "event_type": "system",
  "subtype": "tool_registered",
  "timestamp": "2024-01-15T09:00:00Z",
  "data": {
    "tool_id": "new_tool_mcp",
    "registered_by": "admin",
    "capabilities_count": 5
  }
}
```

## Metrics Schema

### Performance Metric
```json
{
  "metric_type": "performance",
  "timestamp": "2024-01-15T10:00:00Z",
  "dimensions": {
    "tool_id": "filesystem_mcp",
    "domain": "engineering",
    "user_type": "developer"
  },
  "values": {
    "response_time_ms": 250,
    "success_rate": 0.95,
    "usage_count": 1000,
    "error_rate": 0.05
  }
}
```

### Learning Metric
```json
{
  "metric_type": "learning",
  "timestamp": "2024-01-15T10:00:00Z",
  "values": {
    "cumulative_reward": 125.5,
    "exploration_rate": 0.2,
    "q_table_size": 1500,
    "patterns_discovered": 42,
    "convergence_score": 0.78
  }
}
```

### Intent Recognition Metric
```json
{
  "metric_type": "intent_recognition",
  "timestamp": "2024-01-15T10:00:00Z",
  "dimensions": {
    "agent_id": "intent_recognition_agent",
    "pipeline_version": "1.0.0"
  },
  "values": {
    "processing_time_ms": 45.2,
    "confidence_score": 0.85,
    "cache_hit": true,
    "intent_type": "query.search",
    "pipeline_stage_timings": {
      "text_preprocessor": 2.1,
      "tokenizer": 1.5,
      "feature_extractor": 25.3,
      "intent_classifier": 8.2,
      "context_enricher": 4.1,
      "confidence_scorer": 3.0,
      "state_manager": 1.0
    },
    "features": {
      "word_count": 5,
      "has_question": true,
      "multi_intent": false
    }
  }
}
```

## Request/Response Schemas

### Query Request
```json
{
  "query": "string",
  "context": {
    "session_id": "string",
    "domain": "string",
    "history": ["previous_query1", "previous_query2"]
  },
  "options": {
    "stream": false,
    "timeout": 30000,
    "max_tools": 5
  }
}
```

### Query Response
```json
{
  "query_id": "string",
  "status": "success",
  "intent": {
    "type": "string",
    "confidence": 0.85
  },
  "results": {
    "data": {},
    "metadata": {
      "tools_used": ["tool1", "tool2"],
      "execution_time_ms": 250,
      "cache_hit": false
    }
  }
}
```

### Tool Registration Request
```json
{
  "name": "string",
  "type": "mcp|api|local",
  "endpoint": "string",
  "authentication": {
    "type": "none|api_key|oauth",
    "config": {}
  },
  "capabilities": {
    "operations": [],
    "constraints": {},
    "semantic_tags": []
  }
}
```

## Validation Rules

### Common Validations
- IDs: UUID v4 format
- Timestamps: ISO 8601 format
- Scores: Float between 0.0 and 1.0
- Counts: Non-negative integers

### Field Constraints
- Username: 3-30 characters, alphanumeric
- Email: Valid email format
- Query: 1-500 characters
- Tool name: 3-50 characters
- Confidence: 0.0-1.0

## Enumerations

### User Roles
- `admin`
- `developer`
- `analyst`
- `viewer`

### Tool Types
- `mcp` - Model Context Protocol
- `api` - REST API
- `local` - Local executable

### Intent Types
- `query.search`
- `query.retrieve`
- `query.analyze`
- `action.create`
- `action.modify`
- `action.delete`
- `system.configure`
- `system.monitor`

### Execution Status
- `pending`
- `in_progress`
- `success`
- `failure`
- `timeout`
- `cancelled`

### Feedback Types
- `positive`
- `negative`
- `neutral`