# Database Schema Design

## Overview

The system uses SQLite with aiosqlite for async operations. The database stores tool registry information, relationships, capabilities, and learning data.

## Core Tables

### Tools Table
```sql
CREATE TABLE tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    capabilities JSON,
    metadata JSON,
    performance_score REAL DEFAULT 0.5,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Tool Relationships Table
```sql
CREATE TABLE tool_relationships (
    tool1_id TEXT,
    tool2_id TEXT,
    relationship_type TEXT,
    strength REAL,
    FOREIGN KEY (tool1_id) REFERENCES tools(id),
    FOREIGN KEY (tool2_id) REFERENCES tools(id)
);
```

## Enhanced Tool Registry Tables

### Main Tools Table (Extended)
```sql
CREATE TABLE tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL CHECK (type IN ('mcp', 'api', 'local')),
    endpoint TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0.0',
    capabilities JSON NOT NULL,
    metadata JSON,
    performance_score REAL DEFAULT 0.5 CHECK (performance_score BETWEEN 0 AND 1),
    availability_score REAL DEFAULT 1.0 CHECK (availability_score BETWEEN 0 AND 1),
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_health_check TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (type) REFERENCES tool_types(name)
);
```

### Tool Relationships (Extended)
```sql
CREATE TABLE tool_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool1_id TEXT NOT NULL,
    tool2_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL CHECK (relationship_type IN ('complements', 'requires', 'conflicts', 'enhances')),
    strength REAL DEFAULT 0.5 CHECK (strength BETWEEN 0 AND 1),
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evidence_count INTEGER DEFAULT 1,
    FOREIGN KEY (tool1_id) REFERENCES tools(id) ON DELETE CASCADE,
    FOREIGN KEY (tool2_id) REFERENCES tools(id) ON DELETE CASCADE,
    UNIQUE(tool1_id, tool2_id, relationship_type)
);
```

### Tool Capabilities Detail
```sql
CREATE TABLE tool_capabilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_id TEXT NOT NULL,
    capability_name TEXT NOT NULL,
    capability_type TEXT NOT NULL,
    parameters JSON,
    constraints JSON,
    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE,
    UNIQUE(tool_id, capability_name)
);
```

## Performance Indexes

```sql
-- Tools table indexes
CREATE INDEX idx_tools_type ON tools(type);
CREATE INDEX idx_tools_performance ON tools(performance_score DESC);
CREATE INDEX idx_tools_active ON tools(is_active);

-- Relationships table indexes
CREATE INDEX idx_relationships_tool1 ON tool_relationships(tool1_id);
CREATE INDEX idx_relationships_tool2 ON tool_relationships(tool2_id);

-- Capabilities table indexes
CREATE INDEX idx_capabilities_tool ON tool_capabilities(tool_id);
```

## Learning System Tables

### Q-Learning State
```sql
CREATE TABLE q_learning_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_hash TEXT NOT NULL UNIQUE,
    state_vector JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE q_values (
    state_id INTEGER NOT NULL,
    action_hash TEXT NOT NULL,
    action_tools JSON NOT NULL,
    q_value REAL NOT NULL DEFAULT 0.0,
    update_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (state_id) REFERENCES q_learning_states(id),
    UNIQUE(state_id, action_hash)
);
```

### Pattern Storage
```sql
CREATE TABLE discovered_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,
    tool_sequence JSON NOT NULL,
    support REAL NOT NULL,
    confidence REAL NOT NULL,
    lift REAL,
    contexts JSON,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    usage_count INTEGER DEFAULT 0
);
```

### Execution History
```sql
CREATE TABLE execution_history (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    query TEXT NOT NULL,
    intent JSON NOT NULL,
    tools_used JSON NOT NULL,
    execution_time_ms INTEGER,
    success BOOLEAN,
    reward REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Metrics Tables

### Performance Metrics
```sql
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_value REAL NOT NULL,
    dimensions JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tool_id) REFERENCES tools(id)
);

CREATE INDEX idx_metrics_tool_time ON performance_metrics(tool_id, timestamp DESC);
```

### Learning Metrics
```sql
CREATE TABLE learning_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    episode_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Data Types and Constraints

### JSON Field Structures

**Capabilities JSON**:
```json
{
  "operations": [
    {
      "name": "read_file",
      "category": "file_io",
      "parameters": {...},
      "returns": {...}
    }
  ],
  "constraints": {...},
  "semantic_tags": ["filesystem", "io"]
}
```

**State Vector JSON**:
```json
{
  "intent_embedding": [0.1, 0.2, ...],
  "context_features": {...},
  "tool_history": [...],
  "performance_metrics": {...}
}
```

## Migration Strategy

### Version Control
```sql
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Sample Migration
```sql
-- Migration: 001_add_tool_versioning
ALTER TABLE tools ADD COLUMN version TEXT NOT NULL DEFAULT '1.0.0';
INSERT INTO schema_migrations (version, name) VALUES (1, 'add_tool_versioning');
```

## Enhanced Learning System Tables

### Failure History Table
```sql
CREATE TABLE failure_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    recovery_successful BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES execution_history(id),
    FOREIGN KEY (tool_id) REFERENCES tools(id)
);

CREATE INDEX idx_failure_history_execution ON failure_history(execution_id);
CREATE INDEX idx_failure_history_tool ON failure_history(tool_id);
CREATE INDEX idx_failure_history_type ON failure_history(failure_type);
```

### Resource Metrics Table
```sql
CREATE TABLE resource_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    memory_mb REAL,
    cpu_percent REAL,
    api_calls INTEGER,
    execution_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES execution_history(id),
    FOREIGN KEY (tool_id) REFERENCES tools(id)
);

CREATE INDEX idx_resource_metrics_execution ON resource_metrics(execution_id);
CREATE INDEX idx_resource_metrics_tool_time ON resource_metrics(tool_id, created_at DESC);
```

### User Feedback Table
```sql
CREATE TABLE user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('positive', 'negative', 'neutral')),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    query_reformulated BOOLEAN DEFAULT FALSE,
    result_used BOOLEAN,
    follow_up_time_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES execution_history(id)
);

CREATE INDEX idx_user_feedback_execution ON user_feedback(execution_id);
CREATE INDEX idx_user_feedback_type ON user_feedback(feedback_type);
CREATE INDEX idx_user_feedback_rating ON user_feedback(rating);
```

### Tool Synergies Table
```sql
CREATE TABLE tool_synergies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_combination TEXT NOT NULL UNIQUE,
    success_rate REAL CHECK (success_rate BETWEEN 0 AND 1),
    occurrences INTEGER DEFAULT 0,
    synergy_score REAL CHECK (synergy_score BETWEEN -1 AND 1),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tool_synergies_score ON tool_synergies(synergy_score DESC);
CREATE INDEX idx_tool_synergies_success ON tool_synergies(success_rate DESC);
```

## Data Retention Policies

1. **Execution History**: Keep for 90 days
2. **Performance Metrics**: Aggregate after 30 days
3. **Q-Learning States**: Persist indefinitely
4. **Patterns**: Archive low-usage patterns after 60 days
5. **Failure History**: Keep for 180 days for pattern analysis
6. **Resource Metrics**: Aggregate after 30 days, keep summaries for 1 year
7. **User Feedback**: Persist indefinitely for continuous learning
8. **Tool Synergies**: Update continuously, archive unused combinations after 90 days

## Backup and Recovery

### Backup Strategy
- Daily full backups
- Hourly incremental backups
- 30-day retention period

### Recovery Procedures
1. Stop application
2. Restore from backup
3. Replay transaction log
4. Verify data integrity
5. Resume operations