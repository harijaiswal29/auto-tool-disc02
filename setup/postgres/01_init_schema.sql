-- Initialize Auto Tool Discovery Database Schema
-- This script sets up the PostgreSQL database for testing MCP integration

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS public;

-- Grant permissions
GRANT ALL ON SCHEMA public TO auto_tool_user;

-- Tools table
CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL CHECK (type IN ('mcp', 'api', 'local')),
    endpoint TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0.0',
    capabilities JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    performance_score REAL DEFAULT 0.5 CHECK (performance_score BETWEEN 0 AND 1),
    availability_score REAL DEFAULT 1.0 CHECK (availability_score BETWEEN 0 AND 1),
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_health_check TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Tool relationships table
CREATE TABLE IF NOT EXISTS tool_relationships (
    id SERIAL PRIMARY KEY,
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

-- Execution history table
CREATE TABLE IF NOT EXISTS execution_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    session_id TEXT,
    query TEXT NOT NULL,
    intent JSONB NOT NULL,
    tools_used JSONB NOT NULL,
    execution_time_ms INTEGER,
    success BOOLEAN,
    reward REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Q-learning states table
CREATE TABLE IF NOT EXISTS q_learning_states (
    id SERIAL PRIMARY KEY,
    state_hash TEXT NOT NULL UNIQUE,
    state_vector JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Q-values table
CREATE TABLE IF NOT EXISTS q_values (
    state_id INTEGER NOT NULL,
    action_hash TEXT NOT NULL,
    action_tools JSONB NOT NULL,
    q_value REAL NOT NULL DEFAULT 0.0,
    update_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (state_id) REFERENCES q_learning_states(id),
    UNIQUE(state_id, action_hash)
);

-- Discovered patterns table
CREATE TABLE IF NOT EXISTS discovered_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    tool_sequence JSONB NOT NULL,
    support REAL NOT NULL,
    confidence REAL NOT NULL,
    lift REAL,
    contexts JSONB,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    usage_count INTEGER DEFAULT 0
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    tool_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_value REAL NOT NULL,
    dimensions JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tool_id) REFERENCES tools(id)
);

-- Create indexes for better performance
CREATE INDEX idx_tools_type ON tools(type);
CREATE INDEX idx_tools_performance ON tools(performance_score DESC);
CREATE INDEX idx_tools_active ON tools(is_active);
CREATE INDEX idx_relationships_tool1 ON tool_relationships(tool1_id);
CREATE INDEX idx_relationships_tool2 ON tool_relationships(tool2_id);
CREATE INDEX idx_execution_history_created ON execution_history(created_at DESC);
CREATE INDEX idx_metrics_tool_time ON performance_metrics(tool_id, timestamp DESC);

-- Insert sample data for testing
INSERT INTO tools (id, name, type, endpoint, capabilities, metadata) VALUES
    ('filesystem_mcp', 'Filesystem MCP', 'mcp', 'stdio://mcp-server-filesystem', 
     '{"operations": ["read", "write", "list"], "constraints": {"max_file_size_mb": 100}}', 
     '{"description": "File system operations"}'),
    ('sqlite_mcp', 'SQLite MCP', 'mcp', 'stdio://mcp-server-sqlite',
     '{"operations": ["query", "schema"], "constraints": {"read_only": true}}',
     '{"description": "SQLite database operations"}'),
    ('postgres_mcp', 'PostgreSQL MCP', 'mcp', 'stdio://mcp-server-postgres',
     '{"operations": ["query"], "constraints": {"read_only": true}}',
     '{"description": "PostgreSQL database operations"}');

-- Insert sample relationships
INSERT INTO tool_relationships (tool1_id, tool2_id, relationship_type, strength) VALUES
    ('filesystem_mcp', 'sqlite_mcp', 'complements', 0.8),
    ('sqlite_mcp', 'postgres_mcp', 'enhances', 0.7);

-- Insert sample execution history
INSERT INTO execution_history (user_id, session_id, query, intent, tools_used, execution_time_ms, success, reward) VALUES
    ('user_001', 'session_001', 'Find all Python files', 
     '{"type": "query.search", "confidence": 0.85}',
     '["filesystem_mcp"]', 250, true, 0.8),
    ('user_001', 'session_002', 'Query database tables',
     '{"type": "query.retrieve", "confidence": 0.9}',
     '["postgres_mcp"]', 150, true, 0.9);

-- Grant necessary permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO auto_tool_user;
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO auto_tool_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO auto_tool_user;