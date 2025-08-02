"""
Tool Registry - Core component for managing MCP tools
Stores information about available tools and tracks their performance.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ToolRegistry:
    """
    Central registry for all discovered tools.
    
    Real-world analogy: Like a library catalog that keeps track of all
    available books (tools), who uses them, and how useful they are.
    """
    
    def __init__(self, db_path: str = "data/registry/tools.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"[INIT] Tool Registry initialized at {self.db_path}")
    
    async def initialize(self):
        """Async initialization method for compatibility."""
        # Database is already initialized in __init__
        pass
    
    async def close(self):
        """Close database connection."""
        # SQLite connections are closed after each operation
        pass
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tools table with enhanced failure tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    server_type TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    description TEXT,
                    capabilities TEXT,  -- JSON
                    input_schema TEXT,  -- JSON
                    performance_score REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    consecutive_failures INTEGER DEFAULT 0,
                    last_failure_time TIMESTAMP,
                    circuit_breaker_state TEXT DEFAULT 'closed',
                    circuit_breaker_opened_at TIMESTAMP,
                    avg_response_time_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tool relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool1_id TEXT NOT NULL,
                    tool2_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tool1_id) REFERENCES tools(id),
                    FOREIGN KEY (tool2_id) REFERENCES tools(id),
                    UNIQUE(tool1_id, tool2_id, relationship_type)
                )
            """)
            
            # Usage history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT NOT NULL,
                    task_type TEXT,
                    input_summary TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tool_id) REFERENCES tools(id)
                )
            """)
            
            # Retry metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retry_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    delay_ms REAL,
                    error_type TEXT,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tool_id) REFERENCES tools(id)
                )
            """)
            
            # Add indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_performance ON tools(performance_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_circuit_state ON tools(circuit_breaker_state)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_history_tool ON usage_history(tool_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_retry_metrics_tool ON retry_metrics(tool_id)")
            
            conn.commit()
            logger.info("[DB] Database schema initialized with retry support")
    
    def register_tool(self, tool_info: Dict[str, Any]) -> bool:
        """
        Register a new tool in the registry.
        
        Args:
            tool_info: Dictionary containing tool information
                - id: Unique identifier (e.g., "filesystem.read_file")
                - name: Human-readable name
                - server_type: Type of MCP server (e.g., "filesystem", "sqlite")
                - endpoint: Server endpoint or command
                - description: What the tool does
                - input_schema: JSON schema for inputs
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO tools 
                    (id, name, server_type, endpoint, description, capabilities, input_schema)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    tool_info['id'],
                    tool_info['name'],
                    tool_info['server_type'],
                    tool_info['endpoint'],
                    tool_info.get('description', ''),
                    json.dumps(tool_info.get('capabilities', {})),
                    json.dumps(tool_info.get('input_schema', {}))
                ))
                
                conn.commit()
                logger.info(f"[REGISTERED] Tool: {tool_info['id']}")
                return True
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to register tool: {e}")
            return False
    
    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
            row = cursor.fetchone()
            
            if row:
                tool = dict(row)
                tool['capabilities'] = json.loads(tool['capabilities'])
                tool['input_schema'] = json.loads(tool['input_schema'])
                return tool
            return None
    
    def list_tools(self, server_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tools, optionally filtered by server type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if server_type:
                cursor.execute("SELECT * FROM tools WHERE server_type = ?", (server_type,))
            else:
                cursor.execute("SELECT * FROM tools")
            
            tools = []
            for row in cursor.fetchall():
                tool = dict(row)
                tool['capabilities'] = json.loads(tool['capabilities'])
                tool['input_schema'] = json.loads(tool['input_schema'])
                tools.append(tool)
            
            return tools
    
    async def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tools by name or capability.
        
        Args:
            query: Search query
            
        Returns:
            List of matching tools
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Search in name and capabilities
            cursor.execute("""
                SELECT * FROM tools 
                WHERE name LIKE ? OR capabilities LIKE ?
                ORDER BY performance_score DESC
            """, (f'%{query}%', f'%{query}%'))
            
            tools = []
            for row in cursor.fetchall():
                tool = dict(row)
                if tool['capabilities']:
                    try:
                        tool['capabilities'] = json.loads(tool['capabilities'])
                    except:
                        pass
                if tool['input_schema']:
                    try:
                        tool['input_schema'] = json.loads(tool['input_schema'])
                    except:
                        pass
                tools.append(tool)
            
            return tools
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from the registry."""
        return self.list_tools()
    
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Async version of get_all_tools for compatibility."""
        return self.list_tools()
    
    async def get_tool_relationships(self, tool_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get tool relationships from the registry.
        
        Args:
            tool_id: If provided, get relationships for this specific tool.
                    If None, get all relationships.
        
        Returns:
            List of tool relationships
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if tool_id:
                cursor.execute("""
                    SELECT * FROM tool_relationships
                    WHERE tool1_id = ? OR tool2_id = ?
                """, (tool_id, tool_id))
            else:
                cursor.execute("""
                    SELECT * FROM tool_relationships
                """)
            
            relationships = [dict(row) for row in cursor.fetchall()]
            return relationships
    
    def _add_tool_relationship_sync(self, tool1_id: str, tool2_id: str, 
                                   relationship_type: str, strength: float = 0.5) -> bool:
        """
        Add a relationship between two tools (synchronous version).
        
        Args:
            tool1_id: ID of the first tool
            tool2_id: ID of the second tool  
            relationship_type: Type of relationship (e.g., 'complements', 'requires')
            strength: Strength of the relationship (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if both tools exist
                cursor.execute("SELECT id FROM tools WHERE id IN (?, ?)", (tool1_id, tool2_id))
                if len(cursor.fetchall()) != 2:
                    logger.warning(f"[RELATIONSHIP] One or both tools not found: {tool1_id}, {tool2_id}")
                    return False
                
                # Insert or update relationship
                cursor.execute("""
                    INSERT OR REPLACE INTO tool_relationships 
                    (tool1_id, tool2_id, relationship_type, strength)
                    VALUES (?, ?, ?, ?)
                """, (tool1_id, tool2_id, relationship_type, strength))
                
                conn.commit()
                logger.info(f"[RELATIONSHIP] Added: {tool1_id} {relationship_type} {tool2_id} (strength: {strength})")
                return True
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to add tool relationship: {e}")
            return False
    
    def add_tool_relationship(self, tool1_id: str, tool2_id: str,
                             relationship_type: str, strength: float = 0.5) -> bool:
        """Add a relationship between two tools."""
        return self._add_tool_relationship_sync(tool1_id, tool2_id, relationship_type, strength)
    
    async def add_tool_relationship(self, tool1_id: str, tool2_id: str,
                                   relationship_type: str, strength: float = 0.5) -> bool:
        """Async wrapper for add_tool_relationship."""
        return self._add_tool_relationship_sync(tool1_id, tool2_id, relationship_type, strength)
    
    def record_usage(self, tool_id: str, success: bool, execution_time: float, 
                    task_type: Optional[str] = None, error_message: Optional[str] = None,
                    retry_count: int = 0):
        """Record tool usage for learning with enhanced failure tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert execution time to milliseconds
                execution_time_ms = execution_time * 1000
                
                # Record in history
                cursor.execute("""
                    INSERT INTO usage_history 
                    (tool_id, task_type, success, execution_time, error_message, retry_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (tool_id, task_type, success, execution_time, error_message, retry_count))
                
                # Get current tool state
                cursor.execute("""
                    SELECT consecutive_failures, avg_response_time_ms, usage_count
                    FROM tools WHERE id = ?
                """, (tool_id,))
                
                row = cursor.fetchone()
                if row:
                    consecutive_failures, avg_time, usage_count = row
                    
                    # Calculate new average response time
                    if avg_time is None:
                        new_avg_time = execution_time_ms
                    else:
                        new_avg_time = (avg_time * usage_count + execution_time_ms) / (usage_count + 1)
                    
                    # Update tool statistics
                    if success:
                        cursor.execute("""
                            UPDATE tools 
                            SET usage_count = usage_count + 1,
                                success_count = success_count + 1,
                                failure_count = failure_count + 0,
                                consecutive_failures = 0,
                                performance_score = CAST(success_count + 1 AS REAL) / (usage_count + 1),
                                avg_response_time_ms = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (new_avg_time, tool_id))
                    else:
                        cursor.execute("""
                            UPDATE tools 
                            SET usage_count = usage_count + 1,
                                failure_count = failure_count + 1,
                                consecutive_failures = consecutive_failures + 1,
                                last_failure_time = CURRENT_TIMESTAMP,
                                performance_score = CAST(success_count AS REAL) / (usage_count + 1),
                                avg_response_time_ms = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (new_avg_time, tool_id))
                
                conn.commit()
                logger.info(f"[USAGE] Tool: {tool_id}, Success: {success}, Time: {execution_time:.2f}s, Retries: {retry_count}")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to record usage: {e}")
    
    def get_tool_performance(self, tool_id: str) -> Dict[str, Any]:
        """Get performance metrics for a tool."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get tool stats
            cursor.execute("""
                SELECT performance_score, usage_count, success_count
                FROM tools WHERE id = ?
            """, (tool_id,))
            
            row = cursor.fetchone()
            if not row:
                return {}
            
            # Get recent performance
            cursor.execute("""
                SELECT AVG(execution_time) as avg_time,
                       COUNT(*) as recent_uses,
                       SUM(success) as recent_successes
                FROM usage_history 
                WHERE tool_id = ? 
                AND timestamp > datetime('now', '-7 days')
            """, (tool_id,))
            
            recent = cursor.fetchone()
            
            return {
                'overall_score': row['performance_score'],
                'total_uses': row['usage_count'],
                'total_successes': row['success_count'],
                'recent_avg_time': recent['avg_time'] or 0,
                'recent_uses': recent['recent_uses'] or 0,
                'recent_success_rate': (recent['recent_successes'] or 0) / max(recent['recent_uses'] or 1, 1)
            }
    
    def find_related_tools(self, tool_id: str) -> List[Dict[str, Any]]:
        """Find tools that work well with the given tool."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT t.*, tr.strength, tr.usage_count as combo_uses
                FROM tools t
                JOIN tool_relationships tr ON (t.id = tr.tool2_id)
                WHERE tr.tool1_id = ?
                ORDER BY tr.strength DESC, tr.usage_count DESC
                LIMIT 5
            """, (tool_id,))
            
            related = []
            for row in cursor.fetchall():
                tool = dict(row)
                tool['capabilities'] = json.loads(tool['capabilities'])
                tool['input_schema'] = json.loads(tool['input_schema'])
                related.append(tool)
            
            return related
    
    def update_circuit_breaker_state(self, tool_id: str, state: str) -> bool:
        """Update circuit breaker state for a tool."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if state == 'open':
                    cursor.execute("""
                        UPDATE tools 
                        SET circuit_breaker_state = ?,
                            circuit_breaker_opened_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (state, tool_id))
                else:
                    cursor.execute("""
                        UPDATE tools 
                        SET circuit_breaker_state = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (state, tool_id))
                
                conn.commit()
                logger.info(f"[CIRCUIT_BREAKER] Tool {tool_id} state changed to: {state}")
                return True
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to update circuit breaker state: {e}")
            return False
    
    def record_retry_metric(self, tool_id: str, attempt_number: int, delay_ms: float,
                          error_type: Optional[str] = None, error_message: Optional[str] = None):
        """Record retry attempt metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO retry_metrics 
                    (tool_id, attempt_number, delay_ms, error_type, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (tool_id, attempt_number, delay_ms, error_type, error_message))
                
                conn.commit()
                logger.debug(f"[RETRY_METRIC] Tool: {tool_id}, Attempt: {attempt_number}, Delay: {delay_ms}ms")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to record retry metric: {e}")
    
    def get_failure_metrics(self, tool_id: str) -> Dict[str, Any]:
        """Get failure and retry metrics for a tool."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get tool failure stats
            cursor.execute("""
                SELECT failure_count, consecutive_failures, last_failure_time,
                       circuit_breaker_state, circuit_breaker_opened_at
                FROM tools WHERE id = ?
            """, (tool_id,))
            
            tool_row = cursor.fetchone()
            if not tool_row:
                return {}
            
            # Get retry statistics
            cursor.execute("""
                SELECT COUNT(*) as total_retries,
                       AVG(delay_ms) as avg_delay,
                       MAX(attempt_number) as max_attempts
                FROM retry_metrics 
                WHERE tool_id = ? 
                AND timestamp > datetime('now', '-24 hours')
            """, (tool_id,))
            
            retry_row = cursor.fetchone()
            
            # Get error distribution
            cursor.execute("""
                SELECT error_type, COUNT(*) as count
                FROM retry_metrics
                WHERE tool_id = ?
                AND timestamp > datetime('now', '-24 hours')
                GROUP BY error_type
            """, (tool_id,))
            
            error_dist = {row['error_type']: row['count'] for row in cursor.fetchall()}
            
            return {
                'failure_count': tool_row['failure_count'],
                'consecutive_failures': tool_row['consecutive_failures'],
                'last_failure_time': tool_row['last_failure_time'],
                'circuit_breaker_state': tool_row['circuit_breaker_state'],
                'circuit_breaker_opened_at': tool_row['circuit_breaker_opened_at'],
                'retry_stats': {
                    'total_retries_24h': retry_row['total_retries'] or 0,
                    'avg_delay_ms': retry_row['avg_delay'] or 0,
                    'max_attempts': retry_row['max_attempts'] or 0
                },
                'error_distribution': error_dist
            }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get overall system health report including failure rates."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get tools with circuit breaker open
            cursor.execute("""
                SELECT id, name, server_type, consecutive_failures, last_failure_time
                FROM tools 
                WHERE circuit_breaker_state = 'open'
            """)
            
            open_circuit_breakers = [dict(row) for row in cursor.fetchall()]
            
            # Get tools with high failure rates
            cursor.execute("""
                SELECT id, name, server_type, 
                       CAST(failure_count AS REAL) / usage_count as failure_rate,
                       failure_count, usage_count
                FROM tools 
                WHERE usage_count > 10 
                AND CAST(failure_count AS REAL) / usage_count > 0.3
                ORDER BY failure_rate DESC
            """)
            
            high_failure_tools = [dict(row) for row in cursor.fetchall()]
            
            # Get overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_tools,
                    SUM(usage_count) as total_uses,
                    SUM(success_count) as total_successes,
                    SUM(failure_count) as total_failures,
                    AVG(performance_score) as avg_performance
                FROM tools
            """)
            
            overall_stats = dict(cursor.fetchone())
            
            return {
                'overall_stats': overall_stats,
                'open_circuit_breakers': open_circuit_breakers,
                'high_failure_tools': high_failure_tools,
                'health_status': 'healthy' if len(open_circuit_breakers) == 0 else 'degraded'
            }

# Test the registry
def test_registry():
    """Test the tool registry with sample data."""
    logger.info("[TEST] Testing Tool Registry")
    
    registry = ToolRegistry()
    
    # Register some mock tools
    tools = [
        {
            'id': 'filesystem.read_file',
            'name': 'Read File',
            'server_type': 'filesystem',
            'endpoint': 'mock://filesystem',
            'description': 'Read contents of a file',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'File path'}
                },
                'required': ['path']
            }
        },
        {
            'id': 'filesystem.write_file',
            'name': 'Write File',
            'server_type': 'filesystem',
            'endpoint': 'mock://filesystem',
            'description': 'Write contents to a file',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'content': {'type': 'string'}
                },
                'required': ['path', 'content']
            }
        },
        {
            'id': 'time.get_current',
            'name': 'Get Current Time',
            'server_type': 'time',
            'endpoint': 'mock://time',
            'description': 'Get current time in specified timezone',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'timezone': {'type': 'string'}
                }
            }
        }
    ]
    
    # Register tools
    for tool in tools:
        registry.register_tool(tool)
    
    # List all tools
    all_tools = registry.list_tools()
    logger.info(f"[TEST] Registered {len(all_tools)} tools")
    
    # Record some usage
    registry.record_usage('filesystem.read_file', True, 0.5, 'data_processing')
    registry.record_usage('filesystem.read_file', True, 0.3, 'data_processing')
    registry.record_usage('filesystem.read_file', False, 1.2, 'data_processing', 'File not found')
    
    # Check performance
    perf = registry.get_tool_performance('filesystem.read_file')
    logger.info(f"[TEST] Performance: {perf}")
    
    logger.info("[TEST] Tool Registry test complete!")

if __name__ == "__main__":
    test_registry()