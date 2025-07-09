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
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tools table
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
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tool_id) REFERENCES tools(id)
                )
            """)
            
            conn.commit()
            logger.info("[DB] Database schema initialized")
    
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
    
    def record_usage(self, tool_id: str, success: bool, execution_time: float, 
                    task_type: Optional[str] = None, error_message: Optional[str] = None):
        """Record tool usage for learning."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Record in history
                cursor.execute("""
                    INSERT INTO usage_history 
                    (tool_id, task_type, success, execution_time, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (tool_id, task_type, success, execution_time, error_message))
                
                # Update tool statistics
                if success:
                    cursor.execute("""
                        UPDATE tools 
                        SET usage_count = usage_count + 1,
                            success_count = success_count + 1,
                            performance_score = CAST(success_count + 1 AS REAL) / (usage_count + 1),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (tool_id,))
                else:
                    cursor.execute("""
                        UPDATE tools 
                        SET usage_count = usage_count + 1,
                            performance_score = CAST(success_count AS REAL) / (usage_count + 1),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (tool_id,))
                
                conn.commit()
                logger.info(f"[USAGE] Tool: {tool_id}, Success: {success}, Time: {execution_time:.2f}s")
                
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