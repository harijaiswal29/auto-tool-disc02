"""Database manager for learning system persistence."""

import aiosqlite
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database operations for the learning system."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database manager."""
        if db_path is None:
            # Default to data directory
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "learning.db")
        
        self.db_path = db_path
        logger.info(f"Learning database path: {self.db_path}")
        self._initialized = False
    
    async def initialize(self):
        """Create database tables if they don't exist."""
        if self._initialized:
            return
            
        async with aiosqlite.connect(
            self.db_path,
            timeout=30.0,  # Increased timeout
            isolation_level=None  # Autocommit mode
        ) as db:
            # Enable WAL mode to reduce locking issues
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=10000")  # 10 second busy timeout
            await db.execute("PRAGMA synchronous=NORMAL")
            # Create model snapshots table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for efficient lookups
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_snapshots_version 
                ON model_snapshots(version, model_type)
            """)
            
            # Create Q-learning states table (from schema)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS q_learning_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_hash TEXT NOT NULL UNIQUE,
                    state_vector JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create Q-values table (from schema)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS q_values (
                    state_id INTEGER NOT NULL,
                    action_hash TEXT NOT NULL,
                    action_tools JSON NOT NULL,
                    q_value REAL NOT NULL DEFAULT 0.0,
                    update_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (state_id) REFERENCES q_learning_states(id),
                    UNIQUE(state_id, action_hash)
                )
            """)
            
            # Create discovered patterns table (from schema)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS discovered_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    tool_sequence JSON NOT NULL,
                    support REAL NOT NULL,
                    confidence REAL NOT NULL,
                    lift REAL,
                    contexts JSON,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    temporal_metadata JSON  -- Stores temporal pattern information
                )
            """)
            
            # Create execution history table (from schema)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS execution_history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    intent JSON NOT NULL,
                    tools_used JSON NOT NULL,
                    execution_time_ms INTEGER,
                    success BOOLEAN,
                    reward REAL,
                    user_expertise TEXT DEFAULT 'intermediate',
                    domain TEXT DEFAULT 'general',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create learning metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    episode_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create failure history table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS failure_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    tool_id TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    recovery_successful BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (execution_id) REFERENCES execution_history(id)
                )
            """)
            
            # Create resource metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS resource_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    tool_id TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_mb REAL,
                    api_calls INTEGER,
                    execution_time_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (execution_id) REFERENCES execution_history(id)
                )
            """)
            
            # Create user feedback table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating INTEGER,
                    query_reformulated BOOLEAN DEFAULT FALSE,
                    follow_up_query TEXT,
                    follow_up_time_seconds REAL,
                    result_used BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (execution_id) REFERENCES execution_history(id)
                )
            """)
            
            # Create tool synergies table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tool_synergies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_combination TEXT NOT NULL UNIQUE,
                    success_rate REAL,
                    occurrence_count INTEGER DEFAULT 1,
                    synergy_score REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create pattern mining metadata table for incremental updates
            await db.execute("""
                CREATE TABLE IF NOT EXISTS pattern_mining_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert default values if not exist
            await db.execute("""
                INSERT OR IGNORE INTO pattern_mining_metadata (key, value) 
                VALUES ('last_processed_execution_id', NULL)
            """)
            await db.execute("""
                INSERT OR IGNORE INTO pattern_mining_metadata (key, value) 
                VALUES ('last_update_timestamp', datetime('now'))
            """)
            
            # Create pattern statistics table for running counts
            await db.execute("""
                CREATE TABLE IF NOT EXISTS pattern_statistics (
                    pattern_hash TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    tool_sequence JSON NOT NULL,
                    occurrence_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    total_support REAL DEFAULT 0.0,
                    total_confidence REAL DEFAULT 0.0,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_failure_history_tool 
                ON failure_history(tool_id, failure_type)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_history_created_at
                ON execution_history(created_at DESC)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_pattern_statistics_last_seen
                ON pattern_statistics(last_seen DESC)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_resource_metrics_tool 
                ON resource_metrics(tool_id, created_at)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_feedback_execution 
                ON user_feedback(execution_id)
            """)
            
            # Create indexes for context columns
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_history_expertise 
                ON execution_history(user_expertise)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_history_domain 
                ON execution_history(domain)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_history_context 
                ON execution_history(user_expertise, domain)
            """)
            
            await db.commit()
            
        self._initialized = True
        logger.info("Learning database initialized")
    
    def get_connection(self):
        """Get a database connection context manager with proper settings."""
        return aiosqlite.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None
        )
    
    async def close(self):
        """Close any open connections."""
        # Currently using context managers, so nothing to close
        pass
    
    async def record_failure(self, execution_id: str, tool_id: str, 
                           failure_type: str, error_message: str = None,
                           retry_count: int = 0, recovery_successful: bool = False):
        """Record a failure in the failure history."""
        async with self.get_connection() as conn:
            await conn.execute(
                """INSERT INTO failure_history 
                   (execution_id, tool_id, failure_type, error_message, 
                    retry_count, recovery_successful)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (execution_id, tool_id, failure_type, error_message,
                 retry_count, recovery_successful)
            )
            await conn.commit()
    
    async def record_resource_metrics(self, execution_id: str, tool_id: str,
                                    cpu_percent: float = None, memory_mb: float = None,
                                    api_calls: int = None, execution_time_ms: float = None):
        """Record resource usage metrics."""
        async with self.get_connection() as conn:
            await conn.execute(
                """INSERT INTO resource_metrics 
                   (execution_id, tool_id, cpu_percent, memory_mb, 
                    api_calls, execution_time_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (execution_id, tool_id, cpu_percent, memory_mb,
                 api_calls, execution_time_ms)
            )
            await conn.commit()
    
    async def record_user_feedback(self, execution_id: str, feedback_type: str,
                                 rating: int = None, query_reformulated: bool = False,
                                 follow_up_query: str = None, 
                                 follow_up_time_seconds: float = None,
                                 result_used: bool = None):
        """Record user feedback for an execution."""
        async with self.get_connection() as conn:
            await conn.execute(
                """INSERT INTO user_feedback 
                   (execution_id, feedback_type, rating, query_reformulated,
                    follow_up_query, follow_up_time_seconds, result_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (execution_id, feedback_type, rating, query_reformulated,
                 follow_up_query, follow_up_time_seconds, result_used)
            )
            await conn.commit()
    
    async def update_tool_synergy(self, tool_combination: str, success_rate: float,
                                occurrence_count: int, synergy_score: float):
        """Update or insert tool synergy data."""
        async with self.get_connection() as conn:
            # Try to update existing record
            cursor = await conn.execute(
                """UPDATE tool_synergies 
                   SET success_rate = ?, occurrence_count = ?, 
                       synergy_score = ?, last_updated = datetime('now')
                   WHERE tool_combination = ?""",
                (success_rate, occurrence_count, synergy_score, tool_combination)
            )
            
            # If no rows updated, insert new record
            if cursor.rowcount == 0:
                await conn.execute(
                    """INSERT INTO tool_synergies 
                       (tool_combination, success_rate, occurrence_count, synergy_score)
                       VALUES (?, ?, ?, ?)""",
                    (tool_combination, success_rate, occurrence_count, synergy_score)
                )
            
            await conn.commit()
    
    async def get_tool_failure_rates(self, time_window_hours: int = 24) -> dict:
        """Get failure rates for each tool within time window."""
        async with self.get_connection() as conn:
            query = """
                SELECT 
                    tool_id,
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN recovery_successful = 0 THEN 1 ELSE 0 END) as failures,
                    AVG(retry_count) as avg_retry_count
                FROM failure_history
                WHERE created_at > datetime('now', '-{} hours')
                GROUP BY tool_id
            """.format(time_window_hours)
            
            async with conn.execute(query) as cursor:
                rows = await cursor.fetchall()
                
                failure_rates = {}
                for row in rows:
                    tool_id, total, failures, avg_retry = row
                    failure_rates[tool_id] = {
                        'failure_rate': failures / total if total > 0 else 0,
                        'avg_retry_count': avg_retry or 0,
                        'total_attempts': total
                    }
                
                return failure_rates
    
    async def get_failure_type_distribution(self, tool_id: str = None) -> dict:
        """Get distribution of failure types."""
        async with self.get_connection() as conn:
            if tool_id:
                query = """
                    SELECT failure_type, COUNT(*) as count
                    FROM failure_history
                    WHERE tool_id = ?
                    GROUP BY failure_type
                """
                params = (tool_id,)
            else:
                query = """
                    SELECT failure_type, COUNT(*) as count
                    FROM failure_history
                    GROUP BY failure_type
                """
                params = ()
            
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return {row[0]: row[1] for row in rows}