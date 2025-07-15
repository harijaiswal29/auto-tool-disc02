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
            
        async with aiosqlite.connect(self.db_path) as db:
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
                    usage_count INTEGER DEFAULT 0
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
            
            await db.commit()
            
        self._initialized = True
        logger.info("Learning database initialized")
    
    def get_connection(self):
        """Get a database connection context manager."""
        return aiosqlite.connect(self.db_path)
    
    async def close(self):
        """Close any open connections."""
        # Currently using context managers, so nothing to close
        pass