"""
Database models for context management.

This module defines the database schema for persistent storage of
user profiles, sessions, and conversation history.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiosqlite
import os
from pathlib import Path

from src.utils.logger import get_logger


class ContextDatabase:
    """Manages database operations for context persistence."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the context database."""
        self.logger = get_logger(__name__)
        
        if db_path is None:
            # Default to data directory
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "context.db")
        
        self.db_path = db_path
        self.logger.info(f"Context database path: {self.db_path}")
    
    def _get_connection(self):
        """Get a database connection with proper settings."""
        return aiosqlite.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None
        )
    
    async def initialize(self):
        """Create database tables if they don't exist."""
        async with aiosqlite.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None
        ) as db:
            # Enable WAL mode and set pragmas for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=10000")
            await db.execute("PRAGMA synchronous=NORMAL")
            # Enable foreign keys
            await db.execute("PRAGMA foreign_keys = ON")
            
            # Create users table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT,
                    preferences JSON DEFAULT '{}',
                    expertise_level TEXT DEFAULT 'intermediate',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP
                )
            """)
            
            # Create sessions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    domain TEXT DEFAULT 'general',
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    context JSON DEFAULT '{}',
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Create conversation_history table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    query TEXT NOT NULL,
                    normalized_query TEXT,
                    intent_type TEXT,
                    intent_confidence REAL,
                    tools_discovered JSON,
                    tools_selected JSON,
                    execution_success BOOLEAN,
                    execution_time_ms REAL,
                    context JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Create indexes for performance
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user 
                ON sessions(user_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_session 
                ON conversation_history(session_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_user 
                ON conversation_history(user_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_timestamp 
                ON conversation_history(timestamp)
            """)
            
            await db.commit()
            self.logger.info("Context database tables initialized")
    
    # User Management Methods
    
    async def create_user(self, user_id: str, username: Optional[str] = None, 
                         email: Optional[str] = None, preferences: Optional[Dict] = None) -> bool:
        """Create a new user profile."""
        try:
            async with self._get_connection() as db:
                await db.execute("""
                    INSERT INTO users (user_id, username, email, preferences, last_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id, 
                    username, 
                    email, 
                    json.dumps(preferences or {}),
                    datetime.now().isoformat()
                ))
                await db.commit()
                self.logger.info(f"Created user profile: {user_id}")
                return True
        except aiosqlite.IntegrityError:
            self.logger.warning(f"User already exists: {user_id}")
            return False
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile."""
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM users WHERE user_id = ?
            """, (user_id,))
            row = await cursor.fetchone()
            
            if row:
                user = dict(row)
                user['preferences'] = json.loads(user['preferences'])
                return user
            return None
    
    async def update_user(self, user_id: str, **kwargs) -> bool:
        """Update user profile."""
        allowed_fields = ['username', 'email', 'preferences', 'expertise_level']
        updates = []
        values = []
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                if field == 'preferences':
                    value = json.dumps(value)
                updates.append(f"{field} = ?")
                values.append(value)
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        updates.append("last_active = ?")
        values.extend([datetime.now().isoformat(), datetime.now().isoformat()])
        values.append(user_id)
        
        async with self._get_connection() as db:
            await db.execute(f"""
                UPDATE users 
                SET {', '.join(updates)}
                WHERE user_id = ?
            """, values)
            await db.commit()
            return True
    
    # Session Management Methods
    
    async def create_session(self, session_id: str, user_id: Optional[str] = None, 
                           domain: str = 'general') -> bool:
        """Create a new session."""
        try:
            async with self._get_connection() as db:
                await db.execute("""
                    INSERT INTO sessions (session_id, user_id, domain)
                    VALUES (?, ?, ?)
                """, (session_id, user_id, domain))
                await db.commit()
                self.logger.info(f"Created session: {session_id}")
                return True
        except aiosqlite.IntegrityError:
            self.logger.warning(f"Session already exists: {session_id}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data."""
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            row = await cursor.fetchone()
            
            if row:
                session = dict(row)
                session['context'] = json.loads(session['context'])
                return session
            return None
    
    async def update_session_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update session context."""
        async with self._get_connection() as db:
            await db.execute("""
                UPDATE sessions 
                SET context = ?
                WHERE session_id = ?
            """, (json.dumps(context), session_id))
            await db.commit()
            return True
    
    async def end_session(self, session_id: str) -> bool:
        """Mark session as ended."""
        async with self._get_connection() as db:
            await db.execute("""
                UPDATE sessions 
                SET end_time = ?, is_active = FALSE
                WHERE session_id = ?
            """, (datetime.now().isoformat(), session_id))
            await db.commit()
            return True
    
    # Conversation History Methods
    
    async def add_conversation_entry(self, 
                                   session_id: str,
                                   query: str,
                                   normalized_query: Optional[str] = None,
                                   user_id: Optional[str] = None,
                                   intent_type: Optional[str] = None,
                                   intent_confidence: Optional[float] = None,
                                   tools_discovered: Optional[List[str]] = None,
                                   tools_selected: Optional[List[str]] = None,
                                   execution_success: Optional[bool] = None,
                                   execution_time_ms: Optional[float] = None,
                                   context: Optional[Dict[str, Any]] = None) -> int:
        """Add entry to conversation history."""
        async with self._get_connection() as db:
            cursor = await db.execute("""
                INSERT INTO conversation_history 
                (session_id, user_id, query, normalized_query, intent_type, 
                 intent_confidence, tools_discovered, tools_selected, 
                 execution_success, execution_time_ms, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_id,
                query,
                normalized_query,
                intent_type,
                intent_confidence,
                json.dumps(tools_discovered or []),
                json.dumps(tools_selected or []),
                execution_success,
                execution_time_ms,
                json.dumps(context or {})
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_conversation_history(self, 
                                     session_id: Optional[str] = None,
                                     user_id: Optional[str] = None,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve conversation history."""
        query = "SELECT * FROM conversation_history WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
            history = []
            for row in rows:
                entry = dict(row)
                entry['tools_discovered'] = json.loads(entry['tools_discovered'])
                entry['tools_selected'] = json.loads(entry['tools_selected'])
                entry['context'] = json.loads(entry['context'])
                history.append(entry)
            
            return history
    
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        async with self._get_connection() as db:
            # Total queries
            cursor = await db.execute("""
                SELECT COUNT(*) as total_queries
                FROM conversation_history
                WHERE user_id = ?
            """, (user_id,))
            total_queries = (await cursor.fetchone())[0]
            
            # Success rate
            cursor = await db.execute("""
                SELECT 
                    COUNT(CASE WHEN execution_success = 1 THEN 1 END) as successful,
                    COUNT(*) as total
                FROM conversation_history
                WHERE user_id = ? AND execution_success IS NOT NULL
            """, (user_id,))
            row = await cursor.fetchone()
            success_rate = row[0] / row[1] if row[1] > 0 else 0
            
            # Most used intents
            cursor = await db.execute("""
                SELECT intent_type, COUNT(*) as count
                FROM conversation_history
                WHERE user_id = ? AND intent_type IS NOT NULL
                GROUP BY intent_type
                ORDER BY count DESC
                LIMIT 5
            """, (user_id,))
            top_intents = [(row[0], row[1]) for row in await cursor.fetchall()]
            
            # Average execution time
            cursor = await db.execute("""
                SELECT AVG(execution_time_ms) as avg_time
                FROM conversation_history
                WHERE user_id = ? AND execution_time_ms IS NOT NULL
            """, (user_id,))
            avg_execution_time = (await cursor.fetchone())[0] or 0
            
            return {
                'total_queries': total_queries,
                'success_rate': success_rate,
                'top_intents': top_intents,
                'avg_execution_time_ms': avg_execution_time
            }