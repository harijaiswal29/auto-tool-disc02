"""Database migration to add context-aware columns to execution_history table."""

import aiosqlite
import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def add_context_columns(db_path: Optional[str] = None):
    """Add user_expertise and domain columns to execution_history table.
    
    Args:
        db_path: Path to database file. If None, uses default learning.db
    """
    if db_path is None:
        # Default to data directory
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        db_path = str(data_dir / "learning.db")
    
    logger.info(f"Adding context columns to database: {db_path}")
    
    async with aiosqlite.connect(db_path) as db:
        # Check if columns already exist
        cursor = await db.execute("PRAGMA table_info(execution_history)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # Add user_expertise column if it doesn't exist
        if 'user_expertise' not in column_names:
            await db.execute("""
                ALTER TABLE execution_history 
                ADD COLUMN user_expertise TEXT DEFAULT 'intermediate'
            """)
            logger.info("Added user_expertise column to execution_history")
        
        # Add domain column if it doesn't exist
        if 'domain' not in column_names:
            await db.execute("""
                ALTER TABLE execution_history 
                ADD COLUMN domain TEXT DEFAULT 'general'
            """)
            logger.info("Added domain column to execution_history")
        
        # Create indexes for efficient context-based queries
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
        logger.info("Context columns and indexes added successfully")


async def main():
    """Run the migration."""
    await add_context_columns()


if __name__ == "__main__":
    asyncio.run(main())