"""Simple demonstration of Pattern Mining functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from datetime import datetime
import aiosqlite

from src.learning.pattern_miner import PatternMiner
from src.utils.logger import get_logger

logger = get_logger("SimplePatternDemo")


async def create_demo_database():
    """Create a demo database with execution history."""
    db_path = './pattern_demo.db'
    
    async with aiosqlite.connect(db_path) as db:
        # Create tables
        await db.execute("""
            CREATE TABLE IF NOT EXISTS execution_history (
                id TEXT PRIMARY KEY,
                query TEXT,
                intent JSON,
                tools_used JSON,
                success BOOLEAN,
                reward REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS discovered_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                tool_sequence JSON,
                support REAL,
                confidence REAL,
                lift REAL,
                contexts JSON,
                discovered_at TIMESTAMP,
                usage_count INTEGER DEFAULT 0
            )
        """)
        
        # Insert realistic execution history
        executions = [
            # Common pattern: search -> analyze -> store
            ("exec01", "Find customer data", '{"type": "query.search"}', 
             '["search_mcp", "sqlite_mcp", "filesystem_mcp"]', True, 0.95),
            ("exec02", "Search user records", '{"type": "query.search"}', 
             '["search_mcp", "sqlite_mcp", "filesystem_mcp"]', True, 0.9),
            ("exec03", "Look up transactions", '{"type": "query.search"}', 
             '["search_mcp", "sqlite_mcp", "filesystem_mcp"]', True, 0.92),
            
            # Pattern: file -> database
            ("exec04", "Import CSV data", '{"type": "action.create"}', 
             '["filesystem_mcp", "sqlite_mcp"]', True, 0.88),
            ("exec05", "Load config files", '{"type": "query.retrieve"}', 
             '["filesystem_mcp", "sqlite_mcp"]', True, 0.85),
            ("exec06", "Process log files", '{"type": "query.analyze"}', 
             '["filesystem_mcp", "sqlite_mcp"]', True, 0.9),
            ("exec07", "Archive old data", '{"type": "action.modify"}', 
             '["filesystem_mcp", "sqlite_mcp"]', True, 0.87),
            
            # Pattern: github -> local analysis
            ("exec08", "Clone repository", '{"type": "action.create"}', 
             '["github_mcp", "filesystem_mcp"]', True, 0.85),
            ("exec09", "Download project", '{"type": "action.create"}', 
             '["github_mcp", "filesystem_mcp"]', True, 0.82),
            
            # Some failures
            ("exec10", "Bad query", '{"type": "query.search"}', 
             '["weather_mcp"]', False, -0.5),
        ]
        
        for exec_data in executions:
            await db.execute("""
                INSERT INTO execution_history 
                (id, query, intent, tools_used, success, reward, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, exec_data)
        
        await db.commit()
    
    return db_path


async def main():
    """Run simple pattern mining demonstration."""
    print("\n🔍 PATTERN MINING DEMONSTRATION")
    print("="*50)
    
    # Create demo data
    print("\n1️⃣ Creating demo execution history...")
    db_path = await create_demo_database()
    print("   ✅ Created 10 execution records")
    
    # Mock config
    class MockConfig:
        def get(self, key, default=None):
            return {
                'database': {'path': db_path},
                'pattern_mining': {
                    'max_pattern_length': 3,
                    'time_window_days': 30
                }
            }.get(key, default)
    
    # Initialize pattern miner
    print("\n2️⃣ Initializing Pattern Miner...")
    pattern_miner = PatternMiner(MockConfig(), min_support=0.15, min_confidence=0.7)
    pattern_miner.db_path = db_path
    print("   ✅ Pattern Miner ready")
    
    # Extract sequences
    print("\n3️⃣ Extracting execution sequences...")
    sequences = await pattern_miner.extract_sequences()
    print(f"   ✅ Found {len(sequences)} sequences")
    
    # Mine patterns
    print("\n4️⃣ Mining patterns...")
    patterns = await pattern_miner.mine_patterns()
    
    print("\n📊 DISCOVERED PATTERNS:")
    print("-"*50)
    
    # Show sequential patterns
    if patterns['sequential']:
        print("\n🔗 Sequential Patterns (order matters):")
        for i, p in enumerate(patterns['sequential'][:5], 1):
            if len(p.tool_sequence) > 1:  # Show only multi-tool patterns
                print(f"\n   Pattern {i}: {' → '.join(p.tool_sequence)}")
                print(f"   Support: {p.support:.1%} | Confidence: {p.confidence:.1%} | Lift: {p.lift:.2f}")
    
    # Show combination patterns
    if patterns['combination']:
        print("\n🎯 Combination Patterns (tools work well together):")
        for i, p in enumerate(patterns['combination'][:5], 1):
            print(f"\n   Pattern {i}: {' + '.join(p.tool_sequence)}")
            print(f"   Support: {p.support:.1%} | Confidence: {p.confidence:.1%} | Lift: {p.lift:.2f}")
    
    # Demonstrate tool suggestions
    print("\n\n5️⃣ Tool Suggestions Based on Patterns:")
    print("-"*50)
    
    test_sequences = [
        ["search_mcp"],
        ["filesystem_mcp"],
        ["github_mcp"]
    ]
    
    for seq in test_sequences:
        suggestions = pattern_miner.suggest_next_tools(seq, k=2)
        print(f"\n   After using: {' → '.join(seq)}")
        if suggestions:
            print("   Suggested next tools:")
            for tool, score in suggestions:
                print(f"      • {tool} (score: {score:.2f})")
        else:
            print("      • No suggestions available")
    
    # Show pattern statistics
    total_patterns = sum(len(p) for p in patterns.values())
    print(f"\n\n📈 SUMMARY:")
    print(f"   • Total patterns discovered: {total_patterns}")
    print(f"   • Sequential patterns: {len(patterns['sequential'])}")
    print(f"   • Combination patterns: {len(patterns['combination'])}")
    print(f"   • Most common pattern length: 2-3 tools")
    
    # Cleanup
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    asyncio.run(main())