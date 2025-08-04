"""Demo script for Pattern Mining functionality.

This script demonstrates:
1. Extracting execution sequences from database
2. Mining sequential and combination patterns
3. Calculating pattern metrics (support, confidence, lift)
4. Using patterns for tool suggestions
5. Integration with Q-Learning engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from datetime import datetime, timedelta
from tabulate import tabulate

from src.learning.pattern_miner import PatternMiner
from src.learning.q_learning_engine import QLearningEngine
from src.utils.logger import get_logger

logger = get_logger("PatternMiningDemo")


async def create_sample_execution_data(db_path: str):
    """Create sample execution history for demonstration."""
    import aiosqlite
    
    async with aiosqlite.connect(db_path) as db:
        # Create tables if they don't exist
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
                temporal_metadata JSON
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pattern_statistics (
                pattern_hash TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                tool_sequence JSON NOT NULL,
                occurrence_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                total_support REAL DEFAULT 0.0,
                total_confidence REAL DEFAULT 0.0,
                last_seen TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pattern_mining_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert sample execution data showing common patterns
        sample_executions = [
            # Pattern 1: filesystem -> sqlite (file analysis pattern)
            ("demo1", "Find Python files and analyze", '{"type": "query.search"}', 
             '["filesystem_mcp", "sqlite_mcp"]', True, 0.9),
            ("demo2", "Search files and store results", '{"type": "query.search"}', 
             '["filesystem_mcp", "sqlite_mcp"]', True, 0.85),
            ("demo3", "Analyze project structure", '{"type": "query.analyze"}', 
             '["filesystem_mcp", "sqlite_mcp"]', True, 0.8),
            ("demo4", "Find and process data files", '{"type": "query.search"}', 
             '["filesystem_mcp", "sqlite_mcp"]', True, 0.95),
            
            # Pattern 2: search -> github (code search pattern)
            ("demo5", "Search for code examples", '{"type": "query.search"}', 
             '["search_mcp", "github_mcp"]', True, 0.9),
            ("demo6", "Find GitHub repositories", '{"type": "query.search"}', 
             '["search_mcp", "github_mcp"]', True, 0.88),
            ("demo7", "Search open source projects", '{"type": "query.search"}', 
             '["search_mcp", "github_mcp"]', True, 0.92),
            
            # Pattern 3: github -> sqlite -> filesystem (clone and analyze pattern)
            ("demo8", "Clone and analyze repo", '{"type": "action.create"}', 
             '["github_mcp", "sqlite_mcp", "filesystem_mcp"]', True, 0.95),
            ("demo9", "Download and process code", '{"type": "action.create"}', 
             '["github_mcp", "sqlite_mcp", "filesystem_mcp"]', True, 0.9),
            
            # Some failed sequences
            ("demo10", "Bad combination", '{"type": "query.search"}', 
             '["weather_mcp", "filesystem_mcp"]', False, -0.5),
            ("demo11", "Wrong tools", '{"type": "action.delete"}', 
             '["search_mcp", "weather_mcp"]', False, -0.3),
            
            # Single tool successes
            ("demo12", "Quick search", '{"type": "query.search"}', 
             '["search_mcp"]', True, 0.6),
            ("demo13", "File listing", '{"type": "query.retrieve"}', 
             '["filesystem_mcp"]', True, 0.7),
        ]
        
        # Insert data
        for i, exec_data in enumerate(sample_executions):
            # Create timestamp going back in time
            timestamp = (datetime.now() - timedelta(hours=len(sample_executions) - i)).isoformat()
            
            await db.execute("""
                INSERT OR REPLACE INTO execution_history 
                (id, query, intent, tools_used, success, reward, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (*exec_data, timestamp))
        
        await db.commit()
        logger.info(f"Created {len(sample_executions)} sample execution records")


async def demo_pattern_extraction(pattern_miner: PatternMiner):
    """Demonstrate sequence extraction from execution history."""
    print("\n" + "="*60)
    print("1. EXTRACTING EXECUTION SEQUENCES")
    print("="*60)
    
    sequences = await pattern_miner.extract_sequences()
    print(f"\nExtracted {len(sequences)} execution sequences")
    
    # Show some sequences
    print("\nSample sequences:")
    for i, seq in enumerate(sequences[:5]):
        print(f"\n  Sequence {i+1}:")
        print(f"    ID: {seq.execution_id}")
        print(f"    Tools: {' -> '.join(seq.tools)}")
        print(f"    Success: {seq.success}")
        print(f"    Reward: {seq.reward:.2f}")
        print(f"    Query: {seq.context.get('query', 'N/A')[:50]}...")
    
    return sequences


async def demo_sequential_pattern_mining(pattern_miner: PatternMiner, sequences):
    """Demonstrate sequential pattern mining."""
    print("\n" + "="*60)
    print("2. MINING SEQUENTIAL PATTERNS")
    print("="*60)
    
    patterns = await pattern_miner.mine_sequential_patterns(sequences)
    print(f"\nDiscovered {len(patterns)} sequential patterns")
    
    # Display patterns in a table
    if patterns:
        table_data = []
        for p in patterns[:10]:  # Show top 10
            table_data.append([
                ' -> '.join(p.tool_sequence),
                f"{p.support:.3f}",
                f"{p.confidence:.3f}",
                f"{p.lift:.3f}",
                ', '.join(p.contexts[:2]) if p.contexts else 'N/A'
            ])
        
        print("\nTop Sequential Patterns:")
        print(tabulate(table_data, 
                      headers=['Pattern', 'Support', 'Confidence', 'Lift', 'Contexts'],
                      tablefmt='grid'))
    
    return patterns


async def demo_combination_pattern_mining(pattern_miner: PatternMiner, sequences):
    """Demonstrate combination pattern mining."""
    print("\n" + "="*60)
    print("3. MINING COMBINATION PATTERNS")
    print("="*60)
    
    patterns = await pattern_miner.mine_combination_patterns(sequences)
    print(f"\nDiscovered {len(patterns)} combination patterns")
    
    # Display patterns
    if patterns:
        table_data = []
        for p in patterns[:10]:
            table_data.append([
                ' + '.join(sorted(p.tool_sequence)),
                f"{p.support:.3f}",
                f"{p.confidence:.3f}",
                f"{p.lift:.3f}"
            ])
        
        print("\nTop Combination Patterns:")
        print(tabulate(table_data,
                      headers=['Tools', 'Support', 'Confidence', 'Lift'],
                      tablefmt='grid'))
    
    return patterns


async def demo_pattern_suggestions(pattern_miner: PatternMiner):
    """Demonstrate pattern-based tool suggestions."""
    print("\n" + "="*60)
    print("4. PATTERN-BASED TOOL SUGGESTIONS")
    print("="*60)
    
    # Test different starting sequences
    test_sequences = [
        ["filesystem_mcp"],
        ["search_mcp"],
        ["github_mcp", "sqlite_mcp"],
        []
    ]
    
    for seq in test_sequences:
        suggestions = pattern_miner.suggest_next_tools(seq, k=3)
        
        print(f"\nCurrent tools: {' -> '.join(seq) if seq else '[empty]'}")
        if suggestions:
            print("  Suggested next tools:")
            for tool, score in suggestions:
                print(f"    - {tool}: score={score:.3f}")
        else:
            print("  No suggestions available")


async def demo_q_learning_integration():
    """Demonstrate integration with Q-Learning engine."""
    print("\n" + "="*60)
    print("5. Q-LEARNING WITH PATTERN GUIDANCE")
    print("="*60)
    
    # Create Q-learning engine with pattern support
    config = {
        'q_learning': {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'exploration_rate': 0.2,
            'use_patterns': True,
            'pattern_weight': 0.3,
            'pattern_min_support': 0.2,
            'pattern_min_confidence': 0.6
        },
        'database': {
            'path': './demo_pattern_mining.db'
        }
    }
    
    q_engine = QLearningEngine(config)
    
    # Initialize patterns
    await q_engine.initialize_patterns()
    
    # Simulate action selection with pattern guidance
    print("\nSimulating action selection with pattern guidance:")
    
    # Create dummy state
    import numpy as np
    dummy_state = np.random.rand(439)  # 439-dimensional state vector
    
    available_tools = ["filesystem_mcp", "sqlite_mcp", "github_mcp", "search_mcp"]
    constraints = {}
    
    # Without patterns (current_tools=None)
    action1 = await q_engine.select_action(dummy_state, available_tools, constraints)
    print(f"\nAction without pattern context: {action1}")
    
    # With pattern context
    current_tools = ["filesystem_mcp"]
    action2 = await q_engine.select_action(dummy_state, available_tools, constraints, current_tools)
    print(f"Action with pattern context {current_tools}: {action2}")
    
    # Get metrics
    metrics = q_engine.get_metrics()
    print(f"\nQ-Learning metrics:")
    print(f"  Pattern count: {metrics['pattern_count']}")
    print(f"  Patterns loaded: {metrics['patterns_loaded']}")


async def main():
    """Run the pattern mining demonstration."""
    print("\n" + "="*60)
    print("PATTERN MINING DEMONSTRATION")
    print("="*60)
    
    # Setup
    db_path = './demo_pattern_mining.db'
    
    # Create a mock config object that mimics Config class
    class MockConfig:
        def __init__(self, data):
            self.data = data
        
        def get(self, key, default=None):
            return self.data.get(key, default)
    
    config = MockConfig({
        'database': {'path': db_path},
        'pattern_mining': {
            'max_pattern_length': 3,
            'time_window_days': 30
        }
    })
    
    # Create sample data
    print("\nCreating sample execution data...")
    await create_sample_execution_data(db_path)
    
    # Initialize pattern miner
    pattern_miner = PatternMiner(config, min_support=0.2, min_confidence=0.6)
    pattern_miner.db_path = db_path
    
    # Demo 1: Extract sequences
    sequences = await demo_pattern_extraction(pattern_miner)
    
    # Demo 2: Mine sequential patterns
    seq_patterns = await demo_sequential_pattern_mining(pattern_miner, sequences)
    
    # Demo 3: Mine combination patterns
    combo_patterns = await demo_combination_pattern_mining(pattern_miner, sequences)
    
    # Demo 4: Complete mining process
    print("\n" + "="*60)
    print("COMPLETE PATTERN MINING PROCESS")
    print("="*60)
    
    all_patterns = await pattern_miner.mine_patterns()
    total = sum(len(p) for p in all_patterns.values())
    print(f"\nTotal patterns discovered: {total}")
    for ptype, patterns in all_patterns.items():
        print(f"  {ptype}: {len(patterns)} patterns")
    
    # Demo 5: Pattern suggestions
    await demo_pattern_suggestions(pattern_miner)
    
    # Demo 6: Q-Learning integration
    await demo_q_learning_integration()
    
    # Cleanup
    print("\n" + "="*60)
    print("Demo complete!")
    
    # Optional: Remove demo database
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Cleaned up demo database: {db_path}")


if __name__ == "__main__":
    asyncio.run(main())