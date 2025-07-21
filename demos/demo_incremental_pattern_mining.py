"""Demo script for incremental pattern mining functionality."""

import asyncio
import json
from datetime import datetime, timedelta
import random
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.learning.pattern_miner import PatternMiner
from src.database.database import DatabaseManager


async def simulate_executions(db_path: str, num_executions: int = 20):
    """Simulate some tool executions for pattern mining."""
    db_manager = DatabaseManager(db_path)
    
    # Common tool patterns
    tool_patterns = [
        # Sequential patterns
        ['filesystem_mcp', 'sqlite_mcp'],
        ['search_mcp', 'analyze_mcp'],
        ['github_mcp', 'analyze_mcp', 'sqlite_mcp'],
        
        # Single tools
        ['filesystem_mcp'],
        ['search_mcp'],
        
        # Combinations
        ['filesystem_mcp', 'search_mcp', 'sqlite_mcp'],
        ['postgres_mcp', 'analyze_mcp'],
    ]
    
    intent_types = ['search', 'analyze', 'create', 'modify']
    
    print(f"\n📊 Simulating {num_executions} executions...")
    
    for i in range(num_executions):
        exec_id = f"demo_exec_{i:03d}"
        tools = random.choice(tool_patterns)
        intent = {'type': random.choice(intent_types)}
        
        # Higher success rate for known good patterns
        if tools in [['filesystem_mcp', 'sqlite_mcp'], ['search_mcp', 'analyze_mcp']]:
            success = random.random() < 0.85  # 85% success rate
            reward = random.uniform(0.6, 1.0) if success else random.uniform(-0.5, 0)
        else:
            success = random.random() < 0.6  # 60% success rate
            reward = random.uniform(0.3, 0.8) if success else random.uniform(-0.8, -0.2)
        
        # Add temporal variation
        hour = i % 24  # Simulate hourly patterns
        timestamp = datetime.now() - timedelta(hours=num_executions-i, minutes=random.randint(0, 59))
        
        await db_manager.save_execution(
            execution_id=exec_id,
            user_id="demo_user",
            session_id="demo_session",
            query=f"Demo query {i}",
            intent=intent,
            tools_used=tools,
            execution_time_ms=random.randint(100, 5000),
            success=success,
            reward=reward,
            timestamp=timestamp
        )
        
        if (i + 1) % 5 == 0:
            print(f"  ✅ Created {i + 1} executions...")
    
    print(f"  ✅ Created {num_executions} total executions")


async def demonstrate_incremental_mining():
    """Demonstrate incremental pattern mining capabilities."""
    print("\n🎯 Incremental Pattern Mining Demo\n")
    print("=" * 60)
    
    # Setup database
    db_path = './data/demo_incremental_patterns.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Initialize database
    db_manager = DatabaseManager(db_path)
    await db_manager.initialize()
    
    # Initialize pattern miner
    config = {
        'database': {'path': db_path},
        'pattern_mining': {
            'max_pattern_length': 3,
            'time_window_days': 30
        }
    }
    
    pattern_miner = PatternMiner(config, min_support=0.15, min_confidence=0.6)
    
    # Step 1: Initial executions and full pattern mining
    print("\n📥 Step 1: Initial Pattern Mining")
    print("-" * 40)
    
    await simulate_executions(db_path, num_executions=50)
    
    print("\n🔍 Running full pattern mining...")
    patterns = await pattern_miner.mine_patterns()
    
    total_patterns = sum(len(p) for p in patterns.values())
    print(f"\n✨ Initial patterns discovered: {total_patterns}")
    for pattern_type, pattern_list in patterns.items():
        if pattern_list:
            print(f"\n  {pattern_type.capitalize()} patterns ({len(pattern_list)}):")
            for p in pattern_list[:3]:  # Show top 3
                print(f"    • {' → '.join(p.tool_sequence)}")
                print(f"      Support: {p.support:.2f}, Confidence: {p.confidence:.2f}, Lift: {p.lift:.2f}")
    
    # Step 2: Add new executions
    print("\n\n📥 Step 2: Adding New Executions")
    print("-" * 40)
    
    await simulate_executions(db_path, num_executions=20)
    
    # Step 3: Incremental update
    print("\n🔄 Step 3: Incremental Pattern Update")
    print("-" * 40)
    
    print("  • Extracting only new sequences...")
    new_sequences = await pattern_miner.extract_new_sequences(batch_size=100)
    print(f"  • Found {len(new_sequences)} new sequences to process")
    
    # Get metadata
    last_id = await pattern_miner.get_pattern_mining_metadata('last_processed_execution_id')
    print(f"  • Last processed execution: {last_id}")
    
    print("\n  • Running incremental pattern mining...")
    incremental_patterns = await pattern_miner.incremental_update(
        batch_size=100,
        decay_factor=0.95
    )
    
    new_pattern_count = sum(len(p) for p in incremental_patterns.values())
    print(f"\n✨ New/updated patterns: {new_pattern_count}")
    
    for pattern_type, pattern_list in incremental_patterns.items():
        if pattern_list:
            print(f"\n  {pattern_type.capitalize()} patterns ({len(pattern_list)}):")
            for p in pattern_list[:3]:  # Show top 3
                print(f"    • {' → '.join(p.tool_sequence)}")
                print(f"      Support: {p.support:.2f}, Confidence: {p.confidence:.2f}")
    
    # Step 4: Compare with full mining
    print("\n\n📊 Step 4: Efficiency Comparison")
    print("-" * 40)
    
    # Time incremental update
    start_time = datetime.now()
    await pattern_miner.incremental_update(batch_size=100)
    incremental_time = (datetime.now() - start_time).total_seconds()
    
    # Time full mining
    start_time = datetime.now()
    await pattern_miner.mine_patterns()
    full_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n⏱️  Timing comparison:")
    print(f"  • Incremental update: {incremental_time:.3f} seconds")
    print(f"  • Full pattern mining: {full_time:.3f} seconds")
    print(f"  • Speedup: {full_time/incremental_time:.1f}x faster")
    
    # Step 5: Pattern statistics
    print("\n\n📈 Step 5: Pattern Statistics")
    print("-" * 40)
    
    stats = await pattern_miner.load_pattern_statistics()
    print(f"\n  Total tracked patterns: {len(stats)}")
    
    # Show top patterns by occurrence
    sorted_patterns = sorted(stats.items(), 
                           key=lambda x: x[1]['occurrence_count'], 
                           reverse=True)[:5]
    
    print("\n  Top patterns by occurrence:")
    for pattern_hash, pattern_stats in sorted_patterns:
        tools = pattern_stats['tool_sequence']
        print(f"    • {' → '.join(tools)}")
        print(f"      Occurrences: {pattern_stats['occurrence_count']}, "
              f"Success rate: {pattern_stats['success_count']/pattern_stats['occurrence_count']:.2%}")
    
    # Cleanup
    print("\n\n🧹 Cleaning up...")
    pruned = await pattern_miner.prune_outdated_patterns(min_support_threshold=0.05, max_age_days=1)
    print(f"  • Pruned {pruned} outdated patterns")
    
    print("\n✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demonstrate_incremental_mining())