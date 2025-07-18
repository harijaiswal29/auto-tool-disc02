"""Demonstration of Pattern Mining integration with Q-Learning."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
from datetime import datetime

from src.learning.q_learning_engine import QLearningEngine
from src.utils.logger import get_logger

logger = get_logger("PatternQLearningDemo")


async def main():
    """Demonstrate how patterns enhance Q-learning decisions."""
    print("\n🤖 PATTERN-ENHANCED Q-LEARNING DEMONSTRATION")
    print("="*60)
    
    # Configuration for Q-learning with patterns
    config = {
        'q_learning': {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'exploration_rate': 0.2,
            'use_patterns': True,         # Enable pattern usage
            'pattern_weight': 0.3,        # 30% weight to patterns
            'pattern_min_support': 0.15,
            'pattern_min_confidence': 0.7
        },
        'database': {
            'path': './pattern_qlearning_demo.db'
        }
    }
    
    # Initialize Q-learning engine
    print("\n1️⃣ Initializing Q-Learning Engine with Pattern Support...")
    q_engine = QLearningEngine(config)
    print("   ✅ Q-Learning engine ready")
    print(f"   ✅ Pattern weight: {config['q_learning']['pattern_weight']*100}%")
    
    # Create a dummy state (439-dimensional vector)
    state = np.random.rand(439)
    available_tools = ["filesystem_mcp", "sqlite_mcp", "search_mcp", "github_mcp"]
    constraints = {}
    
    print("\n2️⃣ Demonstrating Action Selection:")
    print("-"*60)
    
    # Scenario 1: No pattern context (first tool selection)
    print("\n📍 Scenario 1: Starting fresh (no tools used yet)")
    action1 = await q_engine.select_action(state, available_tools, constraints)
    print(f"   Selected action: {action1}")
    print("   (Selection based on Q-values or exploration)")
    
    # Scenario 2: With pattern context
    print("\n📍 Scenario 2: After using filesystem_mcp")
    current_tools = ["filesystem_mcp"]
    action2 = await q_engine.select_action(state, available_tools, constraints, current_tools)
    print(f"   Selected action: {action2}")
    print("   (Pattern miner might suggest sqlite_mcp based on common patterns)")
    
    # Scenario 3: Different starting point
    print("\n📍 Scenario 3: After using search_mcp")
    current_tools = ["search_mcp"]
    action3 = await q_engine.select_action(state, available_tools, constraints, current_tools)
    print(f"   Selected action: {action3}")
    print("   (Pattern miner might suggest github_mcp or sqlite_mcp)")
    
    # Demonstrate learning from experience
    print("\n\n3️⃣ Learning from Experience:")
    print("-"*60)
    
    # Simulate a successful execution
    reward = 0.9  # High reward for success
    next_state = np.random.rand(439)
    
    print(f"\n   Executing: {action2}")
    print(f"   Result: SUCCESS (reward = {reward})")
    
    # Learn from the experience
    await q_engine.learn_from_experience(
        state, action2, reward, next_state, 
        available_tools, constraints
    )
    print("   ✅ Q-table updated with experience")
    
    # Show metrics
    metrics = q_engine.get_metrics()
    print("\n\n4️⃣ Current Learning Metrics:")
    print("-"*60)
    print(f"   • Total reward: {metrics['total_reward']:.2f}")
    print(f"   • Episodes: {metrics['episode_count']}")
    print(f"   • Success rate: {metrics['success_rate']:.1%}")
    print(f"   • Exploration rate: {metrics['exploration_rate']:.1%}")
    print(f"   • Q-table entries: {metrics['q_table_stats']['total_entries']}")
    
    # Demonstrate pattern influence
    print("\n\n5️⃣ Pattern Influence on Decision Making:")
    print("-"*60)
    print("\n   Without patterns (pattern_weight = 0):")
    print("   - Decisions based purely on Q-values")
    print("   - Slower to learn effective tool combinations")
    print("   - More exploration needed")
    
    print("\n   With patterns (pattern_weight = 0.3):")
    print("   - 70% Q-values + 30% pattern suggestions")
    print("   - Faster convergence to optimal sequences")
    print("   - Benefits from historical success patterns")
    print("   - Better cold-start performance")
    
    print("\n\n📊 KEY BENEFITS:")
    print("-"*60)
    print("   1. Faster Learning: Patterns guide exploration")
    print("   2. Better Generalization: Learn from similar contexts")
    print("   3. Improved Success Rate: Leverage proven combinations")
    print("   4. Reduced Exploration: Focus on promising actions")
    
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    asyncio.run(main())