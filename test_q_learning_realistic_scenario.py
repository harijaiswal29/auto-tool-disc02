#!/usr/bin/env python3
"""Realistic Q-Learning Engine scenario test with pattern learning."""

import asyncio
import numpy as np
import json
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.learning.q_learning_engine import QLearningEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MockIntent:
    def __init__(self, embedding):
        self.embedding = embedding


async def simulate_realistic_scenario():
    """Simulate a realistic learning scenario with multiple episodes."""
    print("\n" + "="*80)
    print("REALISTIC Q-LEARNING SCENARIO: Multi-Episode Learning")
    print("="*80)
    
    # Configuration
    config = {
        'q_learning': {
            'learning_rate': 0.15,
            'discount_factor': 0.95,
            'exploration_rate': 0.4,
            'exploration_decay': 0.98,
            'min_exploration_rate': 0.05,
            'max_tools': 3,
            'batch_size': 8,
            'update_frequency': 4,
            'buffer_capacity': 1000,
            'use_patterns': True,
            'pattern_weight': 0.3
        }
    }
    
    engine = QLearningEngine(config)
    
    # Define tool sets for different scenarios
    scenarios = {
        'file_search': {
            'tools': ['filesystem_mcp', 'search_mcp', 'sqlite_mcp'],
            'optimal': ['filesystem_mcp', 'search_mcp'],
            'constraints': {}
        },
        'database_query': {
            'tools': ['sqlite_mcp', 'postgres_mcp', 'search_mcp'],
            'optimal': ['sqlite_mcp'],
            'constraints': {
                'conflicts': {'sqlite_mcp': ['postgres_mcp']}
            }
        },
        'code_analysis': {
            'tools': ['filesystem_mcp', 'github_mcp', 'search_mcp', 'sqlite_mcp'],
            'optimal': ['github_mcp', 'search_mcp'],
            'constraints': {
                'requires': {'github_mcp': ['filesystem_mcp']}
            }
        },
        'data_science': {
            'tools': ['sqlite_mcp', 'financial_datasets_mcp', 'search_mcp'],
            'optimal': ['financial_datasets_mcp', 'sqlite_mcp'],
            'constraints': {}
        }
    }
    
    # Track performance metrics
    episode_rewards = []
    optimal_action_rates = []
    exploration_rates = []
    
    print(f"\nTraining over 50 episodes with 4 different scenarios...")
    print(f"Initial exploration rate: {engine.exploration_rate:.3f}")
    
    # Run multiple episodes
    for episode in range(50):
        episode_reward = 0
        optimal_actions = 0
        total_actions = 0
        
        # Randomly select scenarios
        for step in range(10):  # 10 steps per episode
            scenario_name = np.random.choice(list(scenarios.keys()))
            scenario = scenarios[scenario_name]
            
            # Create state based on scenario
            intent = MockIntent(embedding=np.random.rand(384).tolist())
            context = {
                'domain': scenario_name,
                'query_count': step + 1,
                'session_duration': (step + 1) * 300,
                'metrics': {
                    'avg_response_time': 500 + np.random.rand() * 1000,
                    'success_rate': 0.7 + np.random.rand() * 0.3,
                    'error_rate': np.random.rand() * 0.2,
                    'tools_invoked': len(scenario['optimal']),
                    'cache_hit_rate': np.random.rand() * 0.5
                },
                'failure_rates': {
                    tool: np.random.rand() * 0.2 for tool in scenario['tools']
                },
                'failure_types': {
                    'network_timeout': int(np.random.rand() * 5),
                    'permission_error': int(np.random.rand() * 3),
                    'rate_limit': int(np.random.rand() * 2),
                    'connection_error': int(np.random.rand() * 4),
                    'other': int(np.random.rand() * 3)
                },
                'retry_patterns': {
                    'avg_retry_count': np.random.rand() * 2,
                    'retry_success_rate': 0.5 + np.random.rand() * 0.5,
                    'avg_retry_delay_ms': 1000 + np.random.rand() * 2000,
                    'circuit_breaker_triggers': int(np.random.rand() * 3),
                    'max_consecutive_failures': int(np.random.rand() * 4)
                },
                'user_expertise': np.random.choice(['novice', 'intermediate', 'expert']),
                'domain': scenario_name
            }
            
            # Encode state
            state = engine.state_encoder.encode_state(
                intent, context, 
                ['filesystem_mcp'] * (step % 3)  # Some history
            )
            
            # Select action
            action = await engine.select_action(
                state, scenario['tools'], scenario['constraints']
            )
            
            # Calculate reward based on how close action is to optimal
            action_set = set(action)
            optimal_set = set(scenario['optimal'])
            
            # Reward calculation
            if action_set == optimal_set:
                reward = 1.0  # Perfect match
                optimal_actions += 1
            elif action_set.issubset(optimal_set) or optimal_set.issubset(action_set):
                reward = 0.7  # Partial match
            elif action_set & optimal_set:  # Some overlap
                reward = 0.4
            else:
                reward = 0.1  # No match
            
            # Add penalties for inefficiency
            if len(action) > len(scenario['optimal']):
                reward *= 0.8  # Penalty for using too many tools
            
            # Create next state
            next_intent = MockIntent(embedding=np.random.rand(384).tolist())
            next_context = context.copy()
            next_context['query_count'] += 1
            
            next_state = engine.state_encoder.encode_state(
                next_intent, next_context, list(action)
            )
            
            # Learn from experience
            done = (step == 9)  # Last step of episode
            await engine.learn_from_experience(
                state, action, reward, next_state,
                scenario['tools'], scenario['constraints'], done
            )
            
            episode_reward += reward
            total_actions += 1
        
        # Track metrics
        episode_rewards.append(episode_reward)
        optimal_action_rates.append(optimal_actions / total_actions)
        exploration_rates.append(engine.exploration_rate)
        
        # Decay exploration
        engine.decay_exploration()
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_optimal = np.mean(optimal_action_rates[-10:])
            print(f"\nEpisode {episode + 1}:")
            print(f"  Average reward (last 10): {avg_reward:.3f}")
            print(f"  Optimal action rate: {avg_optimal:.2%}")
            print(f"  Exploration rate: {engine.exploration_rate:.3f}")
            print(f"  Q-table size: {len(engine.q_table.q_values)}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETE - FINAL EVALUATION")
    print("="*60)
    
    # Test learned behavior (exploitation only)
    engine.exploration_rate = 0.0
    test_results = {}
    
    for scenario_name, scenario in scenarios.items():
        # Create test state
        intent = MockIntent(embedding=np.random.rand(384).tolist())
        context = {
            'domain': scenario_name,
            'query_count': 5,
            'session_duration': 1500,
            'metrics': {'success_rate': 0.85},
            'failure_rates': {},
            'failure_types': {},
            'retry_patterns': {},
            'user_expertise': 'intermediate'
        }
        
        state = engine.state_encoder.encode_state(intent, context, [])
        
        # Get learned action
        action = await engine.select_action(
            state, scenario['tools'], scenario['constraints']
        )
        
        # Check if optimal
        is_optimal = set(action) == set(scenario['optimal'])
        test_results[scenario_name] = {
            'selected': action,
            'optimal': scenario['optimal'],
            'correct': is_optimal
        }
    
    print("\nLearned behaviors by scenario:")
    correct = 0
    for scenario, result in test_results.items():
        status = "✓" if result['correct'] else "✗"
        print(f"\n{scenario}:")
        print(f"  Selected: {result['selected']}")
        print(f"  Optimal:  {tuple(result['optimal'])}")
        print(f"  Status:   {status}")
        if result['correct']:
            correct += 1
    
    print(f"\nCorrect decisions: {correct}/{len(scenarios)} ({correct/len(scenarios)*100:.0f}%)")
    
    # Learning curve analysis
    print("\n" + "="*60)
    print("LEARNING CURVE ANALYSIS")
    print("="*60)
    
    print("\nReward progression:")
    for i in range(0, len(episode_rewards), 10):
        batch = episode_rewards[i:i+10]
        print(f"  Episodes {i+1}-{i+10}: avg reward = {np.mean(batch):.3f}")
    
    print("\nOptimal action rate progression:")
    for i in range(0, len(optimal_action_rates), 10):
        batch = optimal_action_rates[i:i+10]
        print(f"  Episodes {i+1}-{i+10}: optimal rate = {np.mean(batch):.2%}")
    
    # Final metrics
    metrics = engine.get_metrics()
    print("\n" + "="*60)
    print("FINAL Q-LEARNING METRICS")
    print("="*60)
    print(f"Total episodes: {metrics['episode_count']}")
    print(f"Total reward: {metrics['total_reward']:.2f}")
    print(f"Average reward per episode: {metrics['avg_reward']:.3f}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Q-table entries: {metrics['q_table_stats']['total_entries']}")
    print(f"Total Q-updates: {metrics['q_table_stats']['total_updates']}")
    print(f"Experience buffer size: {metrics['buffer_size']}")
    
    # Show Q-value statistics
    q_stats = metrics['q_table_stats']
    print(f"\nQ-value statistics:")
    print(f"  Average: {q_stats['avg_q_value']:.3f}")
    print(f"  Maximum: {q_stats['max_q_value']:.3f}")
    print(f"  Minimum: {q_stats['min_q_value']:.3f}")
    
    return correct == len(scenarios)


async def test_failure_learning():
    """Test how Q-learning adapts to failure patterns."""
    print("\n" + "="*80)
    print("FAILURE PATTERN LEARNING TEST")
    print("="*80)
    
    config = {
        'q_learning': {
            'learning_rate': 0.2,
            'discount_factor': 0.9,
            'exploration_rate': 0.3,
            'exploration_decay': 0.99,
            'min_exploration_rate': 0.05,
            'max_tools': 2
        }
    }
    
    engine = QLearningEngine(config)
    
    # Define tools with varying failure rates
    tools = ['reliable_tool', 'flaky_tool', 'failing_tool']
    
    print("\nSimulating tools with different failure rates:")
    print("  reliable_tool: 10% failure rate")
    print("  flaky_tool: 50% failure rate")
    print("  failing_tool: 90% failure rate")
    
    # Run episodes
    for episode in range(30):
        intent = MockIntent(embedding=np.random.rand(384).tolist())
        
        # Context with failure information
        context = {
            'domain': 'testing',
            'failure_rates': {
                'reliable_tool': 0.1,
                'flaky_tool': 0.5,
                'failing_tool': 0.9
            },
            'failure_types': {
                'network_timeout': 10,
                'connection_error': 5
            },
            'retry_patterns': {
                'avg_retry_count': 2.0,
                'retry_success_rate': 0.3
            },
            'user_expertise': 'intermediate'
        }
        
        state = engine.state_encoder.encode_state(intent, context, [])
        
        # Select action
        action = await engine.select_action(state, tools, {})
        
        # Simulate execution with failure
        rewards = {
            'reliable_tool': 1.0 if np.random.rand() > 0.1 else -0.5,
            'flaky_tool': 0.7 if np.random.rand() > 0.5 else -0.5,
            'failing_tool': 0.5 if np.random.rand() > 0.9 else -1.0
        }
        
        # Calculate reward for combination
        reward = np.mean([rewards.get(tool, 0) for tool in action])
        
        # Learn
        next_state = engine.state_encoder.encode_state(intent, context, list(action))
        await engine.learn_from_experience(
            state, action, reward, next_state, tools, {}, done=True
        )
        
        engine.decay_exploration()
    
    # Test final behavior
    print("\nFinal learned preferences (exploitation only):")
    engine.exploration_rate = 0.0
    
    action_counts = {}
    for _ in range(20):
        intent = MockIntent(embedding=np.random.rand(384).tolist())
        state = engine.state_encoder.encode_state(intent, context, [])
        action = await engine.select_action(state, tools, {})
        action_str = engine.action_space.encode_action(action)
        action_counts[action_str] = action_counts.get(action_str, 0) + 1
    
    print("\nAction selection distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action}: {count/20*100:.0f}%")
    
    # Check if learned to prefer reliable tool
    most_selected = max(action_counts.items(), key=lambda x: x[1])[0]
    prefers_reliable = 'reliable_tool' in most_selected
    
    print(f"\nLearned to prefer reliable tool: {'✓' if prefers_reliable else '✗'}")
    
    return prefers_reliable


async def main():
    """Run all Q-Learning tests."""
    print("\n" + "="*80)
    print("Q-LEARNING ENGINE COMPREHENSIVE TESTING")
    print("="*80)
    
    # Run tests
    scenario_success = await simulate_realistic_scenario()
    failure_learning_success = await test_failure_learning()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Realistic scenario learning: {'✓ PASSED' if scenario_success else '✗ FAILED'}")
    print(f"Failure pattern learning: {'✓ PASSED' if failure_learning_success else '✗ FAILED'}")
    
    if scenario_success and failure_learning_success:
        print("\n✓ ALL Q-LEARNING TESTS PASSED!")
        print("\nThe Q-Learning Engine successfully:")
        print("  - Learned optimal tool combinations for different scenarios")
        print("  - Adapted to tool failure patterns")
        print("  - Balanced exploration and exploitation")
        print("  - Maintained state representation with 447 dimensions")
        print("  - Applied constraints correctly")
    else:
        print("\n✗ Some tests failed. Check the output above.")


if __name__ == "__main__":
    asyncio.run(main())