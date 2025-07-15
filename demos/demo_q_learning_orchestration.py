"""Demo script showing Q-Learning integration with orchestration."""

import asyncio
import json
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.agents.orchestrator_agent import OrchestratorAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def demo_q_learning_orchestration():
    """Demonstrate Q-learning based tool selection in orchestration."""
    
    print("\n" + "="*60)
    print("Q-Learning Orchestration Demo")
    print("="*60 + "\n")
    
    # Load config and enable Q-learning
    config_path = os.path.join(project_root, 'config/config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Make sure Q-learning is enabled
    config['q_learning']['enable_learning'] = True
    
    # Initialize orchestrator with Q-learning
    print("Initializing Orchestrator with Q-Learning enabled...")
    orchestrator = OrchestratorAgent(config)
    await orchestrator.initialize()
    
    # Test queries to demonstrate learning
    test_queries = [
        "Find all Python files in the project",
        "Search for configuration files",
        "Analyze code quality metrics",
        "List recent commits in the repository",
        "Find TODO comments in the codebase"
    ]
    
    print("\nRunning queries to demonstrate Q-learning in action...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Query {i}: {query}")
        print('='*50)
        
        try:
            # Process query
            result = await orchestrator.process_user_query(query)
            
            # Display results
            print(f"\nIntent: {result.intent.primary_intent.type} "
                  f"(confidence: {result.intent.primary_intent.confidence:.2f})")
            
            print(f"\nDiscovered {len(result.discovered_tools)} tools:")
            for tool in result.discovered_tools[:3]:
                print(f"  - {tool['name']} (relevance: {tool.get('relevance_score', 0):.2f})")
            
            print(f"\nSelected tools (via Q-learning):")
            for tool_id in result.selected_tools:
                print(f"  - {tool_id}")
            
            print(f"\nExecution results:")
            for exec_result in result.execution_results:
                status = "✓" if exec_result.success else "✗"
                print(f"  {status} {exec_result.tool_name}: "
                      f"{exec_result.error or 'Success'}")
            
            print(f"\nTotal time: {result.total_time_ms:.0f}ms")
            print(f"Success: {result.success}")
            
            # Show Q-learning metrics if available
            if orchestrator.q_learning_engine:
                metrics = orchestrator.q_learning_engine.get_metrics()
                print(f"\nQ-Learning Metrics:")
                print(f"  - Episodes: {metrics['episode_count']}")
                print(f"  - Avg Reward: {metrics['avg_reward']:.2f}")
                print(f"  - Success Rate: {metrics['success_rate']:.2%}")
                print(f"  - Exploration Rate: {metrics['exploration_rate']:.3f}")
                print(f"  - Q-table Size: {metrics['q_table_stats']['total_entries']}")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")
        
        # Small delay between queries
        await asyncio.sleep(1)
    
    # Show final learning statistics
    if orchestrator.q_learning_engine:
        print(f"\n{'='*60}")
        print("Final Q-Learning Statistics")
        print('='*60)
        
        metrics = orchestrator.q_learning_engine.get_metrics()
        stats = metrics['q_table_stats']
        
        print(f"\nLearning Summary:")
        print(f"  - Total Episodes: {metrics['episode_count']}")
        print(f"  - Total Reward: {metrics['total_reward']:.2f}")
        print(f"  - Average Reward: {metrics['avg_reward']:.2f}")
        print(f"  - Success Rate: {metrics['success_rate']:.2%}")
        print(f"  - Final Exploration Rate: {metrics['exploration_rate']:.3f}")
        
        print(f"\nQ-Table Statistics:")
        print(f"  - Total Entries: {stats['total_entries']}")
        print(f"  - Total Updates: {stats['total_updates']}")
        print(f"  - Average Q-value: {stats['avg_q_value']:.3f}")
        print(f"  - Max Q-value: {stats['max_q_value']:.3f}")
        print(f"  - Min Q-value: {stats['min_q_value']:.3f}")
        
        print(f"\nExperience Buffer:")
        print(f"  - Buffer Size: {metrics['buffer_size']}")
        
        # Save the learned model
        print(f"\nSaving learned model...")
        await orchestrator.q_learning_engine.save_model("demo_model")
        print("Model saved successfully!")
    
    # Cleanup
    await orchestrator.shutdown()
    print("\nDemo complete!")


async def main():
    """Run the demo."""
    try:
        await demo_q_learning_orchestration()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\nDemo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())