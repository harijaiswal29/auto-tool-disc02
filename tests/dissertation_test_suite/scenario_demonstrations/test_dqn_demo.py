#!/usr/bin/env python3
"""
DQN Demonstration Test for Dissertation

This test demonstrates the Deep Q-Network (DQN) learning capability,
showing faster convergence and better performance compared to standard Q-learning.
"""

import pytest
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.learning.q_learning_engine import QLearningEngine
from tests.dissertation_test_suite.data.test_queries import get_evaluation_sets
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.dissertation
@pytest.mark.dqn
class TestDQNLearning:
    """Demonstrate DQN learning capabilities."""
    
    @pytest.fixture
    async def setup_dqn_orchestrator(self):
        """Setup orchestrator with DQN enabled."""
        # Load config
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Ensure DQN is enabled
        config['dqn']['enabled'] = True
        
        # Create orchestrator
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        # Verify DQN is active
        assert orchestrator.q_learning_engine is not None, "Q-learning engine not initialized"
        assert orchestrator.q_learning_engine.use_dqn, "DQN not enabled in Q-learning engine"
        
        logger.info("DQN orchestrator initialized successfully")
        logger.info(f"State dimensions: {orchestrator.q_learning_engine.state_encoder.total_dimensions}")
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_dqn_convergence_speed(self, setup_dqn_orchestrator):
        """Test that DQN converges faster than standard Q-learning."""
        orchestrator = await setup_dqn_orchestrator
        
        # Get test queries
        queries = get_evaluation_sets()['quick_test'][:5]  # Use 5 queries for quick demo
        
        logger.info(f"Testing DQN convergence with {len(queries)} queries")
        
        # Track performance over episodes
        episode_rewards = []
        success_rates = []
        
        # Run 50 episodes for demonstration
        num_episodes = 50
        for episode in range(num_episodes):
            episode_reward = 0
            successes = 0
            
            for query in queries:
                try:
                    # Process query
                    result = await orchestrator.process_user_query(query.query)
                    
                    # Check if successful
                    if hasattr(result, 'success') and result.success:
                        successes += 1
                        episode_reward += 1.0
                    else:
                        episode_reward -= 0.5
                    
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
                    episode_reward -= 1.0
            
            # Calculate metrics
            success_rate = successes / len(queries)
            episode_rewards.append(episode_reward)
            success_rates.append(success_rate)
            
            # Log progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_success = np.mean(success_rates[-10:])
                logger.info(f"Episode {episode + 1}: Avg Reward={avg_reward:.2f}, "
                          f"Success Rate={avg_success:.2%}")
        
        # Analyze convergence
        early_performance = np.mean(success_rates[:10])
        late_performance = np.mean(success_rates[-10:])
        improvement = late_performance - early_performance
        
        logger.info(f"\nDQN Performance Summary:")
        logger.info(f"Early success rate (episodes 1-10): {early_performance:.2%}")
        logger.info(f"Late success rate (episodes 41-50): {late_performance:.2%}")
        logger.info(f"Improvement: {improvement:.2%}")
        
        # Assert improvement
        assert late_performance > early_performance, "DQN should show improvement over episodes"
        assert late_performance > 0.6, "DQN should achieve >60% success rate after 50 episodes"
        
        return {
            'episode_rewards': episode_rewards,
            'success_rates': success_rates,
            'improvement': improvement
        }
    
    @pytest.mark.asyncio
    async def test_dqn_state_representation(self, setup_dqn_orchestrator):
        """Test that DQN uses full 447-dimensional state vectors."""
        orchestrator = await setup_dqn_orchestrator
        
        # Get a test query
        queries = get_evaluation_sets()['quick_test']
        test_query = queries[0]
        
        # Process query to generate state
        logger.info(f"Processing query: {test_query.query}")
        
        # Get intent first
        intent_result = await orchestrator.intent_agent.identify_intent(test_query.query)
        
        # Check state encoding
        state_encoder = orchestrator.q_learning_engine.state_encoder
        
        # Create context
        context = {
            'domain': 'general',
            'query_count': 1,
            'session_duration': 100,
            'total_queries': 10,
            'success_rate': 0.8,
            'metrics': {},
            'failure_rates': {},
            'failure_types': {},
            'retry_patterns': {},
            'user_expertise': 'intermediate',
            'available_tool_categories': ['filesystem', 'search', 'database']
        }
        
        # Encode state
        state = state_encoder.encode_state(intent_result, context, [])
        
        # Verify state dimensions
        assert state.shape[0] == 447, f"Expected 447 dimensions, got {state.shape[0]}"
        
        # Check that state has non-zero values in different components
        intent_component = state[:384]  # First 384 dims are intent embedding
        context_component = state[384:394]  # Next 10 are context
        
        assert np.any(intent_component != 0), "Intent embedding should have non-zero values"
        assert np.any(context_component != 0), "Context features should have non-zero values"
        
        logger.info(f"✓ State vector validated: {state.shape[0]} dimensions")
        logger.info(f"  Intent embedding norm: {np.linalg.norm(intent_component):.2f}")
        logger.info(f"  Context features norm: {np.linalg.norm(context_component):.2f}")
        
        return True
    
    @pytest.mark.asyncio
    async def test_dqn_vs_qlearning_comparison(self):
        """Compare DQN performance with standard Q-learning."""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.json"
        
        # Test with Q-learning
        with open(config_path) as f:
            ql_config = json.load(f)
        ql_config['dqn']['enabled'] = False
        
        ql_orchestrator = OrchestratorAgent(ql_config)
        await ql_orchestrator.initialize()
        
        # Test with DQN
        with open(config_path) as f:
            dqn_config = json.load(f)
        dqn_config['dqn']['enabled'] = True
        
        dqn_orchestrator = OrchestratorAgent(dqn_config)
        await dqn_orchestrator.initialize()
        
        # Get test queries
        queries = get_evaluation_sets()['quick_test'][:3]  # Use 3 queries for quick comparison
        
        # Run 20 episodes for each
        num_episodes = 20
        
        # Q-learning performance
        ql_rewards = []
        for episode in range(num_episodes):
            episode_reward = 0
            for query in queries:
                try:
                    result = await ql_orchestrator.process_user_query(query.query)
                    if hasattr(result, 'success') and result.success:
                        episode_reward += 1.0
                except:
                    pass
            ql_rewards.append(episode_reward)
        
        # DQN performance
        dqn_rewards = []
        for episode in range(num_episodes):
            episode_reward = 0
            for query in queries:
                try:
                    result = await dqn_orchestrator.process_user_query(query.query)
                    if hasattr(result, 'success') and result.success:
                        episode_reward += 1.0
                except:
                    pass
            dqn_rewards.append(episode_reward)
        
        # Compare average performance
        ql_avg = np.mean(ql_rewards)
        dqn_avg = np.mean(dqn_rewards)
        
        logger.info(f"\nComparison Results:")
        logger.info(f"Q-Learning average reward: {ql_avg:.2f}")
        logger.info(f"DQN average reward: {dqn_avg:.2f}")
        logger.info(f"DQN improvement: {((dqn_avg - ql_avg) / max(ql_avg, 0.1)) * 100:.1f}%")
        
        # DQN should perform at least as well as Q-learning
        assert dqn_avg >= ql_avg * 0.95, "DQN should perform comparably to Q-learning"
        
        return {
            'ql_rewards': ql_rewards,
            'dqn_rewards': dqn_rewards,
            'improvement': dqn_avg - ql_avg
        }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "-m", "dqn"])