#!/usr/bin/env python3
"""
Enhanced Baseline Comparison with Pattern Mining
================================================
Modified version that triggers pattern mining during Q-learning episodes.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the original baseline comparison
from tests.dissertation_test_suite.scripts.run_baseline_comparison import *

# Store original method
original_run_strategy_evaluation = BaselineComparisonRunner.run_strategy_evaluation

async def run_strategy_evaluation_with_patterns(self, strategy_name: str, queries: List,
                                               run_id: int, seed: int, checkpoint_enabled: bool = False) -> Dict:
    """Enhanced run_strategy_evaluation that triggers pattern mining for Q-learning strategies."""
    
    # Run original strategy evaluation
    result = await original_run_strategy_evaluation(self, strategy_name, queries, run_id, seed, checkpoint_enabled)
    
    # If it's a Q-learning strategy, trigger pattern mining
    if strategy_name in ['q_learning_tabular', 'q_learning_dqn']:
        try:
            logger.info(f"Attempting pattern mining for {strategy_name}")
            
            # Get the Q-learning engine from the strategy
            strategy = self.evaluation_engine.strategies.get(strategy_name)
            if strategy and hasattr(strategy, 'q_learning_engine'):
                q_engine = strategy.q_learning_engine
                
                # Trigger pattern mining
                logger.info(f"Starting pattern mining for {strategy_name}")
                patterns = await q_engine.update_patterns(use_incremental=True, batch_size=100)
                
                # Log pattern statistics
                total_patterns = sum(len(p) for p in patterns.values()) if patterns else 0
                logger.info(f"Pattern mining complete for {strategy_name}: {total_patterns} patterns discovered")
                
                # Add pattern info to results
                result['pattern_mining'] = {
                    'total_patterns': total_patterns,
                    'pattern_types': list(patterns.keys()) if patterns else [],
                    'enabled': True
                }
                
                # Get detailed pattern statistics if available
                if hasattr(q_engine, 'pattern_miner'):
                    pm = q_engine.pattern_miner
                    if hasattr(pm, 'discovered_patterns'):
                        pattern_stats = {
                            'total_discovered': len(pm.discovered_patterns),
                            'high_confidence': sum(
                                1 for p in pm.discovered_patterns.values()
                                if isinstance(p, dict) and p.get('confidence', 0) > 0.8
                            ),
                            'frequent': sum(
                                1 for p in pm.discovered_patterns.values()
                                if isinstance(p, dict) and p.get('support', 0) > 0.2
                            )
                        }
                        result['pattern_statistics'] = pattern_stats
                        logger.info(f"Pattern statistics for {strategy_name}: {pattern_stats}")
            else:
                logger.warning(f"No Q-learning engine found for {strategy_name}")
                result['pattern_mining'] = {'enabled': False, 'reason': 'No Q-learning engine'}
                
        except Exception as e:
            logger.error(f"Pattern mining failed for {strategy_name}: {e}")
            result['pattern_mining'] = {'enabled': False, 'error': str(e)}
    
    return result

# Apply the patch
BaselineComparisonRunner.run_strategy_evaluation = run_strategy_evaluation_with_patterns

# Log that pattern mining is enabled
logger.info("Pattern mining enhancement loaded successfully")

# Run the original main if this is the main script
if __name__ == "__main__":
    import asyncio
    logger.info("Running baseline comparison with pattern mining enabled")
    asyncio.run(main())