#!/usr/bin/env python3
"""Test tool selection accuracy target of >80%.

This test validates that the system correctly selects appropriate tools
for various queries with high accuracy.
"""

import pytest
import numpy as np
import asyncio
from typing import Dict, List, Set, Tuple
import json
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.agents.tool_discovery_agent import ToolDiscoveryAgent
from src.orchestration.orchestrator_agent import OrchestratorAgent
from src.database.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.dissertation
@pytest.mark.performance
@pytest.mark.asyncio
class TestToolSelectionAccuracy:
    """Test suite for tool selection accuracy requirements."""
    
    @pytest.fixture
    async def tool_discovery_agent(self):
        """Create tool discovery agent."""
        config = {
            'discovery_strategy': 'hybrid',
            'semantic_threshold': 0.7,
            'max_tools_per_query': 5
        }
        registry = ToolRegistry("test_registry.db")
        await registry.initialize()
        
        # Register test tools
        await self._register_test_tools(registry)
        
        agent = ToolDiscoveryAgent(config, registry)
        await agent.initialize()
        
        yield agent
        
        await agent.cleanup()
        await registry.cleanup()
    
    async def _register_test_tools(self, registry: ToolRegistry):
        """Register tools for testing."""
        test_tools = [
            {
                'name': 'filesystem_search',
                'category': 'file_operations',
                'capabilities': ['search_files', 'list_files', 'find_patterns'],
                'description': 'Search and list files in filesystem'
            },
            {
                'name': 'sqlite_query',
                'category': 'database',
                'capabilities': ['query_data', 'execute_sql', 'schema_info'],
                'description': 'Query SQLite databases'
            },
            {
                'name': 'postgres_query',
                'category': 'database', 
                'capabilities': ['query_data', 'execute_sql', 'manage_connections'],
                'description': 'Query PostgreSQL databases'
            },
            {
                'name': 'web_search',
                'category': 'search',
                'capabilities': ['search_web', 'fetch_content', 'summarize'],
                'description': 'Search the web and fetch content'
            },
            {
                'name': 'weather_api',
                'category': 'api',
                'capabilities': ['get_weather', 'forecast', 'historical_data'],
                'description': 'Get weather information'
            },
            {
                'name': 'code_analyzer',
                'category': 'development',
                'capabilities': ['analyze_complexity', 'find_bugs', 'generate_metrics'],
                'description': 'Analyze code quality and metrics'
            },
            {
                'name': 'git_operations',
                'category': 'version_control',
                'capabilities': ['commit', 'diff', 'log', 'branch'],
                'description': 'Git version control operations'
            }
        ]
        
        for tool in test_tools:
            await registry.register_tool(
                name=tool['name'],
                category=tool['category'],
                capabilities=tool['capabilities'],
                metadata={'description': tool['description']}
            )
    
    @pytest.fixture
    def test_cases(self):
        """Test cases with expected tool selections."""
        return [
            # Single tool cases
            {
                'query': 'Find all Python files in the src directory',
                'expected_tools': {'filesystem_search'},
                'category': 'file_operations'
            },
            {
                'query': 'Query the users table in the database',
                'expected_tools': {'sqlite_query', 'postgres_query'},
                'category': 'database'
            },
            {
                'query': 'Search for MCP protocol documentation',
                'expected_tools': {'web_search'},
                'category': 'search'
            },
            {
                'query': 'Get current weather in London',
                'expected_tools': {'weather_api'},
                'category': 'api'
            },
            {
                'query': 'Analyze code complexity in main module',
                'expected_tools': {'code_analyzer'},
                'category': 'development'
            },
            
            # Multi-tool cases
            {
                'query': 'Find Python files and analyze their complexity',
                'expected_tools': {'filesystem_search', 'code_analyzer'},
                'category': 'multi_tool'
            },
            {
                'query': 'Search for weather data and store in database',
                'expected_tools': {'weather_api', 'sqlite_query', 'postgres_query'},
                'category': 'multi_tool'
            },
            {
                'query': 'Get git history and analyze code changes',
                'expected_tools': {'git_operations', 'code_analyzer'},
                'category': 'multi_tool'
            },
            
            # Ambiguous cases (multiple valid options)
            {
                'query': 'Get data from the system',
                'expected_tools': {'filesystem_search', 'sqlite_query', 'postgres_query'},
                'category': 'ambiguous'
            },
            {
                'query': 'Search for information about Python',
                'expected_tools': {'web_search', 'filesystem_search'},
                'category': 'ambiguous'
            }
        ]
    
    async def test_single_tool_selection_accuracy(self, tool_discovery_agent, test_cases):
        """Test accuracy for single-tool selection scenarios."""
        logger.info("Testing single-tool selection accuracy")
        
        single_tool_cases = [tc for tc in test_cases if tc['category'] != 'multi_tool']
        correct_selections = 0
        results = []
        
        for test_case in single_tool_cases:
            query = test_case['query']
            expected = test_case['expected_tools']
            
            # Discover tools
            discovered = await tool_discovery_agent.discover_tools(query)
            selected_tools = {tool['name'] for tool in discovered[:1]}  # Top selection
            
            # Check if selection is correct
            is_correct = bool(selected_tools & expected)
            if is_correct:
                correct_selections += 1
            
            results.append({
                'query': query,
                'expected': list(expected),
                'selected': list(selected_tools),
                'correct': is_correct,
                'confidence': discovered[0]['score'] if discovered else 0
            })
            
            logger.info(f"Query: {query}")
            logger.info(f"Expected: {expected}, Selected: {selected_tools}, "
                       f"Correct: {is_correct}")
        
        accuracy = correct_selections / len(single_tool_cases)
        logger.info(f"Single-tool selection accuracy: {accuracy:.1%}")
        
        # Assert accuracy requirement
        assert accuracy >= 0.80, f"Accuracy {accuracy:.1%} < 80%"
        
        # Save results
        self._save_results({
            'test': 'single_tool_accuracy',
            'accuracy': accuracy,
            'total_cases': len(single_tool_cases),
            'correct': correct_selections,
            'detailed_results': results
        })
    
    async def test_multi_tool_selection_accuracy(self, tool_discovery_agent, test_cases):
        """Test accuracy for multi-tool coordination scenarios."""
        logger.info("Testing multi-tool selection accuracy")
        
        multi_tool_cases = [tc for tc in test_cases if tc['category'] == 'multi_tool']
        results = []
        
        for test_case in multi_tool_cases:
            query = test_case['query']
            expected = test_case['expected_tools']
            
            # Discover tools
            discovered = await tool_discovery_agent.discover_tools(query)
            selected_tools = {tool['name'] for tool in discovered[:3]}  # Top 3 selections
            
            # Calculate precision and recall
            true_positives = len(selected_tools & expected)
            precision = true_positives / len(selected_tools) if selected_tools else 0
            recall = true_positives / len(expected) if expected else 0
            f1_score = 2 * (precision * recall) / (precision + recall) \
                      if (precision + recall) > 0 else 0
            
            results.append({
                'query': query,
                'expected': list(expected),
                'selected': list(selected_tools),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })
            
            logger.info(f"Query: {query}")
            logger.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}, "
                       f"F1: {f1_score:.2f}")
        
        # Calculate overall metrics
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1_score'] for r in results])
        
        logger.info(f"Multi-tool selection - Avg Precision: {avg_precision:.2f}, "
                   f"Avg Recall: {avg_recall:.2f}, Avg F1: {avg_f1:.2f}")
        
        # Assert performance requirements
        assert avg_f1 >= 0.70, f"F1 score {avg_f1:.2f} < 0.70"
        
        # Save results
        self._save_results({
            'test': 'multi_tool_accuracy',
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1,
            'detailed_results': results
        })
    
    async def test_false_positive_rate(self, tool_discovery_agent):
        """Test that false positive rate is <10%."""
        logger.info("Testing false positive rate")
        
        # Queries that should NOT trigger certain tools
        negative_cases = [
            {
                'query': 'Explain quantum computing concepts',
                'should_not_select': {'filesystem_search', 'sqlite_query', 'weather_api'},
                'reason': 'Conceptual query, no specific tool needed'
            },
            {
                'query': 'What is the meaning of life?',
                'should_not_select': {'code_analyzer', 'git_operations', 'postgres_query'},
                'reason': 'Philosophical query, no technical tools needed'
            },
            {
                'query': 'Calculate fibonacci sequence',
                'should_not_select': {'weather_api', 'web_search', 'sqlite_query'},
                'reason': 'Pure computation, no external tools needed'
            }
        ]
        
        false_positives = 0
        total_checks = 0
        results = []
        
        for case in negative_cases:
            query = case['query']
            should_not_select = case['should_not_select']
            
            # Discover tools
            discovered = await tool_discovery_agent.discover_tools(query)
            selected_tools = {tool['name'] for tool in discovered if tool['score'] > 0.5}
            
            # Check for false positives
            false_selections = selected_tools & should_not_select
            false_positives += len(false_selections)
            total_checks += len(should_not_select)
            
            results.append({
                'query': query,
                'selected': list(selected_tools),
                'false_positives': list(false_selections),
                'reason': case['reason']
            })
            
            if false_selections:
                logger.warning(f"False positives for '{query}': {false_selections}")
        
        false_positive_rate = false_positives / total_checks if total_checks > 0 else 0
        logger.info(f"False positive rate: {false_positive_rate:.1%}")
        
        # Assert requirement
        assert false_positive_rate < 0.10, \
            f"False positive rate {false_positive_rate:.1%} >= 10%"
        
        # Save results
        self._save_results({
            'test': 'false_positive_rate',
            'false_positive_rate': false_positive_rate,
            'total_false_positives': false_positives,
            'total_checks': total_checks,
            'detailed_results': results
        })
    
    async def test_optimal_tool_selection(self, tool_discovery_agent):
        """Test selection of optimal tool when multiple options exist."""
        logger.info("Testing optimal tool selection")
        
        # Cases where multiple tools could work but one is optimal
        optimization_cases = [
            {
                'query': 'Query user data from local database',
                'optimal': 'sqlite_query',  # Local, so SQLite is better
                'suboptimal': 'postgres_query',
                'reason': 'SQLite better for local databases'
            },
            {
                'query': 'Search for Python files in current directory',
                'optimal': 'filesystem_search',
                'suboptimal': 'web_search',
                'reason': 'Filesystem search more appropriate for local files'
            },
            {
                'query': 'Get weather forecast for tomorrow in New York',
                'optimal': 'weather_api',
                'suboptimal': 'web_search',
                'reason': 'Dedicated API more reliable than web search'
            }
        ]
        
        optimal_selections = 0
        results = []
        
        for case in optimization_cases:
            query = case['query']
            optimal = case['optimal']
            
            # Discover tools
            discovered = await tool_discovery_agent.discover_tools(query)
            
            # Check if optimal tool is ranked first
            top_tool = discovered[0]['name'] if discovered else None
            is_optimal = top_tool == optimal
            
            if is_optimal:
                optimal_selections += 1
            
            # Get ranking of both tools
            tool_rankings = {tool['name']: idx for idx, tool in enumerate(discovered)}
            optimal_rank = tool_rankings.get(optimal, -1)
            suboptimal_rank = tool_rankings.get(case['suboptimal'], -1)
            
            results.append({
                'query': query,
                'optimal_tool': optimal,
                'selected_first': top_tool,
                'is_optimal': is_optimal,
                'optimal_rank': optimal_rank,
                'suboptimal_rank': suboptimal_rank,
                'reason': case['reason']
            })
            
            logger.info(f"Query: {query}")
            logger.info(f"Optimal: {optimal}, Selected: {top_tool}, "
                       f"Correct: {is_optimal}")
        
        optimality_rate = optimal_selections / len(optimization_cases)
        logger.info(f"Optimal selection rate: {optimality_rate:.1%}")
        
        # Should select optimal tool most of the time
        assert optimality_rate >= 0.70, \
            f"Optimal selection rate {optimality_rate:.1%} < 70%"
        
        # Save results
        self._save_results({
            'test': 'optimal_tool_selection',
            'optimality_rate': optimality_rate,
            'optimal_selections': optimal_selections,
            'total_cases': len(optimization_cases),
            'detailed_results': results
        })
    
    async def test_selection_consistency(self, tool_discovery_agent):
        """Test that tool selection is consistent across multiple runs."""
        logger.info("Testing selection consistency")
        
        test_queries = [
            'Find Python files in the project',
            'Query database for user information',
            'Search for documentation online',
            'Analyze code quality metrics'
        ]
        
        num_runs = 5
        consistency_results = defaultdict(list)
        
        for query in test_queries:
            for run in range(num_runs):
                discovered = await tool_discovery_agent.discover_tools(query)
                top_tool = discovered[0]['name'] if discovered else None
                consistency_results[query].append(top_tool)
        
        # Calculate consistency metrics
        consistency_scores = []
        results = []
        
        for query, selections in consistency_results.items():
            # Most common selection
            most_common = max(set(selections), key=selections.count)
            consistency = selections.count(most_common) / len(selections)
            consistency_scores.append(consistency)
            
            results.append({
                'query': query,
                'selections': selections,
                'most_common': most_common,
                'consistency': consistency
            })
            
            logger.info(f"Query: {query}, Consistency: {consistency:.1%}")
        
        avg_consistency = np.mean(consistency_scores)
        logger.info(f"Average consistency: {avg_consistency:.1%}")
        
        # Should be highly consistent
        assert avg_consistency >= 0.80, \
            f"Consistency {avg_consistency:.1%} < 80%"
        
        # Save results
        self._save_results({
            'test': 'selection_consistency',
            'avg_consistency': avg_consistency,
            'num_runs': num_runs,
            'detailed_results': results
        })
    
    def _save_results(self, results: Dict):
        """Save test results for dissertation."""
        output_dir = Path(__file__).parent.parent / "results" / "performance_metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"tool_selection_accuracy_{results.get('test', 'general')}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / filename}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])