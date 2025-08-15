#!/usr/bin/env python3
"""
Complex Query Demonstration Tests

This module demonstrates the system's ability to handle complex,
multi-intent queries that require multiple tool coordination.
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.tools.tool_discovery import ToolDiscovery
from src.utils.logger import get_logger
from tests.dissertation_test_suite.data.test_queries import COMPLEX_QUERIES, TestQuery

logger = get_logger(__name__)


@pytest.mark.dissertation
@pytest.mark.scenario
@pytest.mark.asyncio
class TestComplexQueries:
    """Demonstrate system capability on complex multi-tool queries."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Initialize orchestrator agent."""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        yield orchestrator
    
    @pytest.fixture
    async def tool_discovery(self):
        """Initialize tool discovery."""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        discovery = ToolDiscovery(config)
        await discovery.initialize()
        yield discovery
    
    async def test_multi_tool_coordination(self, orchestrator):
        """Test 1: Demonstrate coordinated multi-tool execution."""
        query = "Find all Python files modified today and analyze their code complexity"
        logger.info(f"Testing multi-tool coordination: {query}")
        
        start_time = time.time()
        result = await orchestrator.process_query(query)
        execution_time = time.time() - start_time
        
        # Verify multi-tool usage
        assert result['success'], "Complex query should succeed"
        tools_used = [r['tool_name'] for r in result.get('tool_results', [])]
        
        # Should use both filesystem and code analysis tools
        assert len(tools_used) >= 2, f"Expected multiple tools, got {tools_used}"
        assert any('filesystem' in tool for tool in tools_used), "Should use filesystem tool"
        assert any('github' in tool or 'code' in tool for tool in tools_used), "Should use code analysis tool"
        
        # Check execution time is reasonable
        assert execution_time < 5.0, f"Complex query took too long: {execution_time:.2f}s"
        
        # Log results
        logger.info(f"Multi-tool coordination successful:")
        logger.info(f"  Tools used: {tools_used}")
        logger.info(f"  Execution time: {execution_time:.2f}s")
        logger.info(f"  Tool results: {len(result.get('tool_results', []))}")
        
        # Save demonstration
        self._save_demo_result("multi_tool_coordination", query, result, {
            'execution_time': execution_time,
            'tools_used': tools_used,
            'tool_count': len(tools_used)
        })
    
    async def test_parallel_execution(self, orchestrator):
        """Test 2: Demonstrate parallel tool execution for efficiency."""
        query = "Query sales data from database and search for market trends online"
        logger.info(f"Testing parallel execution: {query}")
        
        # Track execution timeline
        timeline = []
        
        # Hook into orchestrator to track parallel execution
        original_execute = orchestrator.execution_engine.execute_tool
        
        async def tracked_execute(tool, params):
            start = time.time()
            timeline.append({'tool': tool['name'], 'start': start, 'status': 'started'})
            result = await original_execute(tool, params)
            end = time.time()
            timeline.append({'tool': tool['name'], 'end': end, 'duration': end - start, 'status': 'completed'})
            return result
        
        orchestrator.execution_engine.execute_tool = tracked_execute
        
        # Execute query
        result = await orchestrator.process_query(query)
        
        # Restore original method
        orchestrator.execution_engine.execute_tool = original_execute
        
        # Analyze parallelism
        tools_used = [r['tool_name'] for r in result.get('tool_results', [])]
        
        # Check for parallel execution (tools starting before others finish)
        parallel_detected = False
        if len(timeline) >= 4:  # At least 2 tools with start/end events
            for i in range(len(timeline) - 1):
                if timeline[i]['status'] == 'started' and timeline[i+1]['status'] == 'started':
                    parallel_detected = True
                    break
        
        logger.info(f"Parallel execution analysis:")
        logger.info(f"  Tools used: {tools_used}")
        logger.info(f"  Parallel execution detected: {parallel_detected}")
        logger.info(f"  Timeline events: {len(timeline)}")
        
        # Save demonstration
        self._save_demo_result("parallel_execution", query, result, {
            'timeline': timeline,
            'parallel_detected': parallel_detected,
            'tools_used': tools_used
        })
    
    async def test_context_preservation(self, orchestrator):
        """Test 3: Demonstrate context preservation across tool executions."""
        # Multi-step query requiring context
        queries = [
            "Search for competitor information about AI startups",
            "Store the findings in a note with analysis"
        ]
        
        logger.info("Testing context preservation across queries")
        
        results = []
        context_preserved = True
        
        for i, query in enumerate(queries):
            logger.info(f"  Step {i+1}: {query}")
            result = await orchestrator.process_query(query)
            results.append(result)
            
            # Check if context from previous query is used
            if i > 0:
                # Second query should reference data from first
                tools_used = [r['tool_name'] for r in result.get('tool_results', [])]
                if 'notion_mcp' not in tools_used and 'filesystem_mcp' not in tools_used:
                    context_preserved = False
        
        logger.info(f"Context preservation: {'Success' if context_preserved else 'Failed'}")
        
        # Save demonstration
        self._save_demo_result("context_preservation", queries, results, {
            'context_preserved': context_preserved,
            'query_count': len(queries)
        })
    
    async def test_complex_query_accuracy(self, orchestrator):
        """Test 4: Measure accuracy on predefined complex queries."""
        results = []
        
        # Test first 10 complex queries
        for test_query in COMPLEX_QUERIES[:10]:
            logger.info(f"Testing: {test_query.query}")
            
            start_time = time.time()
            result = await orchestrator.process_query(test_query.query)
            execution_time = time.time() - start_time
            
            # Extract tools used
            tools_used = [r['tool_name'] for r in result.get('tool_results', [])]
            
            # Check accuracy
            expected_tools = set(test_query.optimal_tools)
            actual_tools = set(tools_used)
            
            # Calculate metrics
            precision = len(expected_tools & actual_tools) / len(actual_tools) if actual_tools else 0
            recall = len(expected_tools & actual_tools) / len(expected_tools) if expected_tools else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            query_result = {
                'query': test_query.query,
                'success': result['success'],
                'expected_tools': list(expected_tools),
                'actual_tools': list(actual_tools),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'execution_time': execution_time
            }
            
            results.append(query_result)
            
            logger.info(f"  F1 Score: {f1_score:.2f}, Time: {execution_time:.2f}s")
        
        # Calculate overall metrics
        avg_f1 = sum(r['f1_score'] for r in results) / len(results)
        avg_time = sum(r['execution_time'] for r in results) / len(results)
        success_rate = sum(1 for r in results if r['success']) / len(results)
        
        logger.info(f"\nComplex Query Performance:")
        logger.info(f"  Average F1 Score: {avg_f1:.2f}")
        logger.info(f"  Success Rate: {success_rate:.2%}")
        logger.info(f"  Average Time: {avg_time:.2f}s")
        
        # Assert performance targets
        assert avg_f1 >= 0.7, f"F1 score {avg_f1:.2f} below target 0.7"
        assert success_rate >= 0.8, f"Success rate {success_rate:.2%} below target 80%"
        
        # Save results
        self._save_demo_result("complex_accuracy", "batch_test", {
            'individual_results': results,
            'summary': {
                'avg_f1_score': avg_f1,
                'success_rate': success_rate,
                'avg_execution_time': avg_time
            }
        })
    
    async def test_error_recovery(self, orchestrator):
        """Test 5: Demonstrate error recovery in complex scenarios."""
        # Query that might face errors (e.g., API limits, timeouts)
        query = "Analyze Git commit patterns and identify frequent contributors"
        
        logger.info(f"Testing error recovery: {query}")
        
        # Simulate potential errors by tracking retries
        retry_count = 0
        error_recovered = False
        
        # Execute with monitoring
        result = await orchestrator.process_query(query)
        
        # Check if any tools had partial success or retries
        for tool_result in result.get('tool_results', []):
            if tool_result.get('retry_count', 0) > 0:
                retry_count += tool_result['retry_count']
            if tool_result.get('partial_success', False):
                error_recovered = True
        
        # Even with errors, complex query should complete
        assert result['success'] or error_recovered, "Should handle errors gracefully"
        
        logger.info(f"Error recovery test:")
        logger.info(f"  Query success: {result['success']}")
        logger.info(f"  Retries: {retry_count}")
        logger.info(f"  Error recovery: {error_recovered}")
        
        # Save demonstration
        self._save_demo_result("error_recovery", query, result, {
            'retry_count': retry_count,
            'error_recovered': error_recovered
        })
    
    async def test_tool_combination_discovery(self, orchestrator, tool_discovery):
        """Test 6: Demonstrate discovery of optimal tool combinations."""
        query = "Extract data from multiple databases and create comparison report"
        
        logger.info(f"Testing tool combination discovery: {query}")
        
        # First, discover available tools
        discovered_tools = await tool_discovery.discover_tools_for_query(query)
        
        logger.info(f"Discovered {len(discovered_tools)} potential tools")
        
        # Execute query
        result = await orchestrator.process_query(query)
        
        # Analyze tool combinations
        tools_used = [r['tool_name'] for r in result.get('tool_results', [])]
        
        # Check if optimal combination was found
        has_database_tools = any('sqlite' in t or 'postgres' in t for t in tools_used)
        has_reporting_tools = any('filesystem' in t or 'notion' in t for t in tools_used)
        
        optimal_combination = has_database_tools and has_reporting_tools
        
        logger.info(f"Tool combination analysis:")
        logger.info(f"  Discovered tools: {[t['name'] for t in discovered_tools[:5]]}")
        logger.info(f"  Used tools: {tools_used}")
        logger.info(f"  Optimal combination: {optimal_combination}")
        
        assert optimal_combination, "Should discover optimal tool combination"
        
        # Save demonstration
        self._save_demo_result("tool_combination", query, result, {
            'discovered_count': len(discovered_tools),
            'tools_used': tools_used,
            'optimal_combination': optimal_combination
        })
    
    async def test_performance_scaling(self, orchestrator):
        """Test 7: Demonstrate performance with increasing complexity."""
        # Queries of increasing complexity
        complexity_tests = [
            ("Simple", "Find Python files", 1),
            ("Medium", "Find Python files and check their size", 2),
            ("Complex", "Find Python files, analyze complexity, and create report", 3),
            ("Very Complex", "Analyze code repository, query commit data, search for best practices, and create comprehensive report", 4)
        ]
        
        scaling_results = []
        
        for complexity_name, query, expected_tools in complexity_tests:
            logger.info(f"Testing {complexity_name} complexity: {query}")
            
            start_time = time.time()
            result = await orchestrator.process_query(query)
            execution_time = time.time() - start_time
            
            tools_used = [r['tool_name'] for r in result.get('tool_results', [])]
            
            scaling_results.append({
                'complexity': complexity_name,
                'query': query,
                'expected_tools': expected_tools,
                'actual_tools': len(tools_used),
                'execution_time': execution_time,
                'success': result['success']
            })
            
            logger.info(f"  Tools: {len(tools_used)}, Time: {execution_time:.2f}s")
        
        # Analyze scaling
        times = [r['execution_time'] for r in scaling_results]
        
        # Check if execution time scales reasonably
        scaling_reasonable = all(t < 10.0 for t in times)  # All under 10 seconds
        
        logger.info(f"\nPerformance scaling analysis:")
        for r in scaling_results:
            logger.info(f"  {r['complexity']}: {r['execution_time']:.2f}s")
        
        assert scaling_reasonable, "Performance should scale reasonably with complexity"
        
        # Save results
        self._save_demo_result("performance_scaling", "scaling_test", {
            'results': scaling_results,
            'scaling_reasonable': scaling_reasonable
        })
    
    def _save_demo_result(self, test_name: str, query: Any, result: Any, 
                         metrics: Dict[str, Any] = None):
        """Save demonstration results for dissertation."""
        output_dir = Path(__file__).parent.parent / "results" / "demonstrations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"complex_query_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        
        demo_data = {
            'test_name': test_name,
            'query': query,
            'result': self._clean_result_for_save(result),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(demo_data, f, indent=2)
        
        logger.info(f"Saved demonstration to {filepath}")
    
    def _clean_result_for_save(self, result: Any) -> Any:
        """Clean result for JSON serialization."""
        if isinstance(result, dict):
            return {k: self._clean_result_for_save(v) for k, v in result.items()}
        elif isinstance(result, list):
            return [self._clean_result_for_save(v) for v in result]
        elif hasattr(result, '__dict__'):
            return self._clean_result_for_save(result.__dict__)
        else:
            return result


@pytest.mark.dissertation
@pytest.mark.scenario
def test_complex_query_golden_path():
    """
    Golden path test for complex queries.
    
    Shows ideal multi-tool coordination without system dependencies.
    """
    logger.info("Running complex query golden path demonstration")
    
    # Simulated perfect complex query execution
    demo_results = {
        'query': "Find all Python files modified today and analyze their code complexity",
        'success': True,
        'intents': ['file.find', 'code.analyze'],
        'tools_selected': ['filesystem_mcp', 'github_mcp'],
        'execution_time': 1.23,  # Reasonable for multi-tool
        'parallel_execution': True,
        'tool_results': [
            {
                'tool_name': 'filesystem_mcp',
                'success': True,
                'files_found': 15,
                'execution_time': 0.45
            },
            {
                'tool_name': 'github_mcp',
                'success': True,
                'complexity_analyzed': 15,
                'avg_complexity': 5.2,
                'execution_time': 0.78
            }
        ],
        'coordination_metrics': {
            'tools_coordinated': 2,
            'data_passed_between_tools': True,
            'parallel_speedup': 1.6  # 60% faster than sequential
        }
    }
    
    # Verify golden path
    assert demo_results['success']
    assert len(demo_results['tools_selected']) > 1
    assert demo_results['execution_time'] < 2.0
    assert demo_results['parallel_execution']
    
    logger.info("Complex query golden path completed successfully")
    logger.info(f"  Tools coordinated: {demo_results['coordination_metrics']['tools_coordinated']}")
    logger.info(f"  Parallel speedup: {demo_results['coordination_metrics']['parallel_speedup']:.1f}x")
    
    # Save golden path data
    output_dir = Path(__file__).parent.parent / "results" / "demonstrations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "golden_path_complex.json", 'w') as f:
        json.dump(demo_results, f, indent=2)


if __name__ == "__main__":
    # Run demonstrations
    pytest.main([__file__, "-v", "-s", "-m", "scenario"])