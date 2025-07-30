"""
Comprehensive Orchestrator Agent Demo

This demo showcases all capabilities of the Orchestrator Agent including:
- Tool selection strategies (traditional and Q-learning)
- Parallel execution management
- Result aggregation
- Learning integration and improvement over time
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.core.tool_registry import ToolRegistry
from src.tools.mock_mcp_servers import start_mock_servers
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrchestratorDemo:
    """Demonstrate Orchestrator Agent capabilities."""
    
    def __init__(self):
        """Initialize demo with configuration."""
        self.config = {
            'orchestration': {
                'max_tools_per_query': 3,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True
            },
            'q_learning': {
                'enable_learning': True,
                'alpha': 0.1,
                'gamma': 0.9,
                'epsilon': 0.3,  # Higher for demo to show exploration
                'model_path': 'demos/demo_q_model.pkl'
            },
            'intent_recognition': {
                'model': 'all-MiniLM-L6-v2',
                'confidence_threshold': 0.7
            },
            'database': {
                'tool_registry': 'data/registry/tools.db',
                'learning_db': 'data/databases/learning.db'
            }
        }
        
        self.orchestrator = None
        self.registry = None
        self.demo_queries = [
            # Basic queries
            "Search for Python programming tutorials",
            "Read the configuration file from the project",
            "Query the database for user information",
            
            # Complex queries requiring multiple tools
            "Search for weather data and save it to a database",
            "Analyze financial data and create a report",
            "Find all Python files in the project and analyze their content",
            
            # Queries to test learning
            "Process large dataset efficiently",  # Should learn fast vs slow tools
            "Retrieve critical system information reliably",  # Should learn reliable tools
            "Export database results to multiple formats"  # Should learn tool combinations
        ]
    
    async def setup(self):
        """Set up the demo environment."""
        print("\n🚀 Setting up Orchestrator Demo Environment...")
        
        # Initialize tool registry
        self.registry = ToolRegistry(self.config['database']['tool_registry'])
        
        # Check if tools are already registered
        existing_tools = await self.registry.get_all_tools()
        if not existing_tools:
            print("📝 Registering demo tools...")
            await self._register_demo_tools()
        else:
            print(f"✅ Found {len(existing_tools)} existing tools")
        
        # Start mock MCP servers
        print("🔌 Starting mock MCP servers...")
        mock_servers = await start_mock_servers()
        print(f"✅ Started {len(mock_servers)} mock servers")
        
        # Initialize orchestrator
        print("🎯 Initializing Orchestrator Agent...")
        self.orchestrator = OrchestratorAgent(self.config)
        await self.orchestrator.initialize()
        print("✅ Orchestrator ready!")
    
    async def _register_demo_tools(self):
        """Register demo tools with varied characteristics."""
        demo_tools = [
            # Search tools
            {
                'id': 'search.fast',
                'name': 'Fast Web Search',
                'type': 'search',
                'server': 'search_mcp',
                'capabilities': {'operations': ['search', 'find', 'query']},
                'performance_score': 0.85,
                'metadata': {'avg_response_time': 100, 'reliability': 0.95}
            },
            {
                'id': 'search.comprehensive',
                'name': 'Comprehensive Search',
                'type': 'search',
                'server': 'search_mcp',
                'capabilities': {'operations': ['search', 'analyze', 'deep_search']},
                'performance_score': 0.95,
                'metadata': {'avg_response_time': 500, 'reliability': 0.98}
            },
            
            # Database tools
            {
                'id': 'database.sqlite',
                'name': 'SQLite Database',
                'type': 'database',
                'server': 'sqlite_mcp',
                'capabilities': {'operations': ['query', 'insert', 'update', 'delete']},
                'performance_score': 0.90,
                'metadata': {'reliability': 0.99}
            },
            {
                'id': 'database.export',
                'name': 'Database Exporter',
                'type': 'database',
                'server': 'sqlite_mcp',
                'capabilities': {'operations': ['export', 'convert', 'save']},
                'performance_score': 0.85,
                'metadata': {'formats': ['csv', 'json', 'xml']}
            },
            
            # File system tools
            {
                'id': 'filesystem.reader',
                'name': 'File Reader',
                'type': 'filesystem',
                'server': 'filesystem_mcp',
                'capabilities': {'operations': ['read', 'list', 'search']},
                'performance_score': 0.92,
                'metadata': {'max_file_size': '10MB'}
            },
            {
                'id': 'filesystem.writer',
                'name': 'File Writer',
                'type': 'filesystem',
                'server': 'filesystem_mcp',
                'capabilities': {'operations': ['write', 'create', 'append']},
                'performance_score': 0.88,
                'metadata': {'safe_mode': True}
            },
            
            # Analysis tools
            {
                'id': 'analyzer.code',
                'name': 'Code Analyzer',
                'type': 'analyzer',
                'server': 'analyzer_mcp',
                'capabilities': {'operations': ['analyze', 'inspect', 'evaluate']},
                'performance_score': 0.87,
                'metadata': {'languages': ['python', 'javascript', 'java']}
            },
            {
                'id': 'analyzer.data',
                'name': 'Data Analyzer',
                'type': 'analyzer',
                'server': 'analyzer_mcp',
                'capabilities': {'operations': ['analyze', 'statistics', 'visualize']},
                'performance_score': 0.91,
                'metadata': {'max_dataset_size': '100MB'}
            }
        ]
        
        # Register tools
        for tool in demo_tools:
            await self.registry.add_tool(
                tool_id=tool['id'],
                name=tool['name'],
                tool_type=tool['type'],
                server=tool['server'],
                capabilities=json.dumps(tool['capabilities']),
                status='active',
                performance_score=tool['performance_score']
            )
        
        # Add relationships
        relationships = [
            ('database.sqlite', 'database.export', 'complements'),
            ('search.fast', 'search.comprehensive', 'alternatives'),
            ('filesystem.reader', 'analyzer.code', 'complements'),
            ('analyzer.data', 'database.export', 'complements')
        ]
        
        for tool1, tool2, rel_type in relationships:
            await self.registry.add_tool_relationship(tool1, tool2, rel_type)
    
    async def demonstrate_tool_selection_strategies(self):
        """Demonstrate different tool selection strategies."""
        print("\n\n📊 DEMONSTRATING TOOL SELECTION STRATEGIES")
        print("=" * 60)
        
        test_query = "Search for data and analyze it"
        strategies = ['performance_weighted', 'relevance_only', 'performance_only']
        
        results = []
        
        for strategy in strategies:
            print(f"\n🔍 Testing strategy: {strategy}")
            self.orchestrator.config['orchestration']['tool_selection_strategy'] = strategy
            
            # Temporarily disable Q-learning to test traditional strategies
            original_q_setting = self.orchestrator.config['q_learning']['enable_learning']
            self.orchestrator.config['q_learning']['enable_learning'] = False
            
            result = await self.orchestrator.process_user_query(test_query)
            
            results.append({
                'Strategy': strategy,
                'Selected Tools': ', '.join(result.selected_tools),
                'Execution Time': f"{result.total_time_ms:.2f}ms",
                'Success': '✅' if result.success else '❌'
            })
            
            # Restore Q-learning setting
            self.orchestrator.config['q_learning']['enable_learning'] = original_q_setting
        
        # Display results
        print("\n📈 Strategy Comparison:")
        print(tabulate(results, headers='keys', tablefmt='grid'))
    
    async def demonstrate_parallel_execution(self):
        """Demonstrate parallel vs sequential execution."""
        print("\n\n⚡ DEMONSTRATING PARALLEL EXECUTION")
        print("=" * 60)
        
        test_query = "Search multiple sources, query database, and analyze results"
        
        # Test parallel execution
        print("\n🚀 Parallel Execution:")
        self.orchestrator.config['orchestration']['parallel_execution'] = True
        start_time = datetime.now()
        result_parallel = await self.orchestrator.process_user_query(test_query)
        parallel_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Completed in {parallel_time:.2f} seconds")
        print(f"Tools executed: {len(result_parallel.execution_results)}")
        
        # Test sequential execution
        print("\n🚶 Sequential Execution:")
        self.orchestrator.config['orchestration']['parallel_execution'] = False
        start_time = datetime.now()
        result_sequential = await self.orchestrator.process_user_query(test_query)
        sequential_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Completed in {sequential_time:.2f} seconds")
        print(f"Tools executed: {len(result_sequential.execution_results)}")
        
        # Compare results
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1
        print(f"\n📊 Speedup from parallel execution: {speedup:.2f}x")
        
        # Restore parallel setting
        self.orchestrator.config['orchestration']['parallel_execution'] = True
    
    async def demonstrate_result_aggregation(self):
        """Demonstrate result aggregation from multiple tools."""
        print("\n\n🔄 DEMONSTRATING RESULT AGGREGATION")
        print("=" * 60)
        
        test_query = "Find Python files, analyze their code, and create a summary report"
        
        result = await self.orchestrator.process_user_query(test_query)
        
        print(f"\n📋 Query: {test_query}")
        print(f"Intent: {result.intent.primary_intent.type} (confidence: {result.intent.primary_intent.confidence:.2f})")
        print(f"\n🔧 Tools Discovered: {len(result.discovered_tools)}")
        print(f"🎯 Tools Selected: {len(result.selected_tools)}")
        
        # Display execution results
        print("\n📊 Execution Results:")
        execution_table = []
        
        for exec_result in result.execution_results:
            execution_table.append({
                'Tool': exec_result.tool_name,
                'Success': '✅' if exec_result.success else '❌',
                'Time (ms)': f"{exec_result.execution_time_ms:.2f}",
                'Quality': f"{exec_result.result_quality:.2f}",
                'Partial': '✓' if exec_result.partial_success else '-',
                'Retries': exec_result.retry_count
            })
        
        print(tabulate(execution_table, headers='keys', tablefmt='grid'))
        
        # Show aggregated summary
        print(f"\n📝 Aggregated Summary:")
        print(result.summary)
    
    async def demonstrate_learning_improvement(self):
        """Demonstrate Q-learning improvement over time."""
        print("\n\n🧠 DEMONSTRATING LEARNING IMPROVEMENT")
        print("=" * 60)
        
        # Enable Q-learning with higher exploration initially
        self.orchestrator.config['q_learning']['enable_learning'] = True
        self.orchestrator.q_learning_engine.exploration_rate = 0.5
        
        # Test queries that benefit from learning
        learning_queries = [
            "Process data as fast as possible",
            "Reliably retrieve critical information",
            "Export data in the best format"
        ]
        
        print("\n📚 Running learning experiments...")
        
        for query in learning_queries:
            print(f"\n🔍 Query: {query}")
            
            # Track performance over multiple iterations
            iteration_results = []
            
            for i in range(5):
                result = await self.orchestrator.process_user_query(f"{query} - iteration {i}")
                
                iteration_results.append({
                    'Iteration': i + 1,
                    'Tools': ', '.join(result.selected_tools[:2]),  # Show first 2 tools
                    'Time': f"{result.total_time_ms:.0f}ms",
                    'Success': '✅' if result.success else '❌',
                    'ε': f"{self.orchestrator.q_learning_engine.exploration_rate:.2f}"
                })
                
                # Simulate user feedback for learning
                if result.success and result.total_time_ms < 300:
                    # Good performance - positive feedback
                    await self.orchestrator.record_user_feedback(
                        execution_id=self.orchestrator.current_session_id,
                        feedback_type='positive',
                        rating=5
                    )
            
            # Display learning progression
            print(tabulate(iteration_results, headers='keys', tablefmt='grid'))
            
            # Show exploration decay
            print(f"Final exploration rate: {self.orchestrator.q_learning_engine.exploration_rate:.3f}")
    
    async def demonstrate_failure_handling(self):
        """Demonstrate failure handling and learning from failures."""
        print("\n\n🛡️ DEMONSTRATING FAILURE HANDLING")
        print("=" * 60)
        
        # Simulate some tool failures
        print("\n🔥 Simulating tool failures and recovery...")
        
        # Mock some tools to fail occasionally
        original_execute = self.orchestrator.mcp_integration.execute_tool
        fail_count = {'database.sqlite': 0, 'search.comprehensive': 0}
        
        async def mock_with_failures(tool_id, tool_input):
            # Simulate failures for specific tools
            if tool_id == 'database.sqlite' and fail_count['database.sqlite'] < 2:
                fail_count['database.sqlite'] += 1
                raise Exception("Database connection timeout")
            elif tool_id == 'search.comprehensive' and fail_count['search.comprehensive'] < 1:
                fail_count['search.comprehensive'] += 1
                # Partial failure
                error = Exception("Search API rate limited")
                error.partial_result = {'results': ['partial result 1'], 'incomplete': True}
                error.completion_percentage = 0.3
                raise error
            
            # Otherwise execute normally
            return {'status': 'success', 'data': f'Result from {tool_id}'}
        
        self.orchestrator.mcp_integration.execute_tool = mock_with_failures
        
        # Run queries that will experience failures
        failure_test_queries = [
            "Query database for user statistics",
            "Perform comprehensive search analysis",
            "Query database again after learning"
        ]
        
        failure_results = []
        
        for query in failure_test_queries:
            print(f"\n🔍 Executing: {query}")
            result = await self.orchestrator.process_user_query(query)
            
            # Collect failure information
            failures = [r for r in result.execution_results if not r.success]
            partial_successes = [r for r in result.execution_results if r.partial_success]
            
            failure_results.append({
                'Query': query[:30] + '...',
                'Success': '✅' if result.success else '❌',
                'Failures': len(failures),
                'Partial': len(partial_successes),
                'Time': f"{result.total_time_ms:.0f}ms"
            })
            
            # Show detailed failure info
            if failures:
                print("  ⚠️ Failures detected:")
                for f in failures:
                    print(f"    - {f.tool_name}: {f.error} (type: {f.error_type})")
            
            if partial_successes:
                print("  ⚡ Partial successes:")
                for p in partial_successes:
                    print(f"    - {p.tool_name}: {p.completion_percentage:.0%} complete")
        
        # Display failure handling summary
        print("\n📊 Failure Handling Summary:")
        print(tabulate(failure_results, headers='keys', tablefmt='grid'))
        
        # Show learned failure rates
        print("\n📈 Learned Failure Rates:")
        for tool_id, rate in self.orchestrator.failure_metrics['failure_rates'].items():
            if rate > 0:
                print(f"  - {tool_id}: {rate:.2%}")
        
        # Restore original execution
        self.orchestrator.mcp_integration.execute_tool = original_execute
    
    async def run_demo(self):
        """Run the complete demo."""
        try:
            # Setup
            await self.setup()
            
            # Run demonstrations
            await self.demonstrate_tool_selection_strategies()
            await asyncio.sleep(1)  # Brief pause between demos
            
            await self.demonstrate_parallel_execution()
            await asyncio.sleep(1)
            
            await self.demonstrate_result_aggregation()
            await asyncio.sleep(1)
            
            await self.demonstrate_learning_improvement()
            await asyncio.sleep(1)
            
            await self.demonstrate_failure_handling()
            
            # Final summary
            print("\n\n🎉 DEMO COMPLETE!")
            print("=" * 60)
            print("✅ Demonstrated all Orchestrator Agent capabilities:")
            print("  - Tool selection strategies")
            print("  - Parallel execution management")
            print("  - Result aggregation")
            print("  - Q-learning integration")
            print("  - Failure handling and recovery")
            
            # Show final statistics
            if self.orchestrator.execution_history:
                print(f"\n📊 Session Statistics:")
                print(f"  - Total queries processed: {len(self.orchestrator.execution_history)}")
                success_rate = sum(1 for h in self.orchestrator.execution_history if h['success']) / len(self.orchestrator.execution_history)
                print(f"  - Overall success rate: {success_rate:.1%}")
                avg_time = sum(h['execution_time_ms'] for h in self.orchestrator.execution_history) / len(self.orchestrator.execution_history)
                print(f"  - Average execution time: {avg_time:.0f}ms")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise
        finally:
            # Cleanup
            if self.orchestrator:
                await self.orchestrator.shutdown()
            print("\n👋 Goodbye!")


async def main():
    """Run the orchestrator demo."""
    demo = OrchestratorDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("🎭 Orchestrator Agent Comprehensive Demo")
    print("=" * 60)
    asyncio.run(main())