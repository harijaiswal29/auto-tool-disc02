"""
End-to-End Demo: Intent-Based Tool Discovery with Parallel Execution

This demo showcases the complete flow from natural language query to
intent recognition, tool discovery, and parallel execution.

Key features demonstrated:
1. Different intents lead to different tool selections
2. Tools are discovered based on intent-capability mapping
3. Multiple tools execute in parallel for better performance
4. Visual representation of the discovery and execution process
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime
from typing import List, Dict, Any
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.orchestrator_agent import OrchestratorAgent
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IntentDiscoveryParallelDemo:
    """Demo showcasing intent-based discovery and parallel execution."""
    
    def __init__(self):
        """Initialize demo with configuration."""
        self.config = {
            'orchestration': {
                'max_tools_per_query': 5,
                'tool_selection_strategy': 'performance_weighted',
                'parallel_execution': True  # Enable parallel execution
            },
            'intent_recognition': {
                'model': 'all-MiniLM-L6-v2',
                'confidence_threshold': 0.7
            },
            'database': {
                'tool_registry': 'data/registry/tools.db'
            }
        }
        
        self.orchestrator = None
        self.registry = None
        
        # Demo queries showcasing different intents
        self.demo_queries = [
            # Search intent
            {
                'query': 'Search for Python machine learning tutorials and documentation',
                'expected_intent': 'query.search',
                'description': 'Search intent - should discover search and web tools'
            },
            # Create intent
            {
                'query': 'Create a new database table for user analytics and generate a report file',
                'expected_intent': 'action.create',
                'description': 'Create intent - should discover database and file creation tools'
            },
            # Analyze intent
            {
                'query': 'Analyze system performance metrics and database query patterns',
                'expected_intent': 'query.analyze',
                'description': 'Analyze intent - should discover analysis and monitoring tools'
            },
            # Complex multi-intent
            {
                'query': 'Search for financial data, analyze trends, and save results to database',
                'expected_intent': 'mixed',
                'description': 'Multi-intent - should discover search, analysis, and storage tools'
            },
            # Monitor intent
            {
                'query': 'Monitor system resources and track database performance',
                'expected_intent': 'system.monitor',
                'description': 'Monitor intent - should discover monitoring and tracking tools'
            }
        ]
        
        self.execution_times = {}
        self.discovery_results = []
    
    async def setup(self):
        """Set up the demo environment."""
        print("\n" + "="*80)
        print("🚀 Intent-Based Tool Discovery with Parallel Execution Demo")
        print("="*80)
        
        print("\n📋 Setting up environment...")
        
        # Initialize tool registry
        self.registry = ToolRegistry(self.config['database']['tool_registry'])
        
        # Register demo tools if needed
        existing_tools = await self.registry.get_all_tools()
        if len(existing_tools) < 10:
            print("📝 Registering comprehensive demo tools...")
            await self._register_comprehensive_tools()
        else:
            print(f"✅ Found {len(existing_tools)} existing tools")
        
        # Initialize orchestrator
        print("🎯 Initializing Orchestrator with parallel execution enabled...")
        self.orchestrator = OrchestratorAgent(self.config)
        await self.orchestrator.initialize()
        
        # Mock MCP integration for demo
        await self._setup_mock_mcp()
        
        print("✅ Setup complete!")
    
    async def _register_comprehensive_tools(self):
        """Register a comprehensive set of tools for demo."""
        tools = [
            # Search tools
            {'id': 'search.web', 'name': 'Web Search Engine', 'type': 'search',
             'capabilities': {'operations': ['search', 'find', 'query', 'discover']},
             'performance_score': 0.88},
            {'id': 'search.academic', 'name': 'Academic Search', 'type': 'search',
             'capabilities': {'operations': ['search', 'research', 'study']},
             'performance_score': 0.92},
            {'id': 'search.code', 'name': 'Code Search', 'type': 'search',
             'capabilities': {'operations': ['search', 'find', 'analyze']},
             'performance_score': 0.85},
            
            # Database tools
            {'id': 'db.query', 'name': 'Database Query Engine', 'type': 'database',
             'capabilities': {'operations': ['query', 'retrieve', 'fetch', 'analyze']},
             'performance_score': 0.94},
            {'id': 'db.create', 'name': 'Database Creator', 'type': 'database',
             'capabilities': {'operations': ['create', 'insert', 'write']},
             'performance_score': 0.90},
            {'id': 'db.analyze', 'name': 'Database Analyzer', 'type': 'database',
             'capabilities': {'operations': ['analyze', 'profile', 'inspect']},
             'performance_score': 0.87},
            
            # File system tools
            {'id': 'file.read', 'name': 'File Reader', 'type': 'filesystem',
             'capabilities': {'operations': ['read', 'retrieve', 'access']},
             'performance_score': 0.96},
            {'id': 'file.write', 'name': 'File Writer', 'type': 'filesystem',
             'capabilities': {'operations': ['write', 'create', 'save', 'generate']},
             'performance_score': 0.93},
            {'id': 'file.analyze', 'name': 'File Analyzer', 'type': 'filesystem',
             'capabilities': {'operations': ['analyze', 'inspect', 'examine']},
             'performance_score': 0.89},
            
            # Analysis tools
            {'id': 'analyze.data', 'name': 'Data Analyzer', 'type': 'analytics',
             'capabilities': {'operations': ['analyze', 'evaluate', 'assess']},
             'performance_score': 0.91},
            {'id': 'analyze.trends', 'name': 'Trend Analyzer', 'type': 'analytics',
             'capabilities': {'operations': ['analyze', 'predict', 'forecast']},
             'performance_score': 0.86},
            
            # System tools
            {'id': 'system.monitor', 'name': 'System Monitor', 'type': 'system',
             'capabilities': {'operations': ['monitor', 'track', 'watch', 'observe']},
             'performance_score': 0.95},
            {'id': 'system.config', 'name': 'System Configurator', 'type': 'system',
             'capabilities': {'operations': ['configure', 'setup', 'initialize']},
             'performance_score': 0.88}
        ]
        
        for tool in tools:
            tool['server'] = 'mock_server'
            tool['status'] = 'active'
            tool['capabilities'] = json.dumps(tool['capabilities'])
            await self.registry.register_tool(tool)
        
        print(f"✅ Registered {len(tools)} demo tools")
    
    async def _setup_mock_mcp(self):
        """Set up mock MCP integration for demo."""
        from unittest.mock import AsyncMock
        
        # Define execution times for different tools (in seconds)
        tool_execution_times = {
            'search.web': 0.3,
            'search.academic': 0.5,
            'search.code': 0.4,
            'db.query': 0.2,
            'db.create': 0.3,
            'db.analyze': 0.6,
            'file.read': 0.1,
            'file.write': 0.2,
            'file.analyze': 0.4,
            'analyze.data': 0.7,
            'analyze.trends': 0.8,
            'system.monitor': 0.3,
            'system.config': 0.2
        }
        
        async def mock_execute_tool(tool_id: str, tool_input: Any):
            """Mock tool execution with realistic delays."""
            exec_time = tool_execution_times.get(tool_id, 0.3)
            await asyncio.sleep(exec_time)
            
            # Return mock results based on tool type
            if 'search' in tool_id:
                return {
                    'results': [f'Result 1 from {tool_id}', f'Result 2 from {tool_id}'],
                    'count': 2,
                    'execution_time': exec_time
                }
            elif 'db' in tool_id:
                return {
                    'status': 'success',
                    'rows_affected': 10,
                    'data': {'sample': 'database data'},
                    'execution_time': exec_time
                }
            elif 'file' in tool_id:
                return {
                    'status': 'complete',
                    'file_path': f'/demo/{tool_id}_output.txt',
                    'size': 1024,
                    'execution_time': exec_time
                }
            elif 'analyze' in tool_id:
                return {
                    'analysis': {'trend': 'upward', 'confidence': 0.85},
                    'metrics': {'accuracy': 0.92, 'coverage': 0.88},
                    'execution_time': exec_time
                }
            else:
                return {
                    'status': 'monitored',
                    'metrics': {'cpu': 45, 'memory': 62},
                    'execution_time': exec_time
                }
        
        mock_mcp = AsyncMock()
        mock_mcp.execute_tool = mock_execute_tool
        mock_mcp.registry = self.registry
        self.orchestrator.mcp_integration = mock_mcp
    
    async def run_demo(self):
        """Run the complete demo."""
        print("\n" + "="*80)
        print("🎬 Starting Demo: Processing queries with different intents")
        print("="*80)
        
        for i, demo_case in enumerate(self.demo_queries, 1):
            print(f"\n\n{'='*80}")
            print(f"📌 Demo Case {i}: {demo_case['description']}")
            print(f"{'='*80}")
            print(f"Query: \"{demo_case['query']}\"")
            
            # Process the query
            start_time = time.time()
            result = await self.orchestrator.process_user_query(demo_case['query'])
            total_time = time.time() - start_time
            
            # Display results
            self._display_results(result, total_time)
            
            # Store for visualization
            self.discovery_results.append({
                'case': i,
                'query': demo_case['query'],
                'intent': result.intent.primary_intent.type,
                'discovered_count': len(result.discovered_tools),
                'selected_count': len(result.selected_tools),
                'execution_count': len(result.execution_results),
                'total_time': total_time,
                'parallel_time': max([r.execution_time_ms/1000 for r in result.execution_results]) if result.execution_results else 0,
                'tools': [t['name'] for t in result.discovered_tools[:5]]
            })
            
            # Pause between demos
            await asyncio.sleep(1)
        
        # Show performance comparison
        await self._show_performance_comparison()
        
        # Generate visualization
        self._generate_visualization()
    
    def _display_results(self, result, total_time):
        """Display orchestration results in a formatted way."""
        print(f"\n🎯 Intent Recognition:")
        print(f"   Primary Intent: {result.intent.primary_intent.type}")
        print(f"   Confidence: {result.intent.primary_intent.confidence:.2%}")
        print(f"   Keywords: {', '.join(result.intent.primary_intent.keywords[:5])}")
        
        print(f"\n🔍 Tool Discovery:")
        print(f"   Discovered {len(result.discovered_tools)} relevant tools")
        print(f"\n   Top 5 discovered tools:")
        tool_table = []
        for tool in result.discovered_tools[:5]:
            tool_table.append([
                tool['name'],
                tool['type'],
                f"{tool.get('relevance_score', 0):.3f}",
                tool.get('performance_score', 0)
            ])
        print(tabulate(tool_table, headers=['Tool Name', 'Type', 'Relevance', 'Performance'], 
                      tablefmt='grid'))
        
        print(f"\n⚡ Tool Execution (Parallel Mode):")
        print(f"   Selected {len(result.selected_tools)} tools for execution")
        
        if result.execution_results:
            exec_table = []
            for exec_result in result.execution_results:
                status = "✅" if exec_result.success else "❌"
                exec_table.append([
                    exec_result.tool_name,
                    status,
                    f"{exec_result.execution_time_ms:.0f}ms",
                    "Success" if exec_result.success else exec_result.error[:30]
                ])
            print(tabulate(exec_table, headers=['Tool', 'Status', 'Time', 'Result'], 
                          tablefmt='grid'))
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Total Processing Time: {total_time:.3f}s")
        if result.execution_results:
            parallel_time = max([r.execution_time_ms/1000 for r in result.execution_results])
            sequential_time = sum([r.execution_time_ms/1000 for r in result.execution_results])
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1
            
            print(f"   Parallel Execution Time: {parallel_time:.3f}s")
            print(f"   Sequential Would Be: {sequential_time:.3f}s")
            print(f"   Speedup: {speedup:.2f}x")
    
    async def _show_performance_comparison(self):
        """Show performance comparison between parallel and sequential execution."""
        print("\n\n" + "="*80)
        print("📈 Performance Comparison: Parallel vs Sequential Execution")
        print("="*80)
        
        # Switch to sequential mode
        self.orchestrator.config['orchestration']['parallel_execution'] = False
        
        comparison_query = "Search for data, analyze patterns, and save results"
        
        # Sequential execution
        print("\n🐌 Sequential Execution Mode:")
        start_time = time.time()
        seq_result = await self.orchestrator.process_user_query(comparison_query)
        seq_time = time.time() - start_time
        
        # Switch back to parallel mode
        self.orchestrator.config['orchestration']['parallel_execution'] = True
        
        # Parallel execution
        print("\n⚡ Parallel Execution Mode:")
        start_time = time.time()
        par_result = await self.orchestrator.process_user_query(comparison_query)
        par_time = time.time() - start_time
        
        # Display comparison
        print(f"\n📊 Execution Time Comparison:")
        print(f"   Sequential: {seq_time:.3f}s")
        print(f"   Parallel: {par_time:.3f}s")
        print(f"   Speedup: {seq_time/par_time:.2f}x")
        print(f"   Time Saved: {seq_time - par_time:.3f}s ({((seq_time - par_time)/seq_time)*100:.1f}%)")
    
    def _generate_visualization(self):
        """Generate visualization of the demo results."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Intent-Based Tool Discovery with Parallel Execution', fontsize=16)
            
            # 1. Tools discovered by intent type
            intents = [r['intent'] for r in self.discovery_results]
            discovered_counts = [r['discovered_count'] for r in self.discovery_results]
            
            ax1.bar(range(len(intents)), discovered_counts, color='skyblue')
            ax1.set_xticks(range(len(intents)))
            ax1.set_xticklabels([f"Case {i+1}" for i in range(len(intents))], rotation=45)
            ax1.set_ylabel('Tools Discovered')
            ax1.set_title('Tools Discovered per Query')
            
            # Add intent labels
            for i, (intent, count) in enumerate(zip(intents, discovered_counts)):
                ax1.text(i, count + 0.5, intent.split('.')[-1], ha='center', fontsize=8)
            
            # 2. Execution time comparison
            cases = [f"Case {r['case']}" for r in self.discovery_results]
            total_times = [r['total_time'] for r in self.discovery_results]
            
            ax2.plot(cases, total_times, 'o-', color='green', linewidth=2, markersize=8)
            ax2.set_ylabel('Total Time (seconds)')
            ax2.set_title('Query Processing Time')
            ax2.grid(True, alpha=0.3)
            
            # 3. Selected vs Executed tools
            selected = [r['selected_count'] for r in self.discovery_results]
            executed = [r['execution_count'] for r in self.discovery_results]
            
            x = np.arange(len(cases))
            width = 0.35
            
            ax3.bar(x - width/2, selected, width, label='Selected', color='orange')
            ax3.bar(x + width/2, executed, width, label='Executed', color='green')
            ax3.set_xticks(x)
            ax3.set_xticklabels(cases, rotation=45)
            ax3.set_ylabel('Number of Tools')
            ax3.set_title('Selected vs Executed Tools')
            ax3.legend()
            
            # 4. Speedup visualization (if we have parallel times)
            if any(r['parallel_time'] > 0 for r in self.discovery_results):
                speedups = []
                for r in self.discovery_results:
                    if r['execution_count'] > 1 and r['parallel_time'] > 0:
                        sequential_estimate = r['execution_count'] * r['parallel_time']
                        speedup = sequential_estimate / r['parallel_time']
                        speedups.append(speedup)
                    else:
                        speedups.append(1.0)
                
                ax4.bar(range(len(speedups)), speedups, color='purple')
                ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
                ax4.set_xticks(range(len(speedups)))
                ax4.set_xticklabels([f"Case {i+1}" for i in range(len(speedups))], rotation=45)
                ax4.set_ylabel('Speedup Factor')
                ax4.set_title('Parallel Execution Speedup')
                ax4.set_ylim(0, max(speedups) + 0.5)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = 'demos/intent_discovery_parallel_demo.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Visualization saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate visualization: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.shutdown()


async def main():
    """Run the demo."""
    demo = IntentDiscoveryParallelDemo()
    
    try:
        await demo.setup()
        await demo.run_demo()
        
        print("\n\n" + "="*80)
        print("✅ Demo Complete!")
        print("="*80)
        print("\nKey Takeaways:")
        print("1. Different intents automatically discover relevant tools")
        print("2. Tools are scored and ranked by relevance to the intent")
        print("3. Multiple tools execute in parallel for better performance")
        print("4. The system adapts tool selection based on query understanding")
        print("="*80)
        
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())