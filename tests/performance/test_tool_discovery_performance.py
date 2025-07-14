"""
Performance tests for Tool Discovery and Pipeline.

Tests the performance of tool discovery algorithms and the complete
pipeline from query to execution.
"""

import pytest
import asyncio
import time
import statistics
import random
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.agents.tool_discovery_agent import ToolDiscoveryAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.intent_models import Intent, IntentResult
from src.core.tool_registry import ToolRegistry
from src.database.tool_registry import ToolRegistryDB


class TestToolDiscoveryPerformance:
    """Performance test cases for Tool Discovery."""
    
    @pytest.fixture
    async def setup_discovery_env(self, tmp_path):
        """Set up tool discovery testing environment."""
        # Create temporary database
        db_path = tmp_path / "test_discovery_perf.db"
        
        # Initialize registry
        registry_db = ToolRegistryDB(str(db_path))
        await registry_db.initialize()
        
        registry = ToolRegistry(str(db_path))
        await registry.initialize()
        
        # Register many tools for realistic testing
        await self._register_test_tools(registry, num_tools=100)
        
        # Create discovery agent
        discovery_agent = ToolDiscoveryAgent(registry)
        
        yield {
            "discovery_agent": discovery_agent,
            "registry": registry,
            "registry_db": registry_db
        }
        
        # Cleanup
        await registry_db.close()
    
    async def _register_test_tools(self, registry: ToolRegistry, num_tools: int):
        """Register test tools with various capabilities."""
        categories = ["file", "database", "api", "compute", "network", "security", "monitoring"]
        operations = ["read", "write", "query", "create", "update", "delete", "analyze", "export"]
        
        for i in range(num_tools):
            category = random.choice(categories)
            ops = random.sample(operations, k=random.randint(2, 5))
            
            tool_id = f"{category}_tool_{i}"
            capabilities = {
                "operations": ops,
                "category": category,
                "performance_tier": random.choice(["fast", "medium", "slow"]),
                "constraints": {
                    "rate_limit": random.randint(10, 1000),
                    "max_size": random.randint(1, 100) * 1024 * 1024
                }
            }
            
            await registry.register_tool(
                tool_id=tool_id,
                name=f"{category.title()} Tool {i}",
                tool_type="mcp",
                endpoint=f"mock://{tool_id}",
                capabilities=capabilities,
                metadata={
                    "version": f"1.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    "reliability": random.choice(["high", "medium", "low"])
                }
            )
            
            # Add some relationships
            if i > 0 and random.random() > 0.7:
                related_tool_id = f"{category}_tool_{random.randint(0, i-1)}"
                await registry.add_tool_relationship(
                    tool_id, 
                    related_tool_id,
                    "complements",
                    strength=random.random()
                )
    
    def create_test_intent(self, intent_type: str = None, keywords: List[str] = None) -> Intent:
        """Create test intent with specified characteristics."""
        if intent_type is None:
            intent_type = random.choice([
                "query.search", "query.retrieve", "action.create",
                "action.modify", "action.delete", "system.monitor"
            ])
        
        if keywords is None:
            keywords = random.sample([
                "find", "search", "get", "create", "update", "delete",
                "analyze", "export", "monitor", "check"
            ], k=random.randint(2, 4))
        
        return Intent(
            type=intent_type,
            confidence=random.uniform(0.7, 0.95),
            entities=keywords,
            keywords=keywords
        )
    
    @pytest.mark.asyncio
    async def test_single_discovery_performance(self, setup_discovery_env):
        """Test performance of single tool discovery operation."""
        discovery_agent = setup_discovery_env["discovery_agent"]
        
        discovery_times = []
        
        for _ in range(50):
            intent = self.create_test_intent()
            
            start_time = time.perf_counter()
            discovered_tools = await discovery_agent.discover_tools(intent, {})
            end_time = time.perf_counter()
            
            discovery_time_ms = (end_time - start_time) * 1000
            discovery_times.append(discovery_time_ms)
            
            # Verify discovery worked
            assert len(discovered_tools) > 0
        
        # Calculate statistics
        avg_time = statistics.mean(discovery_times)
        p95_time = statistics.quantiles(discovery_times, n=20)[18]
        p99_time = max(discovery_times)
        
        print(f"\nSingle Discovery Performance:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  P95: {p95_time:.2f}ms")
        print(f"  P99: {p99_time:.2f}ms")
        
        # Performance requirements
        assert avg_time < 200, f"Average discovery time {avg_time:.2f}ms exceeds 200ms"
        assert p95_time < 200, f"P95 discovery time {p95_time:.2f}ms exceeds 200ms"
    
    @pytest.mark.asyncio
    async def test_discovery_with_large_registry(self, setup_discovery_env):
        """Test discovery performance with large tool registry."""
        discovery_agent = setup_discovery_env["discovery_agent"]
        registry = setup_discovery_env["registry"]
        
        # Add more tools to stress test
        print("\nAdding more tools to registry...")
        await self._register_test_tools(registry, num_tools=400)  # Total 500 tools
        
        # Test discovery with different intent types
        intent_types = ["query.search", "action.create", "system.monitor"]
        results = {}
        
        for intent_type in intent_types:
            discovery_times = []
            
            for _ in range(20):
                intent = self.create_test_intent(intent_type=intent_type)
                
                start_time = time.perf_counter()
                discovered_tools = await discovery_agent.discover_tools(intent, {})
                end_time = time.perf_counter()
                
                discovery_times.append((end_time - start_time) * 1000)
            
            avg_time = statistics.mean(discovery_times)
            results[intent_type] = avg_time
            
            print(f"\n{intent_type} - Average: {avg_time:.2f}ms")
        
        # Performance should be reasonable even with 500 tools
        for intent_type, avg_time in results.items():
            assert avg_time < 300, f"{intent_type} discovery too slow: {avg_time:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_discovery(self, setup_discovery_env):
        """Test concurrent tool discovery performance."""
        discovery_agent = setup_discovery_env["discovery_agent"]
        
        async def discover_batch(intents: List[Intent]) -> float:
            """Discover tools for multiple intents concurrently."""
            start_time = time.perf_counter()
            
            tasks = []
            for intent in intents:
                tasks.append(discovery_agent.discover_tools(intent, {}))
            
            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()
            
            return (end_time - start_time) * 1000, results
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for level in concurrency_levels:
            intents = [self.create_test_intent() for _ in range(level)]
            
            total_time, results = await discover_batch(intents)
            avg_time_per_discovery = total_time / level
            
            print(f"\nConcurrency Level {level}:")
            print(f"  Total Time: {total_time:.2f}ms")
            print(f"  Average per Discovery: {avg_time_per_discovery:.2f}ms")
            print(f"  Throughput: {(level / total_time) * 1000:.2f} discoveries/second")
            
            # All discoveries should succeed
            assert all(len(r) > 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_discovery_algorithm_comparison(self, setup_discovery_env):
        """Compare performance of different discovery algorithms."""
        discovery_agent = setup_discovery_env["discovery_agent"]
        
        # Test intent
        intent = Intent(
            type="query.search",
            confidence=0.85,
            entities=["files", "documents"],
            keywords=["find", "search", "locate"]
        )
        
        # Test different discovery strategies
        strategies = {
            "semantic": discovery_agent._semantic_discovery,
            "keyword": discovery_agent._keyword_discovery,
            "category": discovery_agent._category_discovery,
            "graph": discovery_agent._graph_discovery
        }
        
        results = {}
        
        for name, strategy_func in strategies.items():
            if hasattr(discovery_agent, strategy_func.__name__):
                times = []
                
                for _ in range(20):
                    start_time = time.perf_counter()
                    # Note: This assumes these methods exist and are accessible
                    # In reality, you might need to patch or expose these methods
                    tools = await strategy_func(intent, {})
                    end_time = time.perf_counter()
                    
                    times.append((end_time - start_time) * 1000)
                
                avg_time = statistics.mean(times)
                results[name] = avg_time
                
                print(f"\n{name.title()} Discovery: {avg_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_discovery_caching_impact(self, setup_discovery_env):
        """Test impact of caching on discovery performance."""
        discovery_agent = setup_discovery_env["discovery_agent"]
        
        # Same intent for cache testing
        intent = self.create_test_intent()
        
        # Cold cache runs
        cold_times = []
        for _ in range(5):
            # Clear any caches
            if hasattr(discovery_agent, '_discovery_cache'):
                discovery_agent._discovery_cache.clear()
            
            start_time = time.perf_counter()
            await discovery_agent.discover_tools(intent, {})
            end_time = time.perf_counter()
            
            cold_times.append((end_time - start_time) * 1000)
        
        # Warm cache runs
        warm_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            await discovery_agent.discover_tools(intent, {})
            end_time = time.perf_counter()
            
            warm_times.append((end_time - start_time) * 1000)
        
        avg_cold = statistics.mean(cold_times)
        avg_warm = statistics.mean(warm_times)
        
        print(f"\nDiscovery Cache Impact:")
        print(f"  Cold Cache: {avg_cold:.2f}ms")
        print(f"  Warm Cache: {avg_warm:.2f}ms")
        print(f"  Speedup: {(avg_cold - avg_warm) / avg_cold * 100:.1f}%")


class TestPipelinePerformance:
    """Performance test cases for complete pipeline."""
    
    @pytest.fixture
    async def setup_pipeline_env(self, tmp_path):
        """Set up complete pipeline testing environment."""
        # Create temporary database
        db_path = tmp_path / "test_pipeline_perf.db"
        
        # Initialize components
        registry_db = ToolRegistryDB(str(db_path))
        await registry_db.initialize()
        
        registry = ToolRegistry(str(db_path))
        await registry.initialize()
        
        # Register tools
        await self._register_pipeline_tools(registry)
        
        # Create orchestrator
        orchestrator = OrchestratorAgent(registry)
        
        # Mock tool execution for consistent performance testing
        async def mock_execute(tool_id, params):
            # Simulate execution time based on tool type
            if "fast" in tool_id:
                await asyncio.sleep(0.01)  # 10ms
            elif "slow" in tool_id:
                await asyncio.sleep(0.1)   # 100ms
            else:
                await asyncio.sleep(0.05)  # 50ms
            
            return {"success": True, "result": f"Result from {tool_id}"}
        
        orchestrator.mcp_integration.execute_tool = mock_execute
        
        yield {
            "orchestrator": orchestrator,
            "registry": registry,
            "registry_db": registry_db
        }
        
        # Cleanup
        await registry_db.close()
    
    async def _register_pipeline_tools(self, registry: ToolRegistry):
        """Register tools for pipeline testing."""
        tool_configs = [
            ("fast_search_tool", ["search", "find"], "fast"),
            ("fast_read_tool", ["read", "get"], "fast"),
            ("medium_process_tool", ["process", "analyze"], "medium"),
            ("medium_export_tool", ["export", "save"], "medium"),
            ("slow_compute_tool", ["compute", "calculate"], "slow"),
            ("slow_ml_tool", ["predict", "train"], "slow")
        ]
        
        for tool_id, operations, speed in tool_configs:
            await registry.register_tool(
                tool_id=tool_id,
                name=tool_id.replace("_", " ").title(),
                tool_type="mcp",
                endpoint=f"mock://{tool_id}",
                capabilities={
                    "operations": operations,
                    "speed": speed
                },
                metadata={"performance_tier": speed}
            )
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_performance(self, setup_pipeline_env):
        """Test complete pipeline performance from query to result."""
        orchestrator = setup_pipeline_env["orchestrator"]
        
        test_queries = [
            "Search for documents about Python",
            "Read configuration files from the system",
            "Process and analyze log data",
            "Export results to CSV format",
            "Calculate statistics from the dataset"
        ]
        
        pipeline_times = []
        stage_breakdowns = []
        
        for query in test_queries:
            start_time = time.perf_counter()
            
            # Track stage times
            stage_start = start_time
            
            # Execute pipeline
            result = await orchestrator.process_query(query)
            
            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000
            
            pipeline_times.append(total_time)
            
            # Verify success
            assert result["success"] is True
            
            print(f"\nQuery: '{query[:50]}...'")
            print(f"  Total Time: {total_time:.2f}ms")
            print(f"  Tools Used: {', '.join(result.get('tools_used', []))}")
        
        # Calculate statistics
        avg_time = statistics.mean(pipeline_times)
        p95_time = statistics.quantiles(pipeline_times, n=20)[18] if len(pipeline_times) >= 20 else max(pipeline_times)
        
        print(f"\nPipeline Performance Summary:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  P95: {p95_time:.2f}ms")
        print(f"  Max: {max(pipeline_times):.2f}ms")
        
        # Performance requirements
        assert avg_time < 10000, f"Average pipeline time {avg_time:.2f}ms exceeds 10s"
        assert p95_time < 10000, f"P95 pipeline time {p95_time:.2f}ms exceeds 10s"
    
    @pytest.mark.asyncio
    async def test_pipeline_scalability(self, setup_pipeline_env):
        """Test pipeline scalability with increasing load."""
        orchestrator = setup_pipeline_env["orchestrator"]
        
        load_levels = [1, 2, 5, 10]
        results = {}
        
        for load in load_levels:
            queries = [f"Process data batch {i}" for i in range(load)]
            
            start_time = time.perf_counter()
            
            # Process queries concurrently
            tasks = [orchestrator.process_query(q) for q in queries]
            responses = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000
            
            # Calculate metrics
            throughput = (load / total_time) * 1000  # queries per second
            avg_latency = total_time / load
            
            results[load] = {
                "total_time": total_time,
                "throughput": throughput,
                "avg_latency": avg_latency
            }
            
            print(f"\nLoad Level {load}:")
            print(f"  Total Time: {total_time:.2f}ms")
            print(f"  Throughput: {throughput:.2f} qps")
            print(f"  Avg Latency: {avg_latency:.2f}ms")
            
            # All queries should succeed
            assert all(r["success"] for r in responses)
        
        # Throughput should improve with concurrency
        assert results[5]["throughput"] > results[1]["throughput"]
    
    @pytest.mark.asyncio
    async def test_pipeline_resource_usage(self, setup_pipeline_env):
        """Test pipeline resource usage under load."""
        orchestrator = setup_pipeline_env["orchestrator"]
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline measurements
        initial_cpu = process.cpu_percent(interval=0.1)
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate sustained load
        duration = 5  # seconds
        queries_processed = 0
        start_time = time.time()
        
        cpu_samples = []
        memory_samples = []
        
        while time.time() - start_time < duration:
            # Process query
            await orchestrator.process_query("Analyze system performance metrics")
            queries_processed += 1
            
            # Sample resource usage
            cpu_samples.append(process.cpu_percent(interval=0))
            memory_samples.append(process.memory_info().rss / 1024 / 1024)
            
            await asyncio.sleep(0.05)  # Small delay between queries
        
        # Calculate resource usage statistics
        avg_cpu = statistics.mean(cpu_samples[1:])  # Skip first sample
        max_cpu = max(cpu_samples[1:])
        avg_memory = statistics.mean(memory_samples)
        memory_increase = max(memory_samples) - initial_memory
        
        print(f"\nResource Usage Test:")
        print(f"  Queries Processed: {queries_processed}")
        print(f"  Average CPU: {avg_cpu:.1f}%")
        print(f"  Peak CPU: {max_cpu:.1f}%")
        print(f"  Average Memory: {avg_memory:.1f} MB")
        print(f"  Memory Increase: {memory_increase:.1f} MB")
        
        # Resource usage should be reasonable
        assert avg_cpu < 80, f"Average CPU usage {avg_cpu:.1f}% exceeds 80%"
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f} MB exceeds 100 MB"
    
    @pytest.mark.asyncio
    async def test_pipeline_optimization_impact(self, setup_pipeline_env):
        """Test impact of various optimizations on pipeline performance."""
        orchestrator = setup_pipeline_env["orchestrator"]
        
        # Test query
        query = "Find and process all log files from the last week"
        
        # Baseline - no optimizations
        baseline_times = []
        for _ in range(10):
            start = time.perf_counter()
            await orchestrator.process_query(query)
            baseline_times.append((time.perf_counter() - start) * 1000)
        
        baseline_avg = statistics.mean(baseline_times)
        
        # With caching enabled (simulate)
        orchestrator._enable_caching = True  # Hypothetical flag
        cache_times = []
        for _ in range(10):
            start = time.perf_counter()
            await orchestrator.process_query(query)
            cache_times.append((time.perf_counter() - start) * 1000)
        
        cache_avg = statistics.mean(cache_times)
        
        # With parallel execution (simulate)
        orchestrator._enable_parallel = True  # Hypothetical flag
        parallel_times = []
        for _ in range(10):
            start = time.perf_counter()
            await orchestrator.process_query(query)
            parallel_times.append((time.perf_counter() - start) * 1000)
        
        parallel_avg = statistics.mean(parallel_times)
        
        print(f"\nOptimization Impact:")
        print(f"  Baseline: {baseline_avg:.2f}ms")
        print(f"  With Caching: {cache_avg:.2f}ms ({(baseline_avg - cache_avg) / baseline_avg * 100:.1f}% improvement)")
        print(f"  With Parallel: {parallel_avg:.2f}ms ({(baseline_avg - parallel_avg) / baseline_avg * 100:.1f}% improvement)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])