#!/usr/bin/env python3
"""Test system throughput target of >100 queries per minute.

This test validates that the system can handle the required query load
while maintaining performance and accuracy standards.
"""

import pytest
import numpy as np
import asyncio
import time
from typing import Dict, List, Tuple
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.orchestration.orchestrator_agent import OrchestratorAgent
from src.intent.enhanced_intent_pipeline import EnhancedIntentPipeline
from src.agents.tool_discovery_agent import ToolDiscoveryAgent
from src.execution.parallel_execution_engine import ParallelExecutionEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.dissertation
@pytest.mark.performance
@pytest.mark.asyncio
class TestSystemThroughput:
    """Test suite for system throughput requirements."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create full orchestration system."""
        config = {
            'max_concurrent_queries': 20,
            'timeout_seconds': 30,
            'enable_caching': True,
            'performance_monitoring': True
        }
        
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        
        yield orchestrator
        
        await orchestrator.cleanup()
    
    @pytest.fixture
    def workload_queries(self):
        """Realistic workload of diverse queries."""
        # Mix of different query types to simulate real usage
        return [
            # File operations (25%)
            "Find all Python files in src",
            "Search for TODO comments",
            "List recent file changes",
            "Find large files over 1MB",
            "Search for import statements",
            
            # Database queries (25%)
            "Query users table",
            "Get sales data for last month",
            "Find inactive customers",
            "Show database schema",
            "Count records in products table",
            
            # Web search (20%)
            "Search Python asyncio tutorials",
            "Find MCP documentation",
            "Get latest AI news",
            "Search for code examples",
            
            # Analytics (20%)
            "Analyze code complexity",
            "Generate test coverage report",
            "Find security vulnerabilities",
            "Calculate code metrics",
            
            # Quick queries (10%)
            "Get current time",
            "Show version",
            "List available tools",
            "Check system status"
        ] * 5  # Repeat to get 100 queries
    
    async def test_sustained_throughput(self, orchestrator, workload_queries):
        """Test sustained throughput over 1 minute."""
        logger.info("Testing sustained throughput (100 queries/minute)")
        
        # Shuffle queries for realistic distribution
        queries = workload_queries.copy()
        np.random.shuffle(queries)
        
        # Metrics tracking
        start_time = time.time()
        completed_queries = 0
        failed_queries = 0
        response_times = []
        
        # Resource monitoring
        resource_monitor = ResourceMonitor()
        resource_monitor.start()
        
        # Process queries for 60 seconds
        target_duration = 60  # seconds
        target_queries = 100
        
        async def process_query(query):
            nonlocal completed_queries, failed_queries
            
            query_start = time.perf_counter()
            try:
                result = await orchestrator.process_query(query)
                response_time = (time.perf_counter() - query_start) * 1000
                
                if result and result.get('success', False):
                    completed_queries += 1
                    response_times.append(response_time)
                else:
                    failed_queries += 1
                    
            except Exception as e:
                failed_queries += 1
                logger.error(f"Query failed: {e}")
        
        # Submit queries at controlled rate
        tasks = []
        query_interval = target_duration / target_queries  # 0.6 seconds per query
        
        for i, query in enumerate(queries[:target_queries]):
            # Submit query
            task = asyncio.create_task(process_query(query))
            tasks.append(task)
            
            # Wait for interval (unless it's the last query)
            if i < target_queries - 1:
                await asyncio.sleep(query_interval)
        
        # Wait for all queries to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate final metrics
        elapsed_time = time.time() - start_time
        actual_throughput = completed_queries / elapsed_time * 60  # queries per minute
        success_rate = completed_queries / (completed_queries + failed_queries)
        
        # Stop resource monitoring
        resource_stats = resource_monitor.stop()
        
        # Calculate response time percentiles
        if response_times:
            p50 = np.percentile(response_times, 50)
            p95 = np.percentile(response_times, 95)
            p99 = np.percentile(response_times, 99)
        else:
            p50 = p95 = p99 = 0
        
        logger.info(f"Completed: {completed_queries} queries in {elapsed_time:.1f}s")
        logger.info(f"Throughput: {actual_throughput:.1f} queries/minute")
        logger.info(f"Success rate: {success_rate:.1%}")
        logger.info(f"Response times - p50: {p50:.0f}ms, p95: {p95:.0f}ms, p99: {p99:.0f}ms")
        logger.info(f"Peak CPU: {resource_stats['peak_cpu']:.1f}%, "
                   f"Peak Memory: {resource_stats['peak_memory']:.1f}MB")
        
        # Assert requirements
        assert actual_throughput >= 100, \
            f"Throughput {actual_throughput:.1f} < 100 queries/minute"
        assert success_rate >= 0.95, \
            f"Success rate {success_rate:.1%} < 95%"
        assert p95 <= 1000, \
            f"p95 response time {p95:.0f}ms > 1000ms"
        assert resource_stats['peak_cpu'] < 80, \
            f"CPU usage {resource_stats['peak_cpu']:.1f}% >= 80%"
        
        # Save results
        self._save_results({
            'test': 'sustained_throughput',
            'duration_seconds': elapsed_time,
            'completed_queries': completed_queries,
            'failed_queries': failed_queries,
            'throughput_qpm': actual_throughput,
            'success_rate': success_rate,
            'response_times': {
                'p50': p50,
                'p95': p95,
                'p99': p99,
                'mean': np.mean(response_times) if response_times else 0
            },
            'resource_usage': resource_stats
        })
    
    async def test_burst_throughput(self, orchestrator):
        """Test handling of burst traffic."""
        logger.info("Testing burst throughput handling")
        
        # Burst scenarios
        burst_configs = [
            {'queries': 20, 'duration': 5},    # 240 qpm rate
            {'queries': 50, 'duration': 15},   # 200 qpm rate
            {'queries': 100, 'duration': 30},  # 200 qpm rate
        ]
        
        results = []
        
        for config in burst_configs:
            num_queries = config['queries']
            duration = config['duration']
            
            logger.info(f"Testing burst: {num_queries} queries in {duration}s")
            
            # Generate burst queries
            burst_queries = [
                f"Quick query {i} for burst test" for i in range(num_queries)
            ]
            
            # Track metrics
            start_time = time.time()
            completed = 0
            response_times = []
            
            # Submit all queries as fast as possible
            tasks = []
            for query in burst_queries:
                async def process_with_timing(q):
                    nonlocal completed
                    query_start = time.perf_counter()
                    try:
                        result = await orchestrator.process_query(q)
                        if result and result.get('success', False):
                            completed += 1
                            response_time = (time.perf_counter() - query_start) * 1000
                            response_times.append(response_time)
                    except Exception as e:
                        logger.error(f"Burst query failed: {e}")
                
                task = asyncio.create_task(process_with_timing(query))
                tasks.append(task)
            
            # Wait for completion or timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=duration
                )
            except asyncio.TimeoutError:
                logger.warning(f"Burst test timed out after {duration}s")
            
            # Calculate metrics
            elapsed = time.time() - start_time
            throughput = completed / elapsed * 60
            avg_response = np.mean(response_times) if response_times else 0
            
            burst_result = {
                'config': config,
                'completed': completed,
                'throughput_qpm': throughput,
                'avg_response_ms': avg_response,
                'elapsed_seconds': elapsed
            }
            results.append(burst_result)
            
            logger.info(f"Burst completed: {completed}/{num_queries} queries, "
                       f"{throughput:.1f} qpm")
            
            # Allow system to recover between bursts
            await asyncio.sleep(5)
        
        # All burst scenarios should maintain >100 qpm
        min_throughput = min(r['throughput_qpm'] for r in results)
        assert min_throughput >= 100, \
            f"Minimum burst throughput {min_throughput:.1f} < 100 qpm"
        
        # Save results
        self._save_results({
            'test': 'burst_throughput',
            'burst_results': results,
            'min_throughput_qpm': min_throughput
        })
    
    async def test_concurrent_user_simulation(self, orchestrator, workload_queries):
        """Simulate multiple concurrent users."""
        logger.info("Testing concurrent user simulation")
        
        # Simulate different numbers of concurrent users
        user_counts = [1, 5, 10, 20]
        results = []
        
        for num_users in user_counts:
            logger.info(f"Simulating {num_users} concurrent users")
            
            # Each user submits queries at a realistic rate
            queries_per_user = 20
            user_query_interval = 3.0  # seconds between queries per user
            
            start_time = time.time()
            completed_queries = 0
            all_response_times = []
            
            async def simulate_user(user_id):
                nonlocal completed_queries
                user_queries = workload_queries[
                    user_id * queries_per_user:(user_id + 1) * queries_per_user
                ]
                
                for query in user_queries:
                    query_start = time.perf_counter()
                    try:
                        result = await orchestrator.process_query(
                            f"User{user_id}: {query}"
                        )
                        if result and result.get('success', False):
                            completed_queries += 1
                            response_time = (time.perf_counter() - query_start) * 1000
                            all_response_times.append(response_time)
                    except Exception as e:
                        logger.error(f"User {user_id} query failed: {e}")
                    
                    # Wait before next query
                    await asyncio.sleep(user_query_interval)
            
            # Start all users concurrently
            user_tasks = [
                asyncio.create_task(simulate_user(i))
                for i in range(num_users)
            ]
            
            # Wait for all users to complete
            await asyncio.gather(*user_tasks, return_exceptions=True)
            
            # Calculate metrics
            elapsed = time.time() - start_time
            throughput = completed_queries / elapsed * 60
            
            if all_response_times:
                avg_response = np.mean(all_response_times)
                p95_response = np.percentile(all_response_times, 95)
            else:
                avg_response = p95_response = 0
            
            user_result = {
                'num_users': num_users,
                'completed_queries': completed_queries,
                'throughput_qpm': throughput,
                'avg_response_ms': avg_response,
                'p95_response_ms': p95_response,
                'elapsed_seconds': elapsed
            }
            results.append(user_result)
            
            logger.info(f"{num_users} users: {throughput:.1f} qpm, "
                       f"avg response: {avg_response:.0f}ms")
        
        # System should scale with users
        max_users_result = results[-1]  # 20 users
        assert max_users_result['throughput_qpm'] >= 100, \
            f"Throughput with {max_users_result['num_users']} users " \
            f"({max_users_result['throughput_qpm']:.1f} qpm) < 100"
        
        # Save results
        self._save_results({
            'test': 'concurrent_users',
            'user_scaling_results': results
        })
    
    async def test_mixed_workload_performance(self, orchestrator):
        """Test performance with mixed query complexity."""
        logger.info("Testing mixed workload performance")
        
        # Define workload mix
        workload_mix = {
            'simple': {
                'queries': [
                    "Get current time",
                    "Show version",
                    "List tools"
                ],
                'weight': 0.3,
                'expected_time_ms': 50
            },
            'medium': {
                'queries': [
                    "Find Python files",
                    "Query user table",
                    "Search documentation"
                ],
                'weight': 0.5,
                'expected_time_ms': 200
            },
            'complex': {
                'queries': [
                    "Analyze code and find bugs",
                    "Search web and summarize results",
                    "Query database and generate report"
                ],
                'weight': 0.2,
                'expected_time_ms': 500
            }
        }
        
        # Generate mixed workload
        mixed_queries = []
        for complexity, config in workload_mix.items():
            count = int(100 * config['weight'])
            for _ in range(count):
                query = np.random.choice(config['queries'])
                mixed_queries.append((query, complexity))
        
        np.random.shuffle(mixed_queries)
        
        # Process workload
        start_time = time.time()
        complexity_metrics = {
            'simple': {'count': 0, 'times': []},
            'medium': {'count': 0, 'times': []},
            'complex': {'count': 0, 'times': []}
        }
        
        tasks = []
        for query, complexity in mixed_queries:
            async def process_typed_query(q, c):
                query_start = time.perf_counter()
                try:
                    result = await orchestrator.process_query(q)
                    if result and result.get('success', False):
                        response_time = (time.perf_counter() - query_start) * 1000
                        complexity_metrics[c]['count'] += 1
                        complexity_metrics[c]['times'].append(response_time)
                except Exception as e:
                    logger.error(f"Mixed workload query failed: {e}")
            
            task = asyncio.create_task(process_typed_query(query, complexity))
            tasks.append(task)
        
        # Process with controlled concurrency
        semaphore = asyncio.Semaphore(20)
        
        async def limited_task(task):
            async with semaphore:
                await task
        
        await asyncio.gather(*[limited_task(task) for task in tasks])
        
        # Calculate metrics
        elapsed = time.time() - start_time
        total_completed = sum(m['count'] for m in complexity_metrics.values())
        throughput = total_completed / elapsed * 60
        
        # Analyze by complexity
        complexity_analysis = {}
        for complexity, metrics in complexity_metrics.items():
            if metrics['times']:
                complexity_analysis[complexity] = {
                    'count': metrics['count'],
                    'avg_ms': np.mean(metrics['times']),
                    'p95_ms': np.percentile(metrics['times'], 95)
                }
        
        logger.info(f"Mixed workload throughput: {throughput:.1f} qpm")
        for complexity, analysis in complexity_analysis.items():
            logger.info(f"{complexity}: {analysis['count']} queries, "
                       f"avg: {analysis['avg_ms']:.0f}ms")
        
        # Should maintain throughput even with mixed complexity
        assert throughput >= 100, \
            f"Mixed workload throughput {throughput:.1f} < 100 qpm"
        
        # Save results
        self._save_results({
            'test': 'mixed_workload',
            'throughput_qpm': throughput,
            'total_queries': total_completed,
            'elapsed_seconds': elapsed,
            'complexity_breakdown': complexity_analysis
        })
    
    def _save_results(self, results: Dict):
        """Save test results for dissertation."""
        output_dir = Path(__file__).parent.parent / "results" / "performance_metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"system_throughput_{results.get('test', 'general')}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / filename}")


class ResourceMonitor:
    """Monitor system resources during tests."""
    
    def __init__(self):
        self.monitoring = False
        self.peak_cpu = 0
        self.peak_memory = 0
        self.samples = []
        self._thread = None
    
    def start(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.peak_cpu = 0
        self.peak_memory = 0
        self.samples = []
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring and return stats."""
        self.monitoring = False
        if self._thread:
            self._thread.join()
        
        return {
            'peak_cpu': self.peak_cpu,
            'peak_memory': self.peak_memory,
            'avg_cpu': np.mean([s['cpu'] for s in self.samples]) if self.samples else 0,
            'avg_memory': np.mean([s['memory'] for s in self.samples]) if self.samples else 0,
            'num_samples': len(self.samples)
        }
    
    def _monitor_loop(self):
        """Monitor resources in background."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = process.cpu_percent(interval=0.1)
                
                # Memory usage in MB
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Update peaks
                self.peak_cpu = max(self.peak_cpu, cpu_percent)
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                # Store sample
                self.samples.append({
                    'cpu': cpu_percent,
                    'memory': memory_mb,
                    'timestamp': time.time()
                })
                
                time.sleep(1)  # Sample every second
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])