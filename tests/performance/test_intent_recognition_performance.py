"""
Performance tests for Intent Recognition.

Tests the performance characteristics of the intent recognition system
including processing time, throughput, and scalability.
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import random
import string

from src.agents.intent_recognition_agent import IntentRecognitionAgent
from src.agents.intent_models import Intent, IntentResult
from src.monitoring.intent_recognition_metrics import get_metrics


class TestIntentRecognitionPerformance:
    """Performance test cases for Intent Recognition."""
    
    @pytest.fixture
    def intent_agent(self):
        """Create intent recognition agent."""
        agent = IntentRecognitionAgent()
        # Reset metrics for clean performance testing
        metrics = get_metrics()
        metrics.reset_metrics()
        return agent
    
    @pytest.fixture
    def sample_queries(self):
        """Generate sample queries for testing."""
        queries = [
            # Simple queries
            "Find all Python files",
            "Search for documentation",
            "List recent changes",
            "Show configuration files",
            "Get database records",
            
            # Medium complexity queries
            "Find all Python files modified in the last week",
            "Search for documentation about API endpoints",
            "List recent changes made by John",
            "Show configuration files in the config directory",
            "Get database records for active users",
            
            # Complex queries
            "Find all Python files modified in the last week and search for TODO comments",
            "Search for documentation about API endpoints and create a summary report",
            "List recent changes made by John and analyze the impact on performance",
            "Show configuration files in the config directory and validate their syntax",
            "Get database records for active users and export to CSV format",
            
            # Long queries
            "Find all Python files in the project directory that have been modified in the last 30 days and contain references to deprecated functions that need to be updated",
            "Search through all documentation files to find references to the authentication system and compile a comprehensive guide for new developers joining the team",
        ]
        return queries
    
    def generate_random_query(self, length: int = None) -> str:
        """Generate random query of specified length."""
        if length is None:
            length = random.randint(10, 100)
        
        # Common words for more realistic queries
        verbs = ["find", "search", "get", "list", "show", "create", "update", "delete", "analyze", "export"]
        nouns = ["files", "documents", "records", "data", "users", "configuration", "logs", "reports", "metrics", "results"]
        adjectives = ["recent", "all", "active", "modified", "new", "old", "important", "critical", "pending", "completed"]
        
        words = []
        for _ in range(length // 10):
            words.extend([
                random.choice(verbs),
                random.choice(adjectives),
                random.choice(nouns)
            ])
        
        return " ".join(words[:length // 4])  # Approximate word count
    
    @pytest.mark.asyncio
    async def test_single_query_performance(self, intent_agent, sample_queries):
        """Test performance of single query processing."""
        processing_times = []
        
        for query in sample_queries[:10]:  # Test first 10 queries
            start_time = time.perf_counter()
            result = await intent_agent.process_query(query)
            end_time = time.perf_counter()
            
            processing_time_ms = (end_time - start_time) * 1000
            processing_times.append(processing_time_ms)
            
            # Verify result
            assert result.confidence_passed
            assert result.processing_time_ms > 0
        
        # Calculate statistics
        avg_time = statistics.mean(processing_times)
        p95_time = statistics.quantiles(processing_times, n=20)[18] if len(processing_times) >= 20 else max(processing_times)
        p99_time = max(processing_times)
        
        # Performance assertions
        assert avg_time < 100, f"Average processing time {avg_time:.2f}ms exceeds 100ms requirement"
        assert p95_time < 100, f"P95 processing time {p95_time:.2f}ms exceeds 100ms requirement"
        assert p99_time < 200, f"P99 processing time {p99_time:.2f}ms exceeds reasonable threshold"
        
        print(f"\nSingle Query Performance:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  P95: {p95_time:.2f}ms")
        print(f"  P99: {p99_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, intent_agent):
        """Test performance under concurrent load."""
        concurrent_levels = [1, 5, 10, 20]
        results = {}
        
        async def process_batch(queries: List[str]) -> List[float]:
            """Process a batch of queries concurrently."""
            tasks = []
            for query in queries:
                tasks.append(intent_agent.process_query(query))
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()
            
            total_time = (end_time - start_time) * 1000
            return [r.processing_time_ms for r in results], total_time
        
        for level in concurrent_levels:
            # Generate queries
            queries = [self.generate_random_query() for _ in range(level)]
            
            # Process concurrently
            processing_times, total_time = await process_batch(queries)
            
            # Calculate metrics
            avg_time = statistics.mean(processing_times)
            throughput = (level / total_time) * 1000  # queries per second
            
            results[level] = {
                "avg_time_ms": avg_time,
                "total_time_ms": total_time,
                "throughput_qps": throughput
            }
            
            print(f"\nConcurrency Level {level}:")
            print(f"  Average Time: {avg_time:.2f}ms")
            print(f"  Total Time: {total_time:.2f}ms")
            print(f"  Throughput: {throughput:.2f} queries/second")
        
        # Verify scalability
        assert results[10]["throughput_qps"] > results[1]["throughput_qps"], "Throughput should improve with concurrency"
    
    @pytest.mark.asyncio
    async def test_cache_performance_impact(self, intent_agent):
        """Test the performance impact of caching."""
        # Use the same query multiple times
        test_query = "Find all Python files in the project directory"
        
        # First run - cold cache
        cold_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = await intent_agent.process_query(test_query)
            end_time = time.perf_counter()
            cold_times.append((end_time - start_time) * 1000)
            
            # Clear cache to ensure cold start
            if hasattr(intent_agent, '_embedding_cache'):
                intent_agent._embedding_cache.clear()
        
        # Second run - warm cache
        warm_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = await intent_agent.process_query(test_query)
            end_time = time.perf_counter()
            warm_times.append((end_time - start_time) * 1000)
        
        avg_cold = statistics.mean(cold_times)
        avg_warm = statistics.mean(warm_times)
        cache_speedup = (avg_cold - avg_warm) / avg_cold * 100
        
        print(f"\nCache Performance Impact:")
        print(f"  Cold Cache Average: {avg_cold:.2f}ms")
        print(f"  Warm Cache Average: {avg_warm:.2f}ms")
        print(f"  Cache Speedup: {cache_speedup:.1f}%")
        
        # Cache should provide significant speedup
        assert avg_warm < avg_cold, "Warm cache should be faster than cold cache"
        assert cache_speedup > 20, f"Cache speedup {cache_speedup:.1f}% is less than expected 20%"
    
    @pytest.mark.asyncio
    async def test_query_length_impact(self, intent_agent):
        """Test how query length affects performance."""
        length_ranges = [
            (1, 5, "Very Short"),
            (5, 10, "Short"),
            (10, 25, "Medium"),
            (25, 50, "Long"),
            (50, 100, "Very Long")
        ]
        
        results = {}
        
        for min_words, max_words, label in length_ranges:
            processing_times = []
            
            # Test multiple queries in each range
            for _ in range(10):
                word_count = random.randint(min_words, max_words)
                query = " ".join([self.generate_random_query(10) for _ in range(word_count // 10)])
                
                start_time = time.perf_counter()
                result = await intent_agent.process_query(query)
                end_time = time.perf_counter()
                
                processing_times.append((end_time - start_time) * 1000)
            
            avg_time = statistics.mean(processing_times)
            results[label] = avg_time
            
            print(f"\n{label} Queries ({min_words}-{max_words} words):")
            print(f"  Average Time: {avg_time:.2f}ms")
        
        # Longer queries should take more time, but not proportionally
        assert results["Very Long"] > results["Very Short"]
        assert results["Very Long"] < results["Very Short"] * 10, "Processing time should not scale linearly with length"
    
    @pytest.mark.asyncio
    async def test_pipeline_stage_performance(self, intent_agent):
        """Test performance of individual pipeline stages."""
        test_queries = [self.generate_random_query() for _ in range(20)]
        
        # Enable detailed stage timing
        stage_timings = {
            "text_preprocessor": [],
            "tokenizer": [],
            "feature_extractor": [],
            "intent_classifier": [],
            "context_enricher": [],
            "confidence_scorer": [],
            "state_manager": []
        }
        
        for query in test_queries:
            # Process query and get stage timings
            result = await intent_agent.process_query(query)
            
            # Get metrics to access stage timings
            metrics = get_metrics()
            stage_perf = metrics.get_stage_performance()
            
            # Collect stage timings
            for stage, perf in stage_perf.items():
                if stage in stage_timings:
                    stage_timings[stage].append(perf["avg_ms"])
        
        # Analyze stage performance
        print("\nPipeline Stage Performance:")
        total_time = 0
        for stage, times in stage_timings.items():
            if times:
                avg_time = statistics.mean(times)
                total_time += avg_time
                print(f"  {stage}: {avg_time:.2f}ms")
        
        print(f"  Total Pipeline: {total_time:.2f}ms")
        
        # Feature extraction should be the most expensive stage
        if stage_timings["feature_extractor"]:
            feature_time = statistics.mean(stage_timings["feature_extractor"])
            assert feature_time > 10, "Feature extraction should take significant time"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, intent_agent):
        """Test memory efficiency with large batches."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large batch of queries
        large_batch = [self.generate_random_query() for _ in range(1000)]
        
        for i, query in enumerate(large_batch):
            await intent_agent.process_query(query)
            
            # Check memory every 100 queries
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                print(f"\nAfter {i} queries: Memory increase: {memory_increase:.2f} MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"\nMemory Efficiency Test:")
        print(f"  Initial Memory: {initial_memory:.2f} MB")
        print(f"  Final Memory: {final_memory:.2f} MB")
        print(f"  Total Increase: {total_increase:.2f} MB")
        
        # Memory increase should be reasonable
        assert total_increase < 500, f"Memory increase {total_increase:.2f} MB exceeds 500 MB threshold"
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, intent_agent):
        """Test performance under sustained load."""
        duration_seconds = 10
        queries_processed = 0
        processing_times = []
        errors = 0
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            query = self.generate_random_query()
            
            try:
                query_start = time.perf_counter()
                result = await intent_agent.process_query(query)
                query_end = time.perf_counter()
                
                processing_times.append((query_end - query_start) * 1000)
                queries_processed += 1
                
            except Exception as e:
                errors += 1
                print(f"Error processing query: {e}")
        
        # Calculate results
        total_time = time.time() - start_time
        throughput = queries_processed / total_time
        avg_time = statistics.mean(processing_times)
        p95_time = statistics.quantiles(processing_times, n=20)[18]
        
        print(f"\nSustained Load Test ({duration_seconds}s):")
        print(f"  Queries Processed: {queries_processed}")
        print(f"  Throughput: {throughput:.2f} queries/second")
        print(f"  Average Time: {avg_time:.2f}ms")
        print(f"  P95 Time: {p95_time:.2f}ms")
        print(f"  Errors: {errors}")
        
        # Performance requirements
        assert throughput > 10, f"Throughput {throughput:.2f} qps is below 10 qps requirement"
        assert avg_time < 100, f"Average time {avg_time:.2f}ms exceeds 100ms requirement"
        assert errors == 0, f"Had {errors} errors during sustained load"
    
    @pytest.mark.asyncio
    async def test_performance_degradation(self, intent_agent):
        """Test for performance degradation over time."""
        batches = 5
        queries_per_batch = 100
        batch_times = []
        
        for batch_num in range(batches):
            batch_processing_times = []
            
            for _ in range(queries_per_batch):
                query = self.generate_random_query()
                start_time = time.perf_counter()
                result = await intent_agent.process_query(query)
                end_time = time.perf_counter()
                
                batch_processing_times.append((end_time - start_time) * 1000)
            
            avg_batch_time = statistics.mean(batch_processing_times)
            batch_times.append(avg_batch_time)
            
            print(f"\nBatch {batch_num + 1} Average: {avg_batch_time:.2f}ms")
        
        # Check for degradation
        first_batch_avg = batch_times[0]
        last_batch_avg = batch_times[-1]
        degradation = ((last_batch_avg - first_batch_avg) / first_batch_avg) * 100
        
        print(f"\nPerformance Degradation Test:")
        print(f"  First Batch Average: {first_batch_avg:.2f}ms")
        print(f"  Last Batch Average: {last_batch_avg:.2f}ms")
        print(f"  Degradation: {degradation:.1f}%")
        
        # Performance should not degrade significantly
        assert degradation < 20, f"Performance degraded by {degradation:.1f}%, exceeds 20% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])