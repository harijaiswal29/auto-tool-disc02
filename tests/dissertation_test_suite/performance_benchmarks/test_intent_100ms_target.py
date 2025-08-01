#!/usr/bin/env python3
"""Test H2: Intent recognition performs within 100ms (p95) target.

This test validates that the intent recognition system meets the performance
requirements while maintaining accuracy above 90%.
"""

import pytest
import numpy as np
import time
import asyncio
from typing import Dict, List, Tuple
import json
from pathlib import Path
from dataclasses import dataclass
import statistics

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.intent.intent_recognition import IntentRecognition
from src.intent.enhanced_intent_pipeline import EnhancedIntentPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceResult:
    """Performance measurement result."""
    query: str
    processing_time_ms: float
    intent: str
    confidence: float
    cache_hit: bool


@pytest.mark.dissertation
@pytest.mark.performance
@pytest.mark.asyncio
class TestIntent100msTarget:
    """Test suite for intent recognition performance requirements."""
    
    @pytest.fixture
    async def intent_pipeline(self):
        """Create intent recognition pipeline."""
        config = {
            'model_name': 'all-MiniLM-L6-v2',
            'cache_enabled': True,
            'cache_size': 1000,
            'confidence_threshold': 0.7,
            'max_workers': 4
        }
        pipeline = EnhancedIntentPipeline(config)
        await pipeline.initialize()
        
        # Warm up the model
        await pipeline.process("warm up query")
        
        yield pipeline
        await pipeline.cleanup()
    
    @pytest.fixture
    def test_queries(self):
        """Comprehensive test queries covering different intents."""
        return [
            # File operations
            "Find all Python files in the src directory",
            "Search for TODO comments in code",
            "List files modified in the last week",
            "Show me the largest files in the project",
            
            # Database queries
            "Query the database for active users",
            "Get sales data from last month",
            "Find customers with pending orders",
            "Show database schema for products table",
            
            # Web search
            "Search for Python asyncio tutorials",
            "Find documentation on MCP protocol",
            "Get latest news about AI developments",
            "Look up weather forecast for tomorrow",
            
            # Code analysis
            "Analyze code complexity in main module",
            "Find security vulnerabilities in the code",
            "Generate test coverage report",
            "Check for code style violations",
            
            # Multi-intent
            "Search files and analyze their complexity",
            "Query database and export to CSV",
            "Find documentation and summarize key points",
            
            # Edge cases
            "a",  # Very short query
            "Find all Python files with TODO comments that need to be addressed before the next release and generate a report summarizing the technical debt",  # Long query
            "开发文档在哪里",  # Non-English
            "SELECT * FROM users WHERE active = true",  # SQL-like
        ]
    
    async def test_p95_performance_target(self, intent_pipeline, test_queries):
        """Test that 95% of queries complete within 100ms."""
        logger.info("Testing H2: 100ms p95 performance target")
        
        # Run multiple iterations for statistical validity
        num_iterations = 10
        all_results = []
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Randomize query order
            queries = test_queries.copy()
            np.random.shuffle(queries)
            
            for query in queries:
                start_time = time.perf_counter()
                
                result = await intent_pipeline.process(query)
                
                end_time = time.perf_counter()
                processing_time_ms = (end_time - start_time) * 1000
                
                perf_result = PerformanceResult(
                    query=query,
                    processing_time_ms=processing_time_ms,
                    intent=result.get('intent', 'unknown'),
                    confidence=result.get('confidence', 0.0),
                    cache_hit=result.get('cache_hit', False)
                )
                all_results.append(perf_result)
        
        # Calculate percentiles
        processing_times = [r.processing_time_ms for r in all_results]
        p50 = np.percentile(processing_times, 50)
        p95 = np.percentile(processing_times, 95)
        p99 = np.percentile(processing_times, 99)
        
        # Separate cache hits and misses
        cache_hits = [r for r in all_results if r.cache_hit]
        cache_misses = [r for r in all_results if not r.cache_hit]
        
        cache_hit_times = [r.processing_time_ms for r in cache_hits] if cache_hits else [0]
        cache_miss_times = [r.processing_time_ms for r in cache_misses] if cache_misses else [0]
        
        logger.info(f"Overall - p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")
        logger.info(f"Cache hits: {len(cache_hits)}, avg: {np.mean(cache_hit_times):.2f}ms")
        logger.info(f"Cache misses: {len(cache_misses)}, avg: {np.mean(cache_miss_times):.2f}ms")
        
        # Assert performance requirements
        assert p95 <= 100, f"p95 ({p95:.2f}ms) exceeds 100ms target"
        assert p50 <= 50, f"p50 ({p50:.2f}ms) should be well under target"
        
        # Save detailed results
        self._save_results({
            'test': 'p95_performance',
            'num_queries': len(all_results),
            'percentiles': {
                'p50': p50,
                'p95': p95,
                'p99': p99
            },
            'cache_performance': {
                'hit_rate': len(cache_hits) / len(all_results),
                'hit_avg_ms': np.mean(cache_hit_times),
                'miss_avg_ms': np.mean(cache_miss_times)
            },
            'summary_stats': {
                'mean': np.mean(processing_times),
                'std': np.std(processing_times),
                'min': np.min(processing_times),
                'max': np.max(processing_times)
            }
        })
    
    async def test_accuracy_at_speed(self, intent_pipeline, test_queries):
        """Test that accuracy remains >90% while meeting speed requirements."""
        logger.info("Testing H2: Accuracy at speed")
        
        # Define expected intents for validation
        expected_intents = {
            "Find all Python files in the src directory": "search_files",
            "Query the database for active users": "query_data",
            "Search for Python asyncio tutorials": "web_search",
            "Analyze code complexity in main module": "analyze_code",
            "Search files and analyze their complexity": "multi_intent"
        }
        
        results = []
        correct_predictions = 0
        
        for query, expected in expected_intents.items():
            start_time = time.perf_counter()
            result = await intent_pipeline.process(query)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            predicted_intent = result.get('intent', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            # Check if prediction matches expected (or is acceptable alternative)
            is_correct = (predicted_intent == expected or 
                         (expected == "multi_intent" and "multi" in predicted_intent))
            
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'query': query,
                'expected': expected,
                'predicted': predicted_intent,
                'correct': is_correct,
                'confidence': confidence,
                'time_ms': processing_time
            })
            
            # Ensure this query met performance target
            assert processing_time <= 150, \
                f"Query '{query}' took {processing_time:.2f}ms"
        
        accuracy = correct_predictions / len(expected_intents)
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_time = np.mean([r['time_ms'] for r in results])
        
        logger.info(f"Accuracy: {accuracy:.1%}")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        logger.info(f"Average time: {avg_time:.2f}ms")
        
        # Assert requirements
        assert accuracy >= 0.90, f"Accuracy {accuracy:.1%} < 90%"
        assert avg_confidence >= 0.70, f"Confidence {avg_confidence:.3f} < 0.70"
        
        # Save results
        self._save_results({
            'test': 'accuracy_at_speed',
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_time_ms': avg_time,
            'detailed_results': results
        })
    
    async def test_scalability_under_load(self, intent_pipeline):
        """Test performance under concurrent load."""
        logger.info("Testing H2: Scalability under load")
        
        # Test queries for load testing
        load_queries = [
            "Find Python files",
            "Search database",
            "Analyze code",
            "Get weather data",
            "Run security scan"
        ] * 20  # 100 total queries
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        results = {}
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing with concurrency: {concurrency}")
            
            start_time = time.perf_counter()
            
            # Process queries with limited concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def process_with_semaphore(query):
                async with semaphore:
                    query_start = time.perf_counter()
                    result = await intent_pipeline.process(query)
                    query_time = (time.perf_counter() - query_start) * 1000
                    return query_time
            
            # Run all queries concurrently
            processing_times = await asyncio.gather(*[
                process_with_semaphore(query) for query in load_queries
            ])
            
            total_time = time.perf_counter() - start_time
            throughput = len(load_queries) / total_time
            
            p95_concurrent = np.percentile(processing_times, 95)
            
            results[concurrency] = {
                'throughput_qps': throughput,
                'p95_ms': p95_concurrent,
                'total_time_s': total_time
            }
            
            logger.info(f"Throughput: {throughput:.1f} queries/sec")
            logger.info(f"p95: {p95_concurrent:.2f}ms")
            
            # Even under load, p95 should stay reasonable
            assert p95_concurrent <= 200, \
                f"p95 under load ({p95_concurrent:.2f}ms) too high"
        
        # Verify throughput meets requirements (>100 qps)
        max_throughput = max(r['throughput_qps'] for r in results.values())
        assert max_throughput >= 100, \
            f"Max throughput {max_throughput:.1f} < 100 qps"
        
        # Save results
        self._save_results({
            'test': 'scalability_under_load',
            'concurrency_results': results,
            'max_throughput_qps': max_throughput
        })
    
    async def test_cold_start_performance(self, intent_pipeline):
        """Test performance on cold start (first query)."""
        logger.info("Testing H2: Cold start performance")
        
        # Create a new pipeline instance (cold)
        cold_pipeline = EnhancedIntentPipeline({
            'model_name': 'all-MiniLM-L6-v2',
            'cache_enabled': False  # Disable cache for true cold start
        })
        
        # Time initialization
        init_start = time.perf_counter()
        await cold_pipeline.initialize()
        init_time = (time.perf_counter() - init_start) * 1000
        
        # First query (cold start)
        first_query = "Find all Python files in the project"
        cold_start = time.perf_counter()
        cold_result = await cold_pipeline.process(first_query)
        cold_time = (time.perf_counter() - cold_start) * 1000
        
        # Subsequent queries (warm)
        warm_times = []
        for i in range(5):
            warm_start = time.perf_counter()
            await cold_pipeline.process(f"Query number {i}")
            warm_time = (time.perf_counter() - warm_start) * 1000
            warm_times.append(warm_time)
        
        avg_warm_time = np.mean(warm_times)
        
        logger.info(f"Initialization time: {init_time:.2f}ms")
        logger.info(f"Cold start time: {cold_time:.2f}ms")
        logger.info(f"Average warm time: {avg_warm_time:.2f}ms")
        
        # Cold start should be reasonable (allowing for model loading)
        assert cold_time <= 500, f"Cold start {cold_time:.2f}ms too slow"
        
        # Warm queries should meet target
        assert avg_warm_time <= 100, f"Warm queries {avg_warm_time:.2f}ms > 100ms"
        
        # Cleanup
        await cold_pipeline.cleanup()
        
        # Save results
        self._save_results({
            'test': 'cold_start_performance',
            'init_time_ms': init_time,
            'cold_start_ms': cold_time,
            'avg_warm_ms': avg_warm_time,
            'warmup_factor': cold_time / avg_warm_time
        })
    
    def _save_results(self, results: Dict):
        """Save test results for dissertation."""
        output_dir = Path(__file__).parent.parent / "results" / "performance_metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"h2_intent_performance_{results.get('test', 'general')}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / filename}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])