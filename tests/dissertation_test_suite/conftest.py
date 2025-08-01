"""Pytest configuration for dissertation test suite.

This file configures pytest markers and fixtures specific to dissertation testing.
"""

import pytest
import asyncio
import json
import logging
from pathlib import Path
import numpy as np
import random
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging for dissertation tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Register custom markers
def pytest_configure(config):
    """Register custom markers for dissertation tests."""
    config.addinivalue_line(
        "markers", "dissertation: Core tests for dissertation validation"
    )
    config.addinivalue_line(
        "markers", "hypothesis: Tests validating research hypotheses"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "algorithm: Algorithm validation tests"
    )
    config.addinivalue_line(
        "markers", "statistical: Statistical significance tests"
    )
    config.addinivalue_line(
        "markers", "scenario: End-to-end scenario demonstrations"
    )
    config.addinivalue_line(
        "markers", "slow: Long-running tests (>1 minute)"
    )
    config.addinivalue_line(
        "markers", "reproducible: Tests requiring fixed random seeds"
    )


# Dissertation-specific fixtures
@pytest.fixture(scope="session")
def dissertation_config():
    """Load dissertation-specific configuration."""
    config_path = Path(__file__).parent / "dissertation_config.json"
    
    default_config = {
        "random_seed": 42,
        "num_runs": 30,  # For statistical significance
        "confidence_level": 0.95,
        "min_improvement": 0.30,  # 30% improvement threshold
        "performance_targets": {
            "intent_recognition_ms": 100,
            "tool_selection_accuracy": 0.80,
            "throughput_qpm": 100,
            "convergence_episodes": 1000
        },
        "output_dir": "tests/dissertation_test_suite/results"
    }
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            default_config.update(user_config)
    
    return default_config


@pytest.fixture
def set_random_seed(dissertation_config):
    """Set random seeds for reproducibility."""
    seed = dissertation_config["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    
    # If using PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    yield seed


@pytest.fixture
def results_collector(dissertation_config):
    """Collect and save test results for dissertation."""
    results = {}
    
    def collect(test_name: str, data: dict):
        """Collect results from a test."""
        results[test_name] = data
    
    yield collect
    
    # Save all results after tests complete
    output_dir = Path(dissertation_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = Path("test_results_latest.json")
    with open(output_dir / timestamp, 'w') as f:
        json.dump(results, f, indent=2)


@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = []
        
        def start(self):
            """Start monitoring."""
            self.start_time = time.perf_counter()
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.metrics = []
        
        def checkpoint(self, label: str):
            """Record a checkpoint."""
            elapsed = time.perf_counter() - self.start_time
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = current_memory - self.start_memory
            
            self.metrics.append({
                'label': label,
                'elapsed_seconds': elapsed,
                'memory_mb': current_memory,
                'memory_delta_mb': memory_delta,
                'cpu_percent': process.cpu_percent(interval=0.1)
            })
        
        def get_summary(self):
            """Get performance summary."""
            if not self.metrics:
                return {}
            
            total_time = self.metrics[-1]['elapsed_seconds']
            peak_memory = max(m['memory_mb'] for m in self.metrics)
            
            return {
                'total_time_seconds': total_time,
                'peak_memory_mb': peak_memory,
                'checkpoints': self.metrics
            }
    
    monitor = PerformanceMonitor()
    yield monitor


@pytest.fixture
def mock_mcp_servers():
    """Mock MCP servers for dissertation tests."""
    # Import mock servers
    from src.tools.mock_filesystem import MockFilesystemServer
    from src.tools.mock_search import MockSearchServer
    from src.tools.mock_sqlite import MockSQLiteServer
    
    servers = {
        'filesystem': MockFilesystemServer(),
        'search': MockSearchServer(),
        'sqlite': MockSQLiteServer()
    }
    
    # Start servers
    for server in servers.values():
        asyncio.create_task(server.start())
    
    yield servers
    
    # Cleanup
    for server in servers.values():
        asyncio.create_task(server.stop())


# Hook for dissertation-specific metrics
def pytest_runtest_makereport(item, call):
    """Add dissertation metrics to test reports."""
    if call.when == "call" and call.excinfo is None:
        # Test passed - check if it has dissertation metrics
        if hasattr(item, "dissertation_metrics"):
            # Add metrics to report
            call.dissertation_metrics = item.dissertation_metrics


# Command-line options
def pytest_addoption(parser):
    """Add dissertation-specific command-line options."""
    parser.addoption(
        "--dissertation-metrics",
        action="store_true",
        help="Generate detailed metrics for dissertation"
    )
    parser.addoption(
        "--num-runs",
        action="store",
        default=30,
        type=int,
        help="Number of runs for statistical tests (default: 30)"
    )
    parser.addoption(
        "--episodes",
        action="store",
        default=1000,
        type=int,
        help="Number of episodes for learning tests (default: 1000)"
    )