# Dissertation Test Suite

This directory contains the dissertation-focused test suite designed to validate research hypotheses, demonstrate system capabilities, and provide empirical evidence for the Autonomous Tool Discovery and Integration System dissertation.

## Purpose

This test suite focuses on:
- **Research Validation**: Tests that directly map to dissertation hypotheses
- **Performance Evidence**: Empirical data for claimed performance improvements
- **Algorithm Verification**: Systematic validation of Q-learning, MCP integration, and multi-agent orchestration
- **Reproducibility**: Ensuring consistent, reproducible results for academic publication

## Directory Structure

```
dissertation_test_suite/
├── dissertation-testing-strategy.md    # Comprehensive testing strategy
├── hypothesis_validation/              # Research hypothesis tests
├── performance_benchmarks/             # Performance target validation
├── algorithm_validation/               # Core algorithm verification
├── scenario_demonstrations/            # End-to-end capability demos
└── results/                           # Test outputs and visualizations
```

## Quick Start

### Run All Dissertation Tests
```bash
# Run all dissertation-critical tests
pytest tests/dissertation_test_suite/ -v -m dissertation

# Run with coverage report
pytest tests/dissertation_test_suite/ -v -m dissertation --cov=src --cov-report=html

# Generate performance metrics and visualizations
python -m pytest tests/dissertation_test_suite/ -v --dissertation-metrics
```

### Run Specific Test Categories

```bash
# Hypothesis validation only
pytest tests/dissertation_test_suite/hypothesis_validation/ -v

# Performance benchmarks only
pytest tests/dissertation_test_suite/performance_benchmarks/ -v

# Algorithm validation only
pytest tests/dissertation_test_suite/algorithm_validation/ -v

# Scenario demonstrations
pytest tests/dissertation_test_suite/scenario_demonstrations/ -v
```

## Test Categories

### 1. Hypothesis Validation Tests
Tests that validate core research hypotheses:
- **Learning Improvement**: Q-learning outperforms baseline strategies
- **Convergence Rates**: System converges within 1000 episodes
- **Statistical Significance**: Results are statistically significant (p < 0.05)

### 2. Performance Benchmark Tests
Tests that verify claimed performance targets:
- **Intent Recognition**: <100ms processing time (p95)
- **Tool Selection Accuracy**: >80% accuracy rate
- **System Throughput**: >100 queries per minute
- **Task Completion Rate**: >85% success rate

### 3. Algorithm Validation Tests
Tests that verify core algorithmic components:
- **Q-Learning Engine**: Convergence, stability, and reward improvement
- **Pattern Mining**: Discovery rate and pattern quality
- **Multi-Agent Orchestration**: Coordination and parallel execution

### 4. Scenario Demonstrations
End-to-end tests demonstrating real-world capabilities:
- **Financial Analysis**: Multi-tool coordination for complex queries
- **Adaptive Learning**: System improvement over time
- **Context Persistence**: Maintaining state across interactions

## Key Metrics Collected

### Performance Metrics
- Response time percentiles (p50, p95, p99)
- Throughput (queries per second)
- Resource utilization (CPU, memory)
- API call efficiency

### Learning Metrics
- Convergence episodes
- Cumulative reward curves
- Exploration vs exploitation ratio
- Pattern discovery rate

### Quality Metrics
- Task completion rate
- Tool selection accuracy
- Error recovery rate
- Result quality scores

## Test Markers

Tests are marked with pytest markers for selective execution:

```python
@pytest.mark.dissertation     # Core dissertation tests
@pytest.mark.hypothesis      # Hypothesis validation
@pytest.mark.performance     # Performance benchmarks
@pytest.mark.algorithm       # Algorithm validation
@pytest.mark.scenario        # Scenario demonstrations
@pytest.mark.slow           # Long-running tests (>1 minute)
@pytest.mark.statistical    # Statistical analysis tests
```

## Generating Dissertation Results

### 1. Run Full Evaluation Suite
```bash
# Comprehensive evaluation with all baselines
python demos/demo_baseline_evaluation.py --episodes 1000 --output tests/dissertation_test_suite/results/

# Generate learning curves
python demos/demo_q_learning_orchestration.py --plot --episodes 1000 --output tests/dissertation_test_suite/results/learning_curves/
```

### 2. Generate Visualizations
```bash
# Performance comparison charts
python tests/dissertation_test_suite/scripts/generate_charts.py

# Statistical analysis reports
python tests/dissertation_test_suite/scripts/statistical_analysis.py
```

### 3. Create Dissertation Tables
```bash
# Export metrics to LaTeX/CSV
python tests/dissertation_test_suite/scripts/export_metrics.py --format latex
```

## Success Criteria

Tests in this suite are considered successful when:
1. Q-learning demonstrates >30% improvement over random baseline
2. All performance targets are met (see evaluation-targets.md)
3. Results are statistically significant (p < 0.05, Cohen's d > 0.8)
4. System behavior is consistent across multiple runs
5. All core scenarios complete successfully

## Integration with CI/CD

These tests can be integrated into CI/CD pipelines with:
```yaml
# .github/workflows/dissertation-tests.yml
- name: Run Dissertation Tests
  run: pytest tests/dissertation_test_suite/ -v -m dissertation --junitxml=dissertation-results.xml
```

## Documentation Links

- [Dissertation Testing Strategy](./dissertation-testing-strategy.md)
- [Evaluation Targets](../../docs/evaluation/evaluation-targets.md)
- [Baseline Comparisons](../../docs/evaluation/baseline-comparisons.md)
- [Implementation Status](../../docs/implementation/implementation-status.md)

## Notes

- Tests are designed for reproducibility over exhaustive coverage
- Focus is on positive cases that demonstrate capabilities
- Statistical validation is prioritized over edge case handling
- Results are automatically saved to the `results/` directory for dissertation inclusion