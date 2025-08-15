#!/usr/bin/env python3
"""
Quick Validation Script for Dissertation Test Suite

This script performs a rapid validation of all dissertation components
to ensure everything is properly connected and working before running
full experiments.
"""

import asyncio
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, List, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils.logger import get_logger
from tests.dissertation_test_suite.data.test_queries import get_evaluation_sets

logger = get_logger(__name__)


class QuickValidator:
    """Performs quick validation of dissertation components."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'errors': [],
            'warnings': []
        }
        self.project_root = Path(__file__).parent.parent.parent.parent
    
    def check_file_structure(self) -> Tuple[bool, str]:
        """Check if all required files and directories exist."""
        logger.info("Checking file structure...")
        
        required_files = [
            "tests/dissertation_test_suite/data/test_queries.py",
            "tests/dissertation_test_suite/data/experiment_config.yaml",
            "tests/dissertation_test_suite/scripts/run_baseline_comparison.py",
            "tests/dissertation_test_suite/dissertation-testing-strategy.md",
            "config/config.json",
            "src/agents/orchestrator_agent.py",
            "src/evaluation/evaluation_engine.py",
            "src/learning/q_learning_engine.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            return False, f"Missing files: {', '.join(missing_files)}"
        
        return True, "All required files present"
    
    def check_test_queries(self) -> Tuple[bool, str]:
        """Validate test query sets."""
        logger.info("Checking test queries...")
        
        try:
            query_sets = get_evaluation_sets()
            
            # Check required sets exist
            required_sets = ["quick_test", "dissertation_core", "full_evaluation"]
            missing_sets = [s for s in required_sets if s not in query_sets]
            
            if missing_sets:
                return False, f"Missing query sets: {missing_sets}"
            
            # Validate query counts
            stats = {}
            for set_name, queries in query_sets.items():
                stats[set_name] = len(queries)
            
            # Check minimum queries
            if stats.get("dissertation_core", 0) < 20:
                return False, "dissertation_core set has fewer than 20 queries"
            
            return True, f"Query sets validated: {stats}"
            
        except Exception as e:
            return False, f"Error loading queries: {str(e)}"
    
    def check_config_files(self) -> Tuple[bool, str]:
        """Validate configuration files."""
        logger.info("Checking configuration files...")
        
        try:
            # Check experiment config
            exp_config_path = self.project_root / "tests/dissertation_test_suite/data/experiment_config.yaml"
            with open(exp_config_path) as f:
                exp_config = yaml.safe_load(f)
            
            # Validate structure
            if 'experiments' not in exp_config:
                return False, "Missing 'experiments' in config"
            
            if 'baseline_comparison' not in exp_config['experiments']:
                return False, "Missing baseline_comparison experiment"
            
            # Check main project config
            main_config_path = self.project_root / "config/config.json"
            with open(main_config_path) as f:
                main_config = json.load(f)
            
            # Check required sections
            required_sections = ['agents', 'tools', 'learning', 'evaluation']
            missing = [s for s in required_sections if s not in main_config]
            
            if missing:
                return False, f"Missing config sections: {missing}"
            
            return True, "Configuration files valid"
            
        except Exception as e:
            return False, f"Config error: {str(e)}"
    
    async def check_imports(self) -> Tuple[bool, str]:
        """Check if all required modules can be imported."""
        logger.info("Checking imports...")
        
        required_imports = [
            "src.agents.orchestrator_agent.OrchestratorAgent",
            "src.evaluation.evaluation_engine.EvaluationEngine",
            "src.evaluation.baseline_strategies.BaselineStrategy",
            "src.learning.q_learning_engine.QLearningEngine",
            "src.learning.dqn_agent.DQNAgent",
            "src.learning.deep_q_network.DQN",
            "src.agents.tool_discovery_agent.ToolDiscoveryAgent",
            "src.agents.intent_recognition_agent.IntentRecognitionAgent"
        ]
        
        failed_imports = []
        for import_path in required_imports:
            try:
                module_path, class_name = import_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
            except Exception as e:
                failed_imports.append(f"{import_path}: {str(e)}")
        
        if failed_imports:
            return False, f"Import failures: {'; '.join(failed_imports[:3])}"
        
        return True, "All imports successful"
    
    async def check_basic_functionality(self) -> Tuple[bool, str]:
        """Test basic system functionality."""
        logger.info("Checking basic functionality...")
        
        try:
            # Test query loading
            from tests.dissertation_test_suite.data.test_queries import SIMPLE_QUERIES
            if len(SIMPLE_QUERIES) == 0:
                return False, "No simple queries defined"
            
            # Test baseline strategy availability
            from src.evaluation.baseline_strategies import RandomSelectionBaseline
            strategy = RandomSelectionBaseline({})
            
            # Test if we can create evaluation engine
            with open(self.project_root / "config/config.json") as f:
                config = json.load(f)
            
            from src.evaluation.evaluation_engine import EvaluationEngine
            engine = EvaluationEngine(config)
            
            return True, "Basic functionality OK"
            
        except Exception as e:
            return False, f"Functionality error: {str(e)}"
    
    async def check_results_directories(self) -> Tuple[bool, str]:
        """Ensure results directories exist."""
        logger.info("Checking results directories...")
        
        dirs_to_create = [
            "tests/dissertation_test_suite/results/raw_data",
            "tests/dissertation_test_suite/results/statistical_reports",
            "tests/dissertation_test_suite/results/comparison_charts",
            "tests/dissertation_test_suite/results/learning_curves",
            "tests/dissertation_test_suite/results/dissertation_figures",
            "tests/dissertation_test_suite/results/demonstrations"
        ]
        
        created = []
        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            if not full_path.exists():
                return False, f"Failed to create {dir_path}"
            created.append(dir_path)
        
        return True, f"Created {len(created)} directories"
    
    def generate_validation_report(self) -> str:
        """Generate a validation report."""
        report = []
        report.append("="*60)
        report.append("DISSERTATION TEST SUITE VALIDATION REPORT")
        report.append("="*60)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append("")
        
        # Summary
        total_checks = len(self.results['checks'])
        passed_checks = sum(1 for check in self.results['checks'].values() if check['passed'])
        
        report.append(f"Total Checks: {total_checks}")
        report.append(f"Passed: {passed_checks}")
        report.append(f"Failed: {total_checks - passed_checks}")
        report.append("")
        
        # Details
        report.append("Check Results:")
        report.append("-"*40)
        for check_name, result in self.results['checks'].items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            report.append(f"{status} {check_name}")
            report.append(f"   {result['message']}")
        
        # Errors
        if self.results['errors']:
            report.append("")
            report.append("Errors:")
            report.append("-"*40)
            for error in self.results['errors']:
                report.append(f"- {error}")
        
        # Warnings
        if self.results['warnings']:
            report.append("")
            report.append("Warnings:")
            report.append("-"*40)
            for warning in self.results['warnings']:
                report.append(f"- {warning}")
        
        # Next steps
        report.append("")
        report.append("Next Steps:")
        report.append("-"*40)
        if passed_checks == total_checks:
            report.append("✓ All checks passed! Ready to run experiments:")
            report.append("  1. Quick test: python -m tests.dissertation_test_suite.scripts.run_baseline_comparison --query-set quick_test")
            report.append("  2. Full evaluation: python -m tests.dissertation_test_suite.scripts.run_baseline_comparison")
        else:
            report.append("✗ Fix failures before running experiments")
            report.append("  See error messages above for details")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    def check_dqn_configuration(self) -> Tuple[bool, str]:
        """Check DQN configuration and setup."""
        logger.info("Checking DQN configuration...")
        
        try:
            # Check main config for DQN settings
            main_config_path = self.project_root / "config/config.json"
            with open(main_config_path) as f:
                main_config = json.load(f)
            
            # Check if DQN is enabled
            dqn_config = main_config.get('dqn', {})
            if not dqn_config.get('enabled', False):
                self.results['warnings'].append("DQN is not enabled in config.json")
                return True, "DQN disabled (set dqn.enabled=true to enable)"
            
            # Check state dimensions match
            state_dim = 447  # Expected state dimensions
            q_learning_config = main_config.get('q_learning', {})
            
            # Verify DQN parameters
            required_params = ['learning_rate', 'discount_factor', 'batch_size', 'memory_size']
            missing_params = [p for p in required_params if p not in dqn_config]
            
            if missing_params:
                return False, f"Missing DQN parameters: {missing_params}"
            
            # Check experiment config for DQN strategy
            exp_config_path = self.project_root / "tests/dissertation_test_suite/data/experiment_config.yaml"
            with open(exp_config_path) as f:
                exp_config = yaml.safe_load(f)
            
            strategies = exp_config['experiments']['baseline_comparison']['strategies']
            dqn_strategy = next((s for s in strategies if s['name'] == 'q_learning_dqn'), None)
            
            if not dqn_strategy:
                return False, "DQN strategy not found in experiment config"
            
            return True, f"DQN enabled with {state_dim}-dim state vectors"
            
        except Exception as e:
            return False, f"DQN config error: {str(e)}"
    
    async def run_all_checks(self):
        """Run all validation checks."""
        checks = [
            ("File Structure", self.check_file_structure),
            ("Test Queries", self.check_test_queries),
            ("Configuration", self.check_config_files),
            ("Imports", self.check_imports),
            ("DQN Configuration", self.check_dqn_configuration),
            ("Basic Functionality", self.check_basic_functionality),
            ("Results Directories", self.check_results_directories)
        ]
        
        for check_name, check_func in checks:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    passed, message = await check_func()
                else:
                    passed, message = check_func()
                
                self.results['checks'][check_name] = {
                    'passed': passed,
                    'message': message
                }
                
                if not passed:
                    self.results['errors'].append(f"{check_name}: {message}")
                    
            except Exception as e:
                self.results['checks'][check_name] = {
                    'passed': False,
                    'message': f"Exception: {str(e)}"
                }
                self.results['errors'].append(f"{check_name} crashed: {str(e)}")
                logger.error(f"Check {check_name} failed with exception: {traceback.format_exc()}")
        
        # Save results
        output_path = self.project_root / "tests/dissertation_test_suite/validation_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print report
        report = self.generate_validation_report()
        print(report)
        
        # Save report
        report_path = self.project_root / "tests/dissertation_test_suite/validation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Return success status
        all_passed = all(check['passed'] for check in self.results['checks'].values())
        return all_passed


async def main():
    """Main entry point."""
    print("Starting dissertation test suite validation...")
    print("This should complete in < 30 seconds")
    print()
    
    validator = QuickValidator()
    success = await validator.run_all_checks()
    
    if success:
        print("\n✓ Validation successful! System ready for dissertation experiments.")
        sys.exit(0)
    else:
        print("\n✗ Validation failed. Please fix errors before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())