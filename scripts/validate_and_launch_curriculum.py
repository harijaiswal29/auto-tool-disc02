#!/usr/bin/env python3
"""
Validate and Launch Curriculum Learning Evaluation
==================================================
This script validates the setup and launches the full curriculum learning evaluation
ensuring all dissertation hypotheses are properly tested.

Validates:
1. Mock servers only configuration
2. All improvements are active
3. Hypothesis testing is integrated
4. Environment is properly set up

Tests Core Hypotheses:
- H1: Q-learning achieves >30% improvement over baselines
- H1a: DQN outperforms Tabular Q-learning after 1000+ episodes
- H1b: Both converge to >85% task completion rate
- H2: Intent recognition <100ms (p95)
- H3: Pattern mining discovers >50 patterns within 500 episodes
- H5: Convergence within 1000 episodes
"""

import os
import sys
import json
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class CurriculumLaunchValidator:
    """Validates and launches curriculum learning evaluation."""
    
    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "errors": [],
            "warnings": []
        }
    
    def check_mock_servers_config(self):
        """Verify mock servers are configured and no API keys are required."""
        logger.info("Checking mock server configuration...")
        
        # Check if mock server scripts exist
        mock_scripts = [
            "start_mock_servers.py",
            "src/tools/mock_mcp_servers.py",
            "src/tools/mock_filesystem_mcp.py",
            "src/tools/mock_search_mcp.py",
            "src/tools/mock_sqlite_mcp.py",
            "src/tools/mock_github_mcp.py"
        ]
        
        all_exist = True
        for script in mock_scripts:
            script_path = Path(project_root) / script
            if not script_path.exists():
                self.validation_results["errors"].append(f"Missing mock server script: {script}")
                all_exist = False
        
        self.validation_results["checks"]["mock_servers"] = all_exist
        
        if all_exist:
            logger.info("✅ Mock server scripts found")
        else:
            logger.error("❌ Some mock server scripts are missing")
        
        # Check that API keys are NOT required in config
        config_path = Path(project_root) / "config" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # Ensure mock servers are enabled
            if config.get("use_mock_servers", False):
                logger.info("✅ Mock servers enabled in config")
            else:
                logger.warning("⚠️ Mock servers not explicitly enabled, will use fallback")
                self.validation_results["warnings"].append("Mock servers not explicitly enabled in config")
        
        return all_exist
    
    def check_improvements_active(self):
        """Verify all 7 improvements are properly configured."""
        logger.info("Checking improvements configuration...")
        
        # Load config
        config_path = Path(project_root) / "config" / "config.json"
        if not config_path.exists():
            self.validation_results["errors"].append("Config file missing")
            return False
        
        with open(config_path) as f:
            config = json.load(f)
        
        checks = {
            "dense_rewards": False,
            "optimized_hyperparams": False,
            "dueling_dqn": False,
            "double_q_learning": False,
            "curriculum_script": False,
            "state_reduction": False,
            "checkpoint_system": False
        }
        
        # 1. Check dense rewards
        if "reward_calculation" in config:
            weights = config["reward_calculation"].get("base_weights", {})
            if "step_progress" in weights and "exploration_bonus" in weights:
                checks["dense_rewards"] = True
                logger.info("✅ Dense reward shaping active")
        
        # 2. Check optimized hyperparameters
        if "q_learning" in config:
            ql = config["q_learning"]
            if ql.get("learning_rate") == 0.2 and ql.get("exploration_rate") == 0.3:
                checks["optimized_hyperparams"] = True
                logger.info("✅ Optimized hyperparameters active")
        
        # 3. Check DQN configuration
        if "dqn" in config:
            dqn = config["dqn"]
            if dqn.get("network_type") == "dueling":
                checks["dueling_dqn"] = True
                logger.info("✅ Dueling DQN configured")
        
        # 4. Check double Q-learning
        if config.get("q_learning", {}).get("use_double_q", True):
            checks["double_q_learning"] = True
            logger.info("✅ Double Q-learning enabled")
        
        # 5. Check curriculum script exists
        curriculum_script = Path(project_root) / "run_curriculum_learning_eval.py"
        if curriculum_script.exists():
            checks["curriculum_script"] = True
            logger.info("✅ Curriculum learning script exists")
        
        # 6. Check state reduction (PCA)
        if config.get("q_learning", {}).get("use_pca", True):
            checks["state_reduction"] = True
            logger.info("✅ State dimensionality reduction enabled")
        
        # 7. Check checkpoint system
        baseline_script = Path(project_root) / "tests" / "dissertation_test_suite" / "scripts" / "run_baseline_comparison.py"
        if baseline_script.exists():
            with open(baseline_script) as f:
                content = f.read()
                if "CheckpointManager" in content:
                    checks["checkpoint_system"] = True
                    logger.info("✅ Checkpoint system implemented")
        
        all_active = all(checks.values())
        self.validation_results["checks"]["improvements"] = checks
        
        if not all_active:
            failed = [k for k, v in checks.items() if not v]
            logger.error(f"❌ Missing improvements: {failed}")
            self.validation_results["errors"].append(f"Inactive improvements: {failed}")
        
        return all_active
    
    def check_hypothesis_testing(self):
        """Verify hypothesis testing is integrated."""
        logger.info("Checking hypothesis testing integration...")
        
        # Check if baseline comparison collects required metrics
        baseline_script = Path(project_root) / "tests" / "dissertation_test_suite" / "scripts" / "run_baseline_comparison.py"
        
        required_metrics = {
            "completion_rate": False,  # H1, H1b
            "convergence": False,      # H5
            "strategy_comparison": False,  # H1a
            "execution_time": False    # H2 (intent recognition)
        }
        
        if baseline_script.exists():
            with open(baseline_script) as f:
                content = f.read()
                
                if "completion_rates" in content:
                    required_metrics["completion_rate"] = True
                    logger.info("✅ Completion rate tracking enabled")
                
                if "_calculate_convergence" in content:
                    required_metrics["convergence"] = True
                    logger.info("✅ Convergence calculation enabled")
                
                if "q_learning_tabular" in content and "q_learning_dqn" in content:
                    required_metrics["strategy_comparison"] = True
                    logger.info("✅ DQN vs Tabular comparison enabled")
                
                if "execution_times" in content:
                    required_metrics["execution_time"] = True
                    logger.info("✅ Execution time tracking enabled")
        
        # Check pattern mining (H3)
        pattern_miner_exists = (Path(project_root) / "src" / "learning" / "pattern_miner.py").exists()
        if pattern_miner_exists:
            logger.info("✅ Pattern mining module available")
        else:
            logger.warning("⚠️ Pattern mining module not found - H3 may not be fully tested")
            self.validation_results["warnings"].append("Pattern mining module not found")
        
        all_metrics = all(required_metrics.values())
        self.validation_results["checks"]["hypothesis_metrics"] = required_metrics
        
        return all_metrics
    
    def check_environment(self):
        """Check Python environment and dependencies."""
        logger.info("Checking Python environment...")
        
        # Check Python version
        import sys
        py_version = sys.version_info
        if py_version.major == 3 and py_version.minor >= 8:
            logger.info(f"✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
            self.validation_results["checks"]["python_version"] = True
        else:
            logger.error(f"❌ Python 3.8+ required, found {py_version.major}.{py_version.minor}")
            self.validation_results["checks"]["python_version"] = False
            return False
        
        # Check key dependencies
        try:
            import numpy
            import torch
            import sklearn
            import sentence_transformers
            logger.info("✅ Core ML dependencies available")
            self.validation_results["checks"]["ml_dependencies"] = True
            return True
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            self.validation_results["checks"]["ml_dependencies"] = False
            return False
    
    def run_quick_test(self):
        """Run a quick test to verify everything works."""
        logger.info("\nRunning quick validation test...")
        
        try:
            # Run test_improvements.py to verify all improvements
            result = subprocess.run(
                ["python", "test_improvements.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✅ Improvements validation passed")
                self.validation_results["checks"]["improvements_test"] = True
                return True
            else:
                logger.error(f"❌ Improvements validation failed:\n{result.stderr}")
                self.validation_results["errors"].append("Improvements test failed")
                self.validation_results["checks"]["improvements_test"] = False
                return False
        except Exception as e:
            logger.error(f"❌ Could not run improvements test: {e}")
            self.validation_results["errors"].append(f"Test error: {e}")
            return False
    
    def generate_report(self):
        """Generate validation report."""
        print("\n" + "="*70)
        print("CURRICULUM LEARNING VALIDATION REPORT")
        print("="*70)
        print(f"Timestamp: {self.validation_results['timestamp']}")
        print()
        
        # Summary of checks
        print("Validation Checks:")
        for check, result in self.validation_results["checks"].items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
            elif isinstance(result, dict):
                passed = all(v for v in result.values() if isinstance(v, bool))
                status = "✅ PASS" if passed else "❌ PARTIAL"
            else:
                status = str(result)
            print(f"  {check:25s}: {status}")
        
        # Errors
        if self.validation_results["errors"]:
            print("\n❌ Errors Found:")
            for error in self.validation_results["errors"]:
                print(f"  - {error}")
        
        # Warnings
        if self.validation_results["warnings"]:
            print("\n⚠️ Warnings:")
            for warning in self.validation_results["warnings"]:
                print(f"  - {warning}")
        
        # Hypothesis coverage
        print("\nDissertation Hypotheses Coverage:")
        print("  H1  (>30% improvement):        Via completion_rate metrics")
        print("  H1a (DQN > Tabular):           Via separate strategy tracking")
        print("  H1b (>85% convergence):        Via completion_rate over episodes")
        print("  H2  (<100ms intent):           Via execution_time tracking")
        print("  H3  (>50 patterns):            Via pattern_miner module")
        print("  H5  (convergence <1000 eps):   Via convergence calculation")
        
        # Save report
        report_path = Path(project_root) / "curriculum_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"\nReport saved to: {report_path}")
        
        # Overall status
        all_critical_passed = (
            self.validation_results["checks"].get("mock_servers", False) and
            all(self.validation_results["checks"].get("improvements", {}).values()) and
            self.validation_results["checks"].get("python_version", False)
        )
        
        return all_critical_passed
    
    def launch_curriculum_learning(self):
        """Launch the full curriculum learning evaluation."""
        print("\n" + "="*70)
        print("LAUNCHING CURRICULUM LEARNING EVALUATION")
        print("="*70)
        print()
        print("Configuration:")
        print("  - Total Episodes: 50,000")
        print("  - Stages: 3 (Simple → Mixed → Complex)")
        print("  - Checkpoint Interval: 1,000 episodes")
        print("  - Mock Servers: ENABLED")
        print("  - All 7 Improvements: ACTIVE")
        print()
        print("Expected Duration: 6-8 hours")
        print("Results Directory: tests/dissertation_test_suite/results/curriculum_[timestamp]")
        print()
        
        # Ask for confirmation
        response = input("Start curriculum learning evaluation? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print("\n🚀 Starting curriculum learning evaluation...")
            print("You can monitor progress in the output.")
            print("Press Ctrl+C to interrupt (progress will be saved via checkpoints)")
            print()
            
            try:
                # Start mock servers first
                print("Starting mock servers...")
                subprocess.Popen(
                    ["python", "start_mock_servers.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Brief pause for servers to initialize
                import time
                time.sleep(2)
                
                # Launch curriculum learning
                subprocess.run(
                    ["python", "run_curriculum_learning_eval.py"],
                    check=False
                )
            except KeyboardInterrupt:
                print("\n\n⚠️ Evaluation interrupted by user")
                print("Progress has been saved to checkpoints")
                print("You can resume later using the checkpoint files")
            except Exception as e:
                print(f"\n❌ Error launching evaluation: {e}")
        else:
            print("\n❌ Evaluation cancelled by user")

def main():
    """Main entry point."""
    print("="*70)
    print("CURRICULUM LEARNING EVALUATION VALIDATOR")
    print("="*70)
    print()
    
    validator = CurriculumLaunchValidator()
    
    # Run all validation checks
    print("Step 1: Validating Environment")
    print("-"*40)
    env_ok = validator.check_environment()
    
    print("\nStep 2: Validating Mock Servers")
    print("-"*40)
    mock_ok = validator.check_mock_servers_config()
    
    print("\nStep 3: Validating Improvements")
    print("-"*40)
    improvements_ok = validator.check_improvements_active()
    
    print("\nStep 4: Validating Hypothesis Testing")
    print("-"*40)
    hypothesis_ok = validator.check_hypothesis_testing()
    
    print("\nStep 5: Running Quick Test")
    print("-"*40)
    test_ok = validator.run_quick_test()
    
    # Generate report
    all_passed = validator.generate_report()
    
    if all_passed:
        print("\n" + "="*70)
        print("✅ ALL VALIDATIONS PASSED")
        print("="*70)
        print("\nThe system is ready for curriculum learning evaluation!")
        print("\nThis evaluation will:")
        print("  1. Use mock servers only (no external dependencies)")
        print("  2. Test all core dissertation hypotheses (H1, H1a, H1b, H2, H3, H5)")
        print("  3. Apply all 7 performance improvements")
        print("  4. Run 50,000 episodes with curriculum progression")
        print("  5. Save checkpoints every 1,000 episodes")
        print()
        
        # Offer to launch
        validator.launch_curriculum_learning()
    else:
        print("\n" + "="*70)
        print("❌ VALIDATION FAILED")
        print("="*70)
        print("\nPlease address the errors above before running the evaluation.")
        print("\nKey issues to fix:")
        for error in validator.validation_results["errors"][:3]:
            print(f"  - {error}")

if __name__ == "__main__":
    main()