#!/usr/bin/env python3
"""
Setup Verification Script - Windows Compatible Version
No emoji characters to avoid encoding issues on Windows.
"""

import sys
import os
import json
import sqlite3
import importlib
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger("setup_verification")

class SetupVerifier:
    """Verify all components are properly set up."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
    
    def check(self, description: str, check_func):
        """Run a single check and log results."""
        try:
            result = check_func()
            if result:
                logger.info(f"[PASS] {description}")
                self.checks_passed += 1
            else:
                logger.error(f"[FAIL] {description}")
                self.checks_failed += 1
            return result
        except Exception as e:
            logger.error(f"[FAIL] {description}: {str(e)}")
            self.checks_failed += 1
            return False
    
    def verify_directory_structure(self):
        """Check if all required directories exist."""
        required_dirs = [
            "src/core", "src/agents", "src/tools", "src/learning", "src/utils",
            "data/logs", "data/registry", "data/metrics",
            "experiments", "tests", "config", "docs"
        ]
        
        logger.info("\n=== CHECKING DIRECTORY STRUCTURE ===")
        logger.info("-" * 40)
        
        for dir_path in required_dirs:
            self.check(
                f"Directory exists: {dir_path}",
                lambda p=dir_path: Path(p).exists()
            )
    
    def verify_python_environment(self):
        """Check Python version and required packages."""
        logger.info("\n=== CHECKING PYTHON ENVIRONMENT ===")
        logger.info("-" * 40)
        
        # Check Python version
        python_version = sys.version_info
        self.check(
            f"Python version >= 3.8 (current: {python_version.major}.{python_version.minor})",
            lambda: python_version >= (3, 8)
        )
        
        # Check required packages
        required_packages = {
            "numpy": "numpy",
            "pandas": "pandas",
            "sklearn": "scikit-learn",
            "sentence_transformers": "sentence-transformers",
            "networkx": "networkx",
            "asyncio": "asyncio (built-in)"
        }
        
        for import_name, package_name in required_packages.items():
            self.check(
                f"Package installed: {package_name}",
                lambda n=import_name: self._can_import(n)
            )
    
    def _can_import(self, module_name):
        """Check if a module can be imported."""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    def verify_configuration(self):
        """Check if configuration files are valid."""
        logger.info("\n=== CHECKING CONFIGURATION ===")
        logger.info("-" * 40)
        
        config_path = Path("config/config.json")
        
        def check_config():
            if not config_path.exists():
                logger.warning("Config file not found, creating default...")
                config_path.parent.mkdir(exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump({"project": {"name": "Auto Tool Discovery"}}, f, indent=2)
                return True
            
            with open(config_path, 'r') as f:
                config = json.load(f)
                return "project" in config
        
        self.check("Configuration file is valid", check_config)
    
    def verify_logging(self):
        """Test logging functionality."""
        logger.info("\n=== TESTING LOGGING SYSTEM ===")
        logger.info("-" * 40)
        
        def test_logging():
            test_logger = get_logger("test_module")
            test_logger.debug("Debug test")
            test_logger.info("Info test")
            test_logger.warning("Warning test")
            test_logger.error("Error test")
            
            # Check if log files were created
            log_dir = Path("data/logs")
            log_files = list(log_dir.glob("test_module_*.log"))
            return len(log_files) > 0
        
        self.check("Logging system works", test_logging)
    
    def verify_mcp_prerequisites(self):
        """Check if MCP prerequisites are met."""
        logger.info("\n=== CHECKING MCP PREREQUISITES ===")
        logger.info("-" * 40)
        
        # Check if npm is installed
        def check_npm():
            try:
                result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
                return result.returncode == 0
            except FileNotFoundError:
                return False
        
        npm_installed = self.check("npm is installed", check_npm)
        
        if not npm_installed:
            self.warnings.append("Install Node.js and npm to use MCP servers")
            self.warnings.append("Download from: https://nodejs.org/")
        
        # Check if any MCP servers are installed
        mcp_servers = [
            "@modelcontextprotocol/server-filesystem",
            "@modelcontextprotocol/server-sqlite",
            "@modelcontextprotocol/server-time"
        ]
        
        for server in mcp_servers:
            def check_server(s=server):
                try:
                    result = subprocess.run(
                        ["npm", "list", "-g", s], 
                        capture_output=True, 
                        text=True
                    )
                    return s in result.stdout
                except:
                    return False
            
            if not self.check(f"MCP server installed: {server}", check_server):
                self.warnings.append(f"Install with: npm install -g {server}")
    
    def verify_database_setup(self):
        """Check database configuration."""
        logger.info("\n=== CHECKING DATABASE SETUP ===")
        logger.info("-" * 40)
        
        def test_sqlite():
            test_db = Path("data/registry/test_verify.db")
            test_db.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
            conn.commit()
            conn.close()
            
            # Clean up
            test_db.unlink()
            return True
        
        self.check("SQLite database creation works", test_sqlite)
    
    def run_all_checks(self):
        """Run all verification checks."""
        logger.info("STARTING DISSERTATION PROJECT SETUP VERIFICATION")
        logger.info("=" * 50)
        
        self.verify_directory_structure()
        self.verify_python_environment()
        self.verify_configuration()
        self.verify_logging()
        self.verify_mcp_prerequisites()
        self.verify_database_setup()
        
        # Summary
        logger.info("\n=== VERIFICATION SUMMARY ===")
        logger.info("=" * 50)
        logger.info(f"[PASSED] Checks passed: {self.checks_passed}")
        logger.info(f"[FAILED] Checks failed: {self.checks_failed}")
        
        if self.warnings:
            logger.warning("\n=== WARNINGS ===")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if self.checks_failed == 0:
            logger.info("\n*** ALL CHECKS PASSED! Your setup is ready. ***")
            logger.info("Next step: Run 'python src/hello_mcp.py' to test MCP concepts")
        else:
            logger.error("\n*** SOME CHECKS FAILED. Please fix the issues above. ***")
            logger.info("\nMost issues can be fixed by:")
            logger.info("  1. Installing missing packages: pip install -r requirements.txt")
            logger.info("  2. Installing Node.js from https://nodejs.org/")
            logger.info("  3. Installing MCP servers: npm install -g @modelcontextprotocol/server-[name]")
        
        return self.checks_failed == 0

def main():
    """Run the setup verification."""
    verifier = SetupVerifier()
    success = verifier.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()