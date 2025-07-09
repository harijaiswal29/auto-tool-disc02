"""
Logging configuration for the Autonomous Tool Discovery system.
This module sets up comprehensive logging with both file and console outputs.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
import json
from typing import Optional
import sys
import platform

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

class ProjectLogger:
    """
    Centralized logging system for the dissertation project.
    
    Real-world analogy: Like a flight recorder that tracks everything
    happening in your system - crucial for debugging and analysis.
    """
    
    def __init__(self, name: str, log_dir: str = "data/logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Fix Windows console encoding issues
        if platform.system() == 'Windows':
            # Enable Windows console to handle Unicode
            if sys.stdout.encoding != 'utf-8':
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        
        # Create a unique log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{name}_{timestamp}.log"
        self.json_log_file = self.log_dir / f"{name}_{timestamp}.json"
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure the logger with multiple handlers."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for detailed logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # JSON handler for structured logs (useful for analysis)
        json_handler = logging.FileHandler(self.json_log_file)
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JsonFormatter())
        
        # Add all handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
    
    def get_logger(self):
        """Return the configured logger instance."""
        return self.logger
    
    def log_experiment(self, experiment_name: str, params: dict, results: dict):
        """
        Special method for logging experiment results.
        
        Args:
            experiment_name: Name of the experiment
            params: Dictionary of experiment parameters
            results: Dictionary of experiment results
        """
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Parameters: {json.dumps(params, indent=2)}")
        self.logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        # Also save to a separate experiments log
        exp_log = self.log_dir / "experiments.json"
        experiment_data = {
            "timestamp": datetime.now().isoformat(),
            "name": experiment_name,
            "parameters": params,
            "results": results
        }
        
        # Append to experiments file
        experiments = []
        if exp_log.exists():
            with open(exp_log, 'r') as f:
                experiments = json.load(f)
        
        experiments.append(experiment_data)
        
        with open(exp_log, 'w') as f:
            json.dump(experiments, f, indent=2)

class JsonFormatter(logging.Formatter):
    """Format logs as JSON for easy parsing and analysis."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# Convenience function to get a logger
def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Optional custom log directory
    
    Returns:
        Configured logger instance
    """
    if log_dir:
        project_logger = ProjectLogger(name, log_dir)
    else:
        project_logger = ProjectLogger(name)
    
    return project_logger.get_logger()

# Example usage for dissertation milestones
def log_milestone(logger: logging.Logger, milestone: str, details: dict):
    """Log important dissertation milestones."""
    logger.info(f"🎯 MILESTONE: {milestone}")
    for key, value in details.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    # Test the logger
    logger = get_logger("test_logger")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test experiment logging
    project_logger = ProjectLogger("experiments")
    project_logger.log_experiment(
        "initial_test",
        {"learning_rate": 0.1, "epsilon": 0.2},
        {"accuracy": 0.85, "iterations": 100}
    )
    
    # Test milestone logging
    log_milestone(logger, "Logger Setup Complete", {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "component": "Logging Framework",
        "status": "✅ Success"
    })