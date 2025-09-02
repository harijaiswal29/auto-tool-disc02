#!/usr/bin/env python3
"""
Synchronous wrapper for the optimized baseline comparison script.

This wrapper allows the async optimized baseline comparison to be called
from subprocess in the curriculum learning script.
"""

import os
# DISABLE CUDA to prevent errors in WSL2
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import asyncio
import logging

# Suppress verbose logging by default
logging.basicConfig(level=logging.WARNING)
for logger_name in ['src', 'learning', 'evaluation', 'PatternMiner']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Import the async main function
from run_baseline_comparison_optimized import main

if __name__ == "__main__":
    # Run the async main function synchronously
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)