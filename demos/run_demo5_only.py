#!/usr/bin/env python3
"""Run only Demo 5 of the A/B testing framework."""
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from demos.demo_ab_testing_framework import ABTestingDemo

async def main():
    demo = ABTestingDemo()
    await demo.demo_strategy_comparison()

if __name__ == "__main__":
    asyncio.run(main())