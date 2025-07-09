#!/usr/bin/env python3
"""
Quick test to check if Unicode is working after our fixes.
"""

import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()}")
print(f"Default encoding: {sys.stdout.encoding}")

# Test without emojis (should always work)
print("\n--- Text-only test ---")
print("[PASS] This should work")
print("[FAIL] This should also work")
print("[INFO] No emojis here")

# Import our logger with the fix
try:
    from src.utils.logger import get_logger
    logger = get_logger("encoding_test")
    
    print("\n--- Logger test (with potential Unicode fix) ---")
    logger.info("Testing logger without emojis")
    logger.info("If you see this, the logger works!")
    
    # Try with emojis (might work with our fix)
    print("\n--- Emoji test (might work now) ---")
    try:
        print("✅ Check mark")
        print("❌ Cross mark")
        print("🚀 Rocket")
        logger.info("✅ Logger with emoji")
    except UnicodeEncodeError:
        print("[INFO] Emojis still not supported, but that's OK!")
        
except Exception as e:
    print(f"\n[ERROR] Could not import logger: {e}")
    print("[INFO] Make sure you're in the project root directory")

print("\n[DONE] Test complete!")