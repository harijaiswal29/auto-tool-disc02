#!/usr/bin/env python3
"""
Standalone Filesystem MCP Test

This test can run without all dependencies installed.
It tests the filesystem MCP functionality directly.
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


async def test_mock_filesystem_server():
    """Test the mock filesystem MCP server directly."""
    print("\n" + "="*50)
    print("Testing Mock Filesystem MCP Server")
    print("="*50 + "\n")
    
    # Import only what we need
    from src.tools.mock_filesystem_mcp import MockFileSystemMCPServer
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="fs_test_")
    print(f"Created test directory: {test_dir}")
    
    try:
        # Initialize mock server
        server = MockFileSystemMCPServer(base_path=test_dir)
        
        # Test 1: Write file
        print("\n1. Testing write_file...")
        write_result = await server.write_file(
            {"path": "test.txt", "content": "Hello from standalone test!"},
            request_id=1
        )
        print(f"   Result: {write_result}")
        assert write_result["result"]["success"], "Write failed"
        
        # Test 2: Read file
        print("\n2. Testing read_file...")
        read_result = await server.read_file({"path": "test.txt"}, request_id=2)
        print(f"   Content: {read_result['result']['content']}")
        assert read_result["result"]["content"] == "Hello from standalone test!", "Content mismatch"
        
        # Test 3: List directory
        print("\n3. Testing list_directory...")
        list_result = await server.list_directory({"path": "."}, request_id=3)
        print(f"   Items: {[f['name'] for f in list_result['result']['items']]}")
        assert any(f["name"] == "test.txt" for f in list_result["result"]["items"]), "File not found"
        
        # Test 4: Create directory
        print("\n4. Testing create_directory...")
        mkdir_result = await server.create_directory({"path": "subdir/nested"}, request_id=4)
        print(f"   Result: {mkdir_result}")
        assert mkdir_result["result"]["created"], "Directory creation failed"
        
        # Test 5: Write file in subdirectory
        print("\n5. Testing write in subdirectory...")
        sub_write = await server.write_file(
            {"path": "subdir/nested/data.json", "content": '{"test": true}'},
            request_id=5
        )
        assert sub_write["result"]["success"], "Subdirectory write failed"
        
        # Test 6: Error handling - read non-existent
        print("\n6. Testing error handling...")
        error_result = await server.read_file({"path": "nonexistent.txt"}, request_id=6)
        if "error" in error_result:
            print(f"   Correctly got error: {error_result['error']['message']}")
        else:
            assert False, "Should have returned an error"
        
        # Test 7: Security - path traversal
        print("\n7. Testing security (path traversal)...")
        security_result = await server.read_file({"path": "../../../etc/passwd"}, request_id=7)
        if "error" in security_result:
            print(f"   Security check passed: {security_result['error']['message']}")
        else:
            assert False, "Should have blocked path traversal"
        
        print("\n" + "="*50)
        print("✅ All tests passed!")
        print("="*50)
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")


async def test_filesystem_client():
    """Test the filesystem MCP client."""
    print("\n" + "="*50)
    print("Testing Filesystem MCP Client")
    print("="*50 + "\n")
    
    # Import only what we need
    from src.tools.filesystem_mcp import FileSystemMCPClient
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="fs_client_test_")
    print(f"Created test directory: {test_dir}")
    
    try:
        # Initialize client
        client = FileSystemMCPClient(base_path=test_dir)
        
        # Connect (will use mock)
        print("Connecting to filesystem MCP...")
        await client.connect(use_mock=True)
        print("Connected successfully (using mock)")
        
        # Test 1: Write file
        print("\n1. Testing write_file...")
        result = await client.write_file("client_test.txt", "Testing from client")
        print(f"   Success: {result.get('success', False)}")
        
        # Test 2: File exists
        print("\n2. Testing file_exists...")
        exists_result = await client.file_exists("client_test.txt")
        print(f"   File exists result: {exists_result}")
        # Check if it's a dict with exists key or just a boolean
        if isinstance(exists_result, dict):
            exists = exists_result.get("exists", False)
        else:
            exists = exists_result
        assert exists, "File should exist"
        
        # Test 3: Read file
        print("\n3. Testing read_file...")
        read_result = await client.read_file("client_test.txt")
        print(f"   Read result: {read_result}")
        # Extract content from the result
        if isinstance(read_result, dict):
            if "result" in read_result:
                content = read_result["result"]["content"]
            elif "content" in read_result:
                content = read_result["content"]
            else:
                content = str(read_result)
        else:
            content = read_result
        print(f"   Extracted content: {content}")
        assert content == "Testing from client", f"Content mismatch: expected 'Testing from client', got '{content}'"
        
        # Test 4: List directory
        print("\n4. Testing list_directory...")
        list_result = await client.list_directory(".")
        # Extract files from result
        if isinstance(list_result, dict):
            if "result" in list_result and "items" in list_result["result"]:
                files = list_result["result"]["items"]
            elif "items" in list_result:
                files = list_result["items"]
            elif "files" in list_result:
                files = list_result["files"]
            else:
                files = []
        else:
            files = list_result if isinstance(list_result, list) else []
        print(f"   Found {len(files)} items")
        assert any(f["name"] == "client_test.txt" for f in files), "File not in listing"
        
        # Test 5: Create directory
        print("\n5. Testing create_directory...")
        result = await client.create_directory("test_dir/sub")
        print(f"   Created: {result.get('created', False)}")
        
        # Disconnect
        await client.disconnect()
        print("\nDisconnected from filesystem MCP")
        
        print("\n" + "="*50)
        print("✅ Client tests passed!")
        print("="*50)
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")


async def main():
    """Run standalone tests."""
    print("Filesystem MCP Standalone Tests")
    print("This test runs without full dependencies")
    print("="*50)
    
    try:
        # Test mock server directly
        await test_mock_filesystem_server()
        
        # Test client
        await test_filesystem_client()
        
        print("\n" + "="*50)
        print("✅ ALL STANDALONE TESTS PASSED!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)