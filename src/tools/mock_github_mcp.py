"""
Mock GitHub MCP Server

This module provides a mock GitHub MCP server for testing and development
when the real GitHub API is not available or when testing without credentials.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
import random

class MockGitHubMCPServer:
    """Mock GitHub MCP server that simulates GitHub API operations.
    
    Since the official GitHub MCP server is archived, this mock implements
    a reasonable set of core GitHub operations commonly needed for development.
    """
    
    def __init__(self):
        self.name = "github-mock"
        self.description = "Mock GitHub operations"
        self.tools = [
            {
                "name": "list_repositories",
                "description": "List repositories for a user or organization",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string", "description": "GitHub username or organization"},
                        "type": {"type": "string", "enum": ["all", "owner", "member"], "default": "owner"},
                        "sort": {"type": "string", "enum": ["created", "updated", "pushed", "full_name"], "default": "created"}
                    }
                }
            },
            {
                "name": "search_repositories",
                "description": "Search for repositories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Search query"},
                        "sort": {"type": "string", "enum": ["stars", "forks", "updated"], "default": "stars"},
                        "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"}
                    },
                    "required": ["q"]
                }
            },
            {
                "name": "get_repository",
                "description": "Get repository details",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"}
                    },
                    "required": ["owner", "repo"]
                }
            },
            {
                "name": "create_issue",
                "description": "Create a new issue",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "title": {"type": "string", "description": "Issue title"},
                        "body": {"type": "string", "description": "Issue body"},
                        "labels": {"type": "array", "items": {"type": "string"}, "description": "Issue labels"},
                        "assignees": {"type": "array", "items": {"type": "string"}, "description": "Assignees"}
                    },
                    "required": ["owner", "repo", "title"]
                }
            },
            {
                "name": "list_issues",
                "description": "List issues for a repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                        "labels": {"type": "string", "description": "Comma-separated list of labels"}
                    },
                    "required": ["owner", "repo"]
                }
            },
            {
                "name": "create_pull_request",
                "description": "Create a pull request",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "title": {"type": "string", "description": "PR title"},
                        "head": {"type": "string", "description": "Source branch"},
                        "base": {"type": "string", "description": "Target branch"},
                        "body": {"type": "string", "description": "PR description"}
                    },
                    "required": ["owner", "repo", "title", "head", "base"]
                }
            },
            {
                "name": "list_pull_requests",
                "description": "List pull requests",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"}
                    },
                    "required": ["owner", "repo"]
                }
            },
            {
                "name": "search_code",
                "description": "Search for code across GitHub",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "Search query"},
                        "language": {"type": "string", "description": "Programming language filter"}
                    },
                    "required": ["q"]
                }
            },
            {
                "name": "create_or_update_file",
                "description": "Create or update a file in a repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "path": {"type": "string", "description": "File path in the repository"},
                        "content": {"type": "string", "description": "File content (base64 encoded)"},
                        "message": {"type": "string", "description": "Commit message"},
                        "branch": {"type": "string", "description": "Branch name", "default": "main"},
                        "sha": {"type": "string", "description": "SHA of the file being replaced (for updates)"}
                    },
                    "required": ["owner", "repo", "path", "content", "message"]
                }
            },
            {
                "name": "get_file_contents",
                "description": "Get the contents of a file from a repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "path": {"type": "string", "description": "File path in the repository"},
                        "ref": {"type": "string", "description": "Branch, tag, or commit to get file from"}
                    },
                    "required": ["owner", "repo", "path"]
                }
            },
            {
                "name": "push_files",
                "description": "Push multiple files to a repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "branch": {"type": "string", "description": "Branch name"},
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"}
                                },
                                "required": ["path", "content"]
                            },
                            "description": "Array of files to push"
                        },
                        "message": {"type": "string", "description": "Commit message"}
                    },
                    "required": ["owner", "repo", "branch", "files", "message"]
                }
            },
            {
                "name": "get_user",
                "description": "Get user information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string", "description": "GitHub username"}
                    },
                    "required": ["username"]
                }
            },
            {
                "name": "get_repository_content",
                "description": "Get repository content at a specific path",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "path": {"type": "string", "description": "Path in repository", "default": ""},
                        "ref": {"type": "string", "description": "Branch or commit ref"}
                    },
                    "required": ["owner", "repo"]
                }
            }
        ]
        
        # Mock data storage
        self.mock_repos = [
            {
                "id": 1,
                "name": "auto-tool-disc",
                "full_name": "user/auto-tool-disc",
                "owner": {"login": "user"},
                "description": "Autonomous Tool Discovery System",
                "private": False,
                "fork": False,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T00:00:00Z",
                "pushed_at": "2024-01-15T00:00:00Z",
                "stargazers_count": 42,
                "watchers_count": 42,
                "forks_count": 5,
                "language": "Python"
            },
            {
                "id": 2,
                "name": "mcp-test",
                "full_name": "user/mcp-test",
                "owner": {"login": "user"},
                "description": "MCP Testing Repository",
                "private": False,
                "fork": False,
                "created_at": "2024-01-10T00:00:00Z",
                "updated_at": "2024-01-12T00:00:00Z",
                "pushed_at": "2024-01-12T00:00:00Z",
                "stargazers_count": 10,
                "watchers_count": 10,
                "forks_count": 2,
                "language": "JavaScript"
            }
        ]
        
        self.mock_issues = []
        self.mock_pulls = []
        self.issue_counter = 1
        self.pr_counter = 1
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC requests."""
        method = request.get("method", "")
        request_id = request.get("id", 0)
        
        if method == "initialize":
            return self.handle_initialize(request_id)
        elif method == "tools/list":
            return self.handle_tools_list(request_id)
        elif method == "tools/call":
            return await self.handle_tool_call(request, request_id)
        else:
            return self.error_response(request_id, -32601, "Method not found")
    
    def handle_initialize(self, request_id: int) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "1.0",
                "serverInfo": {
                    "name": self.name,
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": True
                }
            }
        }
    
    def handle_tools_list(self, request_id: int) -> Dict[str, Any]:
        """List available tools."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self.tools
            }
        }
    
    async def handle_tool_call(self, request: Dict[str, Any], request_id: int) -> Dict[str, Any]:
        """Execute GitHub operations."""
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        # Map old tool names to new ones for backward compatibility
        tool_name_mapping = {
            "list_repos": "list_repositories",
            "search_repos": "search_repositories",
            "get_repo": "get_repository",
            "create_pull": "create_pull_request",
            "list_pulls": "list_pull_requests"
        }
        
        # Apply mapping if old name is used
        if tool_name in tool_name_mapping:
            tool_name = tool_name_mapping[tool_name]
        
        try:
            if tool_name == "list_repositories":
                result = self._list_repos(arguments.get("username"))
            
            elif tool_name == "search_repositories":
                result = self._search_repos(arguments["q"])
            
            elif tool_name == "get_repository":
                result = self._get_repo(arguments["owner"], arguments["repo"])
            
            elif tool_name == "create_issue":
                result = self._create_issue(
                    arguments["owner"],
                    arguments["repo"],
                    arguments["title"],
                    arguments.get("body", "")
                )
            
            elif tool_name == "list_issues":
                result = self._list_issues(
                    arguments["owner"],
                    arguments["repo"],
                    arguments.get("state", "open")
                )
            
            elif tool_name == "create_pull_request":
                result = self._create_pull(
                    arguments["owner"],
                    arguments["repo"],
                    arguments["title"],
                    arguments["head"],
                    arguments["base"],
                    arguments.get("body", "")
                )
            
            elif tool_name == "list_pull_requests":
                result = self._list_pulls(
                    arguments["owner"],
                    arguments["repo"],
                    arguments.get("state", "open")
                )
            
            elif tool_name == "search_code":
                result = self._search_code(arguments["q"], arguments.get("language"))
            
            elif tool_name == "create_or_update_file":
                result = self._create_or_update_file(
                    arguments["owner"],
                    arguments["repo"],
                    arguments["path"],
                    arguments["content"],
                    arguments["message"],
                    arguments.get("branch", "main"),
                    arguments.get("sha")
                )
            
            elif tool_name == "get_file_contents":
                result = self._get_file_contents(
                    arguments["owner"],
                    arguments["repo"],
                    arguments["path"],
                    arguments.get("ref")
                )
            
            elif tool_name == "push_files":
                result = self._push_files(
                    arguments["owner"],
                    arguments["repo"],
                    arguments["branch"],
                    arguments["files"],
                    arguments["message"]
                )
            
            elif tool_name == "get_user":
                result = self._get_user(arguments["username"])
            
            elif tool_name == "get_repository_content":
                result = self._get_repository_content(
                    arguments["owner"],
                    arguments["repo"],
                    arguments.get("path", ""),
                    arguments.get("ref")
                )
            
            else:
                # Check if it's an unimplemented real server tool
                unimplemented_tools = [
                    "create_branch", "merge_pull_request", "create_release",
                    "list_commits", "get_commit", "list_branches", "delete_branch",
                    "list_releases", "get_release", "update_issue", "close_issue",
                    "add_labels", "remove_labels", "create_comment", "update_comment"
                ]
                
                if tool_name in unimplemented_tools:
                    return self.error_response(
                        request_id, 
                        -32602, 
                        f"Tool '{tool_name}' is not implemented in mock server. Available in real GitHub MCP server only."
                    )
                
                return self.error_response(request_id, -32602, f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self.error_response(request_id, -32603, str(e))
    
    def error_response(self, request_id: int, code: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    # Mock implementation methods
    
    def _list_repos(self, username: str = None) -> List[Dict]:
        """Mock list repositories."""
        return self.mock_repos
    
    def _search_repos(self, query: str) -> Dict[str, Any]:
        """Mock search repositories."""
        # Simple mock search - return repos that contain query in name or description
        results = []
        for repo in self.mock_repos:
            if (query.lower() in repo["name"].lower() or 
                query.lower() in repo["description"].lower()):
                results.append(repo)
        
        return {
            "total_count": len(results),
            "incomplete_results": False,
            "items": results
        }
    
    def _get_repo(self, owner: str, repo: str) -> Dict[str, Any]:
        """Mock get repository details."""
        for r in self.mock_repos:
            if r["owner"]["login"] == owner and r["name"] == repo:
                return r
        
        raise Exception(f"Repository {owner}/{repo} not found")
    
    def _create_issue(self, owner: str, repo: str, title: str, body: str) -> Dict[str, Any]:
        """Mock create issue."""
        issue = {
            "id": len(self.mock_issues) + 1,
            "number": self.issue_counter,
            "title": title,
            "body": body,
            "state": "open",
            "user": {"login": "mock-user"},
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "repository": {"owner": {"login": owner}, "name": repo}
        }
        
        self.mock_issues.append(issue)
        self.issue_counter += 1
        
        return issue
    
    def _list_issues(self, owner: str, repo: str, state: str) -> List[Dict]:
        """Mock list issues."""
        results = []
        for issue in self.mock_issues:
            if (issue["repository"]["owner"]["login"] == owner and
                issue["repository"]["name"] == repo):
                if state == "all" or issue["state"] == state:
                    results.append(issue)
        
        return results
    
    def _create_pull(self, owner: str, repo: str, title: str, 
                    head: str, base: str, body: str) -> Dict[str, Any]:
        """Mock create pull request."""
        pr = {
            "id": len(self.mock_pulls) + 1,
            "number": self.pr_counter,
            "title": title,
            "body": body,
            "state": "open",
            "head": {"ref": head},
            "base": {"ref": base},
            "user": {"login": "mock-user"},
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "repository": {"owner": {"login": owner}, "name": repo},
            "mergeable": True,
            "merged": False
        }
        
        self.mock_pulls.append(pr)
        self.pr_counter += 1
        
        return pr
    
    def _list_pulls(self, owner: str, repo: str, state: str) -> List[Dict]:
        """Mock list pull requests."""
        results = []
        for pr in self.mock_pulls:
            if (pr["repository"]["owner"]["login"] == owner and
                pr["repository"]["name"] == repo):
                if state == "all" or pr["state"] == state:
                    results.append(pr)
        
        return results
    
    def _search_code(self, query: str, language: str = None) -> Dict[str, Any]:
        """Mock search code."""
        # Simple mock implementation
        mock_results = [
            {
                "name": "example.py",
                "path": "src/example.py",
                "repository": {
                    "name": "auto-tool-disc",
                    "full_name": "user/auto-tool-disc",
                    "owner": {"login": "user"}
                },
                "text_matches": [
                    {
                        "fragment": f"# Code containing {query}",
                        "matches": [{"text": query, "indices": [17, 17 + len(query)]}]
                    }
                ]
            }
        ]
        
        if language:
            mock_results = [r for r in mock_results if r["name"].endswith(f".{language}")]
        
        return {
            "total_count": len(mock_results),
            "incomplete_results": False,
            "items": mock_results
        }
    
    def _create_or_update_file(self, owner: str, repo: str, path: str, 
                              content: str, message: str, branch: str = "main", 
                              sha: str = None) -> Dict[str, Any]:
        """Mock create or update file."""
        import base64
        
        # Mock response
        return {
            "content": {
                "name": path.split("/")[-1],
                "path": path,
                "sha": sha or f"mock-sha-{random.randint(1000, 9999)}",
                "size": len(content),
                "url": f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                "type": "file",
                "encoding": "base64"
            },
            "commit": {
                "sha": f"mock-commit-{random.randint(1000, 9999)}",
                "message": message,
                "url": f"https://api.github.com/repos/{owner}/{repo}/commits/mock-sha",
                "author": {
                    "name": "Mock User",
                    "email": "mock@example.com",
                    "date": datetime.utcnow().isoformat() + "Z"
                }
            }
        }
    
    def _get_file_contents(self, owner: str, repo: str, path: str, ref: str = None) -> Dict[str, Any]:
        """Mock get file contents."""
        import base64
        
        # Mock file content
        mock_content = f"# Mock file content for {path}\n\nThis is mock data."
        encoded_content = base64.b64encode(mock_content.encode()).decode()
        
        return {
            "type": "file",
            "encoding": "base64",
            "size": len(mock_content),
            "name": path.split("/")[-1],
            "path": path,
            "content": encoded_content,
            "sha": f"mock-sha-{random.randint(1000, 9999)}",
            "url": f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            "git_url": f"https://api.github.com/repos/{owner}/{repo}/git/blobs/mock-sha",
            "download_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{ref or 'main'}/{path}"
        }
    
    def _push_files(self, owner: str, repo: str, branch: str, 
                   files: List[Dict[str, str]], message: str) -> Dict[str, Any]:
        """Mock push multiple files."""
        # Mock response
        commit_sha = f"mock-commit-{random.randint(1000, 9999)}"
        
        return {
            "commit": {
                "sha": commit_sha,
                "message": message,
                "url": f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}",
                "author": {
                    "name": "Mock User",
                    "email": "mock@example.com",
                    "date": datetime.utcnow().isoformat() + "Z"
                }
            },
            "files_changed": len(files),
            "files": [
                {
                    "path": f["path"],
                    "status": "added",
                    "sha": f"mock-sha-{random.randint(1000, 9999)}"
                }
                for f in files
            ]
        }
    
    def _get_user(self, username: str) -> Dict[str, Any]:
        """Mock get user information."""
        return {
            "login": username,
            "id": random.randint(1000000, 9999999),
            "avatar_url": f"https://avatars.githubusercontent.com/u/{random.randint(1000, 9999)}",
            "type": "User",
            "name": f"Mock {username}",
            "company": "Mock Company",
            "blog": f"https://{username}.example.com",
            "location": "Mock City",
            "email": f"{username}@example.com",
            "bio": f"Mock bio for {username}",
            "public_repos": random.randint(10, 100),
            "public_gists": random.randint(0, 50),
            "followers": random.randint(0, 1000),
            "following": random.randint(0, 500),
            "created_at": "2020-01-01T00:00:00Z",
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
    
    def _get_repository_content(self, owner: str, repo: str, path: str = "", ref: str = None) -> List[Dict[str, Any]]:
        """Mock get repository content."""
        if not path:
            # Return root directory listing
            return [
                {
                    "type": "file",
                    "name": "README.md",
                    "path": "README.md",
                    "sha": f"mock-sha-{random.randint(1000, 9999)}",
                    "size": 1234,
                    "url": f"https://api.github.com/repos/{owner}/{repo}/contents/README.md"
                },
                {
                    "type": "dir",
                    "name": "src",
                    "path": "src",
                    "sha": f"mock-sha-{random.randint(1000, 9999)}",
                    "url": f"https://api.github.com/repos/{owner}/{repo}/contents/src"
                },
                {
                    "type": "file",
                    "name": ".gitignore",
                    "path": ".gitignore",
                    "sha": f"mock-sha-{random.randint(1000, 9999)}",
                    "size": 150,
                    "url": f"https://api.github.com/repos/{owner}/{repo}/contents/.gitignore"
                }
            ]
        else:
            # Return specific path content
            if path == "src":
                return [
                    {
                        "type": "file",
                        "name": "main.py",
                        "path": "src/main.py",
                        "sha": f"mock-sha-{random.randint(1000, 9999)}",
                        "size": 2500,
                        "url": f"https://api.github.com/repos/{owner}/{repo}/contents/src/main.py"
                    },
                    {
                        "type": "file",
                        "name": "utils.py",
                        "path": "src/utils.py",
                        "sha": f"mock-sha-{random.randint(1000, 9999)}",
                        "size": 1800,
                        "url": f"https://api.github.com/repos/{owner}/{repo}/contents/src/utils.py"
                    }
                ]
            else:
                # Single file
                return self._get_file_contents(owner, repo, path, ref)