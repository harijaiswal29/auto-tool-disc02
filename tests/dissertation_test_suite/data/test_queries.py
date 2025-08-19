"""
Comprehensive test query sets for dissertation evaluation.

This module provides diverse query sets with ground truth for evaluating
the autonomous tool discovery and integration system against baselines.
Each query includes optimal tool selections for accuracy measurement.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Single intent, single tool
    COMPLEX = "complex"    # Multi-intent, multi-tool
    AMBIGUOUS = "ambiguous"  # Requires disambiguation


@dataclass
class TestQuery:
    """Test query with ground truth."""
    query: str
    optimal_tools: List[str]
    complexity: QueryComplexity
    intents: List[str]
    domain: str
    expected_success_rate: float = 0.9  # Expected success for optimal system


# Simple queries - single intent, single tool
SIMPLE_QUERIES = [
    TestQuery(
        query="List all Python files in the current directory",
        optimal_tools=["filesystem_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["file.list"],
        domain="file_operations"
    ),
    TestQuery(
        query="Get the current weather in San Francisco",
        optimal_tools=["weather_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["weather.get"],
        domain="information_retrieval"
    ),
    TestQuery(
        query="Search for recent AI research papers",
        optimal_tools=["search_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["search.web"],
        domain="web_search"
    ),
    TestQuery(
        query="Query the database for all active users",
        optimal_tools=["sqlite_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["database.query"],
        domain="data_analysis"
    ),
    TestQuery(
        query="Show Git commit history",
        optimal_tools=["github_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["git.history"],
        domain="development"
    ),
    TestQuery(
        query="Find all TODO comments in code",
        optimal_tools=["filesystem_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["code.search"],
        domain="development"
    ),
    TestQuery(
        query="Get system resource usage",
        optimal_tools=["system_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["system.monitor"],
        domain="system_monitoring"
    ),
    TestQuery(
        query="List recent file modifications",
        optimal_tools=["filesystem_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["file.recent"],
        domain="file_operations"
    ),
    TestQuery(
        query="Check PostgreSQL connection status",
        optimal_tools=["postgres_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["database.status"],
        domain="data_analysis"
    ),
    TestQuery(
        query="Search for Python documentation",
        optimal_tools=["search_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["search.documentation"],
        domain="web_search"
    ),
    TestQuery(
        query="Get current stock price for AAPL",
        optimal_tools=["zerodha_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["finance.stock_price"],
        domain="finance"
    ),
    TestQuery(
        query="Create a new note in Notion",
        optimal_tools=["notion_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["note.create"],
        domain="productivity"
    ),
    TestQuery(
        query="List all tables in the database",
        optimal_tools=["sqlite_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["database.schema"],
        domain="data_analysis"
    ),
    TestQuery(
        query="Find files containing 'error'",
        optimal_tools=["filesystem_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["file.search"],
        domain="file_operations"
    ),
    TestQuery(
        query="Get repository information",
        optimal_tools=["github_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["git.info"],
        domain="development"
    ),
    TestQuery(
        query="Check weather forecast for tomorrow",
        optimal_tools=["weather_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["weather.forecast"],
        domain="information_retrieval"
    ),
    TestQuery(
        query="Search for machine learning tutorials",
        optimal_tools=["search_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["search.tutorial"],
        domain="web_search"
    ),
    TestQuery(
        query="Execute SQL query on users table",
        optimal_tools=["sqlite_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["database.execute"],
        domain="data_analysis"
    ),
    TestQuery(
        query="Find largest files in directory",
        optimal_tools=["filesystem_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["file.analyze"],
        domain="file_operations"
    ),
    TestQuery(
        query="Get GitHub issue details",
        optimal_tools=["github_mcp"],
        complexity=QueryComplexity.SIMPLE,
        intents=["git.issue"],
        domain="development"
    )
]


# Complex queries - multiple intents, multiple tools
COMPLEX_QUERIES = [
    TestQuery(
        query="Find all Python files modified today and analyze their code complexity",
        optimal_tools=["filesystem_mcp", "github_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["file.find", "code.analyze"],
        domain="development"
    ),
    TestQuery(
        query="Query sales data from database and search for market trends online",
        optimal_tools=["sqlite_mcp", "search_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["database.query", "search.trends"],
        domain="business_analysis"
    ),
    TestQuery(
        query="Get weather data and save it to the database",
        optimal_tools=["weather_mcp", "sqlite_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["weather.get", "database.insert"],
        domain="data_collection"
    ),
    TestQuery(
        query="Search for competitor information and store findings in Notion",
        optimal_tools=["search_mcp", "notion_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["search.business", "note.create"],
        domain="business_research"
    ),
    TestQuery(
        query="Analyze code repository statistics and create a report",
        optimal_tools=["github_mcp", "filesystem_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["git.analyze", "file.create"],
        domain="development"
    ),
    TestQuery(
        query="Compare database schemas between SQLite and PostgreSQL",
        optimal_tools=["sqlite_mcp", "postgres_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["database.schema", "database.compare"],
        domain="data_analysis"
    ),
    TestQuery(
        query="Find error logs and search for solutions online",
        optimal_tools=["filesystem_mcp", "search_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["file.search", "search.solution"],
        domain="troubleshooting"
    ),
    TestQuery(
        query="Get stock prices and store historical data in database",
        optimal_tools=["zerodha_mcp", "sqlite_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["finance.prices", "database.store"],
        domain="finance"
    ),
    TestQuery(
        query="Monitor system resources and log anomalies to file",
        optimal_tools=["system_mcp", "filesystem_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["system.monitor", "file.write"],
        domain="system_monitoring"
    ),
    TestQuery(
        query="Search for API documentation and create implementation notes",
        optimal_tools=["search_mcp", "notion_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["search.api", "note.create"],
        domain="development"
    ),
    TestQuery(
        query="Analyze Git commit patterns and identify frequent contributors",
        optimal_tools=["github_mcp", "sqlite_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["git.analyze", "data.aggregate"],
        domain="development"
    ),
    TestQuery(
        query="Find configuration files and validate against documentation",
        optimal_tools=["filesystem_mcp", "search_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["file.find", "search.validate"],
        domain="configuration"
    ),
    TestQuery(
        query="Query customer data and search for industry benchmarks",
        optimal_tools=["postgres_mcp", "search_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["database.query", "search.benchmark"],
        domain="business_analysis"
    ),
    TestQuery(
        query="Scan code for security issues and find remediation guides",
        optimal_tools=["github_mcp", "search_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["code.security", "search.guide"],
        domain="security"
    ),
    TestQuery(
        query="Extract data from multiple databases and create comparison report",
        optimal_tools=["sqlite_mcp", "postgres_mcp", "filesystem_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["database.extract", "report.create"],
        domain="data_analysis"
    ),
    # New challenging multi-tool queries (3-4 tools required)
    TestQuery(
        query="Analyze code repository, find security issues, search for fixes, and document findings",
        optimal_tools=["github_mcp", "filesystem_mcp", "search_mcp", "notion_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["code.analyze", "security.scan", "search.solution", "document.create"],
        domain="security",
        expected_success_rate=0.6
    ),
    TestQuery(
        query="Collect weather data for multiple cities, store in database, and create visualization report",
        optimal_tools=["weather_mcp", "sqlite_mcp", "filesystem_mcp", "search_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["weather.multi", "database.store", "report.visualize", "search.tools"],
        domain="data_analysis",
        expected_success_rate=0.5
    ),
    TestQuery(
        query="Monitor system performance, query historical metrics, find optimization guides, and apply fixes",
        optimal_tools=["system_mcp", "postgres_mcp", "search_mcp", "filesystem_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["system.monitor", "metrics.history", "search.optimize", "config.update"],
        domain="system_optimization",
        expected_success_rate=0.55
    ),
    TestQuery(
        query="Analyze stock portfolio, search market news, update database, and create investment report",
        optimal_tools=["zerodha_mcp", "search_mcp", "sqlite_mcp", "notion_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["finance.analyze", "news.search", "database.update", "report.create"],
        domain="finance",
        expected_success_rate=0.5
    ),
    TestQuery(
        query="Find all test files, run tests, analyze failures, and search for debugging solutions",
        optimal_tools=["filesystem_mcp", "github_mcp", "search_mcp", "sqlite_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["test.find", "test.run", "failure.analyze", "debug.search"],
        domain="testing",
        expected_success_rate=0.6
    ),
    TestQuery(
        query="Extract API logs, analyze error patterns, search documentation, and create incident report",
        optimal_tools=["filesystem_mcp", "postgres_mcp", "search_mcp", "notion_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["log.extract", "error.analyze", "docs.search", "incident.report"],
        domain="troubleshooting",
        expected_success_rate=0.55
    ),
    TestQuery(
        query="Compare code branches, identify conflicts, search merge strategies, and update documentation",
        optimal_tools=["github_mcp", "filesystem_mcp", "search_mcp", "notion_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["git.compare", "conflict.find", "strategy.search", "docs.update"],
        domain="development",
        expected_success_rate=0.6
    ),
    TestQuery(
        query="Query customer data, analyze trends, search competitors, and prepare market report",
        optimal_tools=["postgres_mcp", "sqlite_mcp", "search_mcp", "filesystem_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["customer.query", "trend.analyze", "competitor.search", "report.prepare"],
        domain="business_analysis",
        expected_success_rate=0.5
    ),
    TestQuery(
        query="Scan infrastructure configs, check compliance rules, search regulations, and generate audit report",
        optimal_tools=["filesystem_mcp", "postgres_mcp", "search_mcp", "notion_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["config.scan", "compliance.check", "regulation.search", "audit.report"],
        domain="compliance",
        expected_success_rate=0.55
    ),
    TestQuery(
        query="Analyze database performance, find slow queries, search optimization tips, and implement indexes",
        optimal_tools=["sqlite_mcp", "postgres_mcp", "search_mcp", "filesystem_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["db.analyze", "query.profile", "optimize.search", "index.create"],
        domain="database_optimization",
        expected_success_rate=0.5
    ),
    TestQuery(
        query="Extract deployment logs, identify failures, search solutions, and update runbooks",
        optimal_tools=["filesystem_mcp", "github_mcp", "search_mcp", "notion_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["deploy.logs", "failure.identify", "solution.search", "runbook.update"],
        domain="operations",
        expected_success_rate=0.6
    ),
    TestQuery(
        query="Monitor API usage, query billing data, find cost optimization guides, and create budget report",
        optimal_tools=["system_mcp", "postgres_mcp", "search_mcp", "filesystem_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["api.monitor", "billing.query", "cost.optimize", "budget.report"],
        domain="cost_management",
        expected_success_rate=0.55
    ),
    TestQuery(
        query="Analyze code dependencies, find vulnerabilities, search patches, and update packages",
        optimal_tools=["github_mcp", "filesystem_mcp", "search_mcp", "system_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["dependency.analyze", "vuln.scan", "patch.search", "package.update"],
        domain="security",
        expected_success_rate=0.5
    ),
    TestQuery(
        query="Collect performance metrics, compare with baselines, search best practices, and optimize configuration",
        optimal_tools=["system_mcp", "sqlite_mcp", "search_mcp", "filesystem_mcp"],
        complexity=QueryComplexity.COMPLEX,
        intents=["metrics.collect", "baseline.compare", "practice.search", "config.optimize"],
        domain="performance",
        expected_success_rate=0.55
    )
]


# Ambiguous queries requiring disambiguation
AMBIGUOUS_QUERIES = [
    TestQuery(
        query="Show me the data",
        optimal_tools=["sqlite_mcp", "filesystem_mcp"],  # Could be database or files
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["data.show"],
        domain="general",
        expected_success_rate=0.7
    ),
    TestQuery(
        query="Get information about Python",
        optimal_tools=["search_mcp", "filesystem_mcp"],  # Could be docs or files
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["info.get"],
        domain="general",
        expected_success_rate=0.7
    ),
    TestQuery(
        query="Find the latest updates",
        optimal_tools=["github_mcp", "search_mcp"],  # Could be code or news
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["update.find"],
        domain="general",
        expected_success_rate=0.7
    ),
    TestQuery(
        query="Check the status",
        optimal_tools=["system_mcp", "github_mcp", "postgres_mcp"],  # Multiple possibilities
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["status.check"],
        domain="general",
        expected_success_rate=0.6
    ),
    TestQuery(
        query="Analyze the performance",
        optimal_tools=["system_mcp", "sqlite_mcp"],  # System or query performance
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["performance.analyze"],
        domain="general",
        expected_success_rate=0.7
    ),
    TestQuery(
        query="Get the report",
        optimal_tools=["filesystem_mcp", "notion_mcp"],  # File or note
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["report.get"],
        domain="general",
        expected_success_rate=0.7
    ),
    TestQuery(
        query="Search for issues",
        optimal_tools=["github_mcp", "search_mcp"],  # Code issues or general
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["issue.search"],
        domain="general",
        expected_success_rate=0.7
    ),
    TestQuery(
        query="Review the changes",
        optimal_tools=["github_mcp", "filesystem_mcp"],  # Git or file changes
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["change.review"],
        domain="general",
        expected_success_rate=0.7
    ),
    TestQuery(
        query="Get the latest data",
        optimal_tools=["weather_mcp", "zerodha_mcp", "sqlite_mcp"],  # Many options
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["data.latest"],
        domain="general",
        expected_success_rate=0.6
    ),
    TestQuery(
        query="Show me trends",
        optimal_tools=["search_mcp", "sqlite_mcp", "zerodha_mcp"],  # Various trends
        complexity=QueryComplexity.AMBIGUOUS,
        intents=["trend.show"],
        domain="general",
        expected_success_rate=0.6
    )
]


# Domain-specific query sets
DOMAIN_QUERIES = {
    "file_operations": [
        TestQuery(
            query="Find all log files from last week",
            optimal_tools=["filesystem_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["file.find"],
            domain="file_operations"
        ),
        TestQuery(
            query="Delete temporary files older than 30 days",
            optimal_tools=["filesystem_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["file.delete"],
            domain="file_operations"
        ),
        TestQuery(
            query="Compare file sizes between directories",
            optimal_tools=["filesystem_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["file.compare"],
            domain="file_operations"
        )
    ],
    "data_analysis": [
        TestQuery(
            query="Calculate average order value by month",
            optimal_tools=["sqlite_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["database.aggregate"],
            domain="data_analysis"
        ),
        TestQuery(
            query="Join customer and order tables",
            optimal_tools=["postgres_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["database.join"],
            domain="data_analysis"
        ),
        TestQuery(
            query="Export query results to CSV",
            optimal_tools=["sqlite_mcp", "filesystem_mcp"],
            complexity=QueryComplexity.COMPLEX,
            intents=["database.export", "file.write"],
            domain="data_analysis"
        )
    ],
    "web_search": [
        TestQuery(
            query="Find Python async programming tutorials",
            optimal_tools=["search_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["search.tutorial"],
            domain="web_search"
        ),
        TestQuery(
            query="Research best practices for API design",
            optimal_tools=["search_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["search.research"],
            domain="web_search"
        ),
        TestQuery(
            query="Compare cloud service providers",
            optimal_tools=["search_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["search.compare"],
            domain="web_search"
        )
    ],
    "development": [
        TestQuery(
            query="Find functions with cyclomatic complexity > 10",
            optimal_tools=["github_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["code.analyze"],
            domain="development"
        ),
        TestQuery(
            query="List pull requests needing review",
            optimal_tools=["github_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["git.pr"],
            domain="development"
        ),
        TestQuery(
            query="Identify code duplication across files",
            optimal_tools=["github_mcp", "filesystem_mcp"],
            complexity=QueryComplexity.COMPLEX,
            intents=["code.analyze", "file.compare"],
            domain="development"
        )
    ],
    "business": [
        TestQuery(
            query="Track competitor pricing changes",
            optimal_tools=["search_mcp", "sqlite_mcp"],
            complexity=QueryComplexity.COMPLEX,
            intents=["search.monitor", "database.store"],
            domain="business"
        ),
        TestQuery(
            query="Analyze customer churn patterns",
            optimal_tools=["postgres_mcp"],
            complexity=QueryComplexity.SIMPLE,
            intents=["database.analyze"],
            domain="business"
        ),
        TestQuery(
            query="Generate monthly sales report",
            optimal_tools=["sqlite_mcp", "notion_mcp"],
            complexity=QueryComplexity.COMPLEX,
            intents=["database.report", "note.create"],
            domain="business"
        )
    ]
}


def get_all_queries() -> List[TestQuery]:
    """Get all test queries."""
    all_queries = []
    all_queries.extend(SIMPLE_QUERIES)
    all_queries.extend(COMPLEX_QUERIES)
    all_queries.extend(AMBIGUOUS_QUERIES)
    for domain_queries in DOMAIN_QUERIES.values():
        all_queries.extend(domain_queries)
    return all_queries


def get_queries_by_complexity(complexity: QueryComplexity) -> List[TestQuery]:
    """Get queries filtered by complexity."""
    return [q for q in get_all_queries() if q.complexity == complexity]


def get_queries_by_domain(domain: str) -> List[TestQuery]:
    """Get queries filtered by domain."""
    return [q for q in get_all_queries() if q.domain == domain]


def get_evaluation_sets() -> Dict[str, List[TestQuery]]:
    """Get organized evaluation sets for experiments."""
    return {
        "quick_test": SIMPLE_QUERIES[:5] + COMPLEX_QUERIES[:3] + AMBIGUOUS_QUERIES[:2],
        "simple_only": SIMPLE_QUERIES,
        "complex_only": COMPLEX_QUERIES,
        "ambiguous_only": AMBIGUOUS_QUERIES,
        "full_evaluation": get_all_queries(),
        # Updated: Reduced simple queries, increased complex queries for harder evaluation
        "dissertation_core": SIMPLE_QUERIES[:5] + COMPLEX_QUERIES[:15] + AMBIGUOUS_QUERIES[:5],
        # New: Hard evaluation set with 80% complex queries
        "hard_evaluation": SIMPLE_QUERIES[:3] + COMPLEX_QUERIES[-15:] + AMBIGUOUS_QUERIES[:2]
    }


# Export ground truth for evaluation
def export_ground_truth() -> Dict[str, Any]:
    """Export ground truth data for evaluation scripts."""
    ground_truth = {}
    for query in get_all_queries():
        ground_truth[query.query] = {
            "optimal_tools": query.optimal_tools,
            "complexity": query.complexity.value,
            "intents": query.intents,
            "domain": query.domain,
            "expected_success_rate": query.expected_success_rate
        }
    return ground_truth


if __name__ == "__main__":
    # Print statistics when run directly
    all_queries = get_all_queries()
    print(f"Total test queries: {len(all_queries)}")
    print(f"Simple queries: {len(SIMPLE_QUERIES)}")
    print(f"Complex queries: {len(COMPLEX_QUERIES)}")
    print(f"Ambiguous queries: {len(AMBIGUOUS_QUERIES)}")
    print(f"\nDomain distribution:")
    domain_counts = {}
    for q in all_queries:
        domain_counts[q.domain] = domain_counts.get(q.domain, 0) + 1
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")