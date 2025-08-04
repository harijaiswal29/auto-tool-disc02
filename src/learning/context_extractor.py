"""Context Extractor for lightweight context-aware pattern mining.

This module extracts user expertise and domain context from queries and user history
to enable context-aware pattern discovery.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re

from src.utils.logger import get_logger

logger = get_logger("ContextExtractor")


@dataclass
class UserContext:
    """Represents extracted user and domain context."""
    user_expertise: str  # novice, intermediate, expert
    domain: str  # general, engineering, data_science, web_dev, devops
    raw_expertise_indicators: Dict[str, float]
    raw_domain_indicators: Dict[str, float]


class ContextExtractor:
    """Extracts user expertise and domain context for pattern mining."""
    
    def __init__(self):
        """Initialize context extractor with domain mappings."""
        # Domain keywords mapping
        self.domain_keywords = {
            'engineering': [
                'code', 'build', 'compile', 'git', 'debug', 'function',
                'class', 'module', 'refactor', 'test', 'bug', 'fix',
                'python', 'javascript', 'java', 'cpp', 'rust'
            ],
            'data_science': [
                'analyze', 'data', 'model', 'csv', 'dataset', 'statistics',
                'machine learning', 'ml', 'ai', 'predict', 'train',
                'pandas', 'numpy', 'jupyter', 'visualization', 'plot'
            ],
            'web_dev': [
                'html', 'css', 'javascript', 'frontend', 'backend', 'api',
                'website', 'webpage', 'react', 'vue', 'angular', 'node',
                'server', 'client', 'responsive', 'ui', 'ux', 'styling'
            ],
            'devops': [
                'deploy', 'docker', 'kubernetes', 'ci', 'cd', 'pipeline',
                'container', 'orchestration', 'monitoring', 'logs',
                'infrastructure', 'cloud', 'aws', 'azure', 'terraform'
            ],
            'database': [
                'sql', 'query', 'database', 'table', 'schema', 'index',
                'postgres', 'mysql', 'mongodb', 'redis', 'nosql',
                'migration', 'backup', 'restore'
            ]
        }
        
        # Expertise indicators
        self.expertise_indicators = {
            'novice': {
                'simple_queries': ['what is', 'how to', 'help', 'show me', 'example'],
                'basic_tools': ['filesystem_mcp'],
                'low_complexity': True
            },
            'intermediate': {
                'specific_queries': ['find', 'search', 'update', 'modify'],
                'multiple_tools': True,
                'moderate_complexity': True
            },
            'expert': {
                'complex_queries': ['optimize', 'refactor', 'integrate', 'automate'],
                'advanced_tools': ['git_mcp', 'postgres_mcp', 'github_mcp'],
                'high_complexity': True
            }
        }
        
        logger.info("ContextExtractor initialized")
    
    def extract_context(self, query: str, user_stats: Optional[Dict[str, Any]] = None,
                       intent_type: Optional[str] = None) -> UserContext:
        """Extract user expertise and domain from query and user statistics.
        
        Args:
            query: User's query text
            user_stats: Optional user statistics (success_rate, query_count, etc.)
            intent_type: Optional intent type from intent recognition
            
        Returns:
            UserContext with expertise and domain information
        """
        # Extract domain
        domain, domain_scores = self._extract_domain(query, intent_type)
        
        # Extract expertise
        expertise, expertise_scores = self._extract_expertise(query, user_stats)
        
        return UserContext(
            user_expertise=expertise,
            domain=domain,
            raw_expertise_indicators=expertise_scores,
            raw_domain_indicators=domain_scores
        )
    
    def _extract_domain(self, query: str, intent_type: Optional[str] = None) -> tuple[str, Dict[str, float]]:
        """Extract domain from query text and intent.
        
        Args:
            query: User's query text
            intent_type: Optional intent type
            
        Returns:
            Tuple of (domain, domain_scores)
        """
        query_lower = query.lower()
        domain_scores = {}
        domain_match_counts = {}  # Track number of matches for tie-breaking
        
        # Calculate keyword matches for each domain
        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            matches = 0
            
            for keyword in keywords:
                if keyword in query_lower:
                    # Longer keywords get higher weight
                    weight = len(keyword.split()) * 0.5
                    score += weight
                    matches += 1
            
            # Normalize by square root of number of keywords for better scaling
            if matches > 0:
                domain_scores[domain] = score / (len(keywords) ** 0.5)
                domain_match_counts[domain] = matches
        
        # Consider intent type for domain hints
        if intent_type:
            if 'create' in intent_type or 'modify' in intent_type:
                domain_scores['engineering'] = domain_scores.get('engineering', 0) + 0.2
            elif 'analyze' in intent_type:
                domain_scores['data_science'] = domain_scores.get('data_science', 0) + 0.2
            elif 'deploy' in intent_type or 'monitor' in intent_type:
                domain_scores['devops'] = domain_scores.get('devops', 0) + 0.2
        
        # Select domain with highest score (use match count for tie-breaking)
        if domain_scores:
            # Sort by score first, then by match count for tie-breaking
            sorted_domains = sorted(
                domain_scores.items(), 
                key=lambda x: (x[1], domain_match_counts.get(x[0], 0)),
                reverse=True
            )
            best_domain = sorted_domains[0]
            
            # Only select if score is significant
            if best_domain[1] > 0.1:
                return best_domain[0], domain_scores
        
        return 'general', domain_scores
    
    def _extract_expertise(self, query: str, user_stats: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, float]]:
        """Extract user expertise level from query and statistics.
        
        Args:
            query: User's query text
            user_stats: Optional user statistics
            
        Returns:
            Tuple of (expertise_level, expertise_scores)
        """
        query_lower = query.lower()
        expertise_scores = {
            'novice': 0.0,
            'intermediate': 0.0,
            'expert': 0.0
        }
        
        # Analyze query complexity
        query_words = query_lower.split()
        query_length = len(query_words)
        
        # Check for expertise indicators in query
        for level, indicators in self.expertise_indicators.items():
            # Check for query patterns
            if 'simple_queries' in indicators:
                for pattern in indicators['simple_queries']:
                    if pattern in query_lower:
                        expertise_scores[level] += 0.3
            
            if 'specific_queries' in indicators:
                for pattern in indicators['specific_queries']:
                    if pattern in query_lower:
                        expertise_scores[level] += 0.3
            
            if 'complex_queries' in indicators:
                for pattern in indicators['complex_queries']:
                    if pattern in query_lower:
                        expertise_scores[level] += 0.3
        
        # Query complexity scoring
        if query_length < 5:
            expertise_scores['novice'] += 0.2
        elif query_length < 10:
            expertise_scores['intermediate'] += 0.2
        else:
            expertise_scores['expert'] += 0.2
        
        # Use of technical terms (progressive scoring)
        technical_terms = ['api', 'endpoint', 'schema', 'regex', 'async', 'pipeline', 'optimize']
        tech_count = sum(1 for term in technical_terms if term in query_lower)
        if tech_count == 0:
            expertise_scores['novice'] += 0.1
        elif tech_count <= 2:
            expertise_scores['intermediate'] += 0.1
        else:
            # Progressive scoring for expert - more terms = higher score
            expertise_scores['expert'] += 0.1 + (tech_count - 2) * 0.05
        
        # Consider user statistics if available
        if user_stats:
            success_rate = user_stats.get('success_rate', 0.5)
            query_count = user_stats.get('query_count', 0)
            avg_tools_used = user_stats.get('avg_tools_used', 1)
            
            # Success rate contribution
            if success_rate < 0.6:
                expertise_scores['novice'] += 0.3
            elif success_rate < 0.8:
                expertise_scores['intermediate'] += 0.3
            else:
                expertise_scores['expert'] += 0.3
            
            # Experience contribution
            if query_count < 10:
                expertise_scores['novice'] += 0.2
            elif query_count < 50:
                expertise_scores['intermediate'] += 0.2
            else:
                expertise_scores['expert'] += 0.2
            
            # Tool usage complexity
            if avg_tools_used < 1.5:
                expertise_scores['novice'] += 0.1
            elif avg_tools_used < 2.5:
                expertise_scores['intermediate'] += 0.1
            else:
                expertise_scores['expert'] += 0.1
        
        # Determine expertise level
        sorted_levels = sorted(expertise_scores.items(), key=lambda x: x[1], reverse=True)
        best_level = sorted_levels[0]
        
        # Default to intermediate if top two scores are too close
        if len(sorted_levels) > 1 and abs(sorted_levels[0][1] - sorted_levels[1][1]) < 0.05:
            # If intermediate is one of the top two, prefer it
            if 'intermediate' in [sorted_levels[0][0], sorted_levels[1][0]]:
                return 'intermediate', expertise_scores
        
        return best_level[0], expertise_scores
    
    def get_context_vector(self, context: UserContext) -> List[float]:
        """Convert context to numerical vector for Q-learning state.
        
        Args:
            context: UserContext object
            
        Returns:
            List of floats representing context (8 dimensions total)
        """
        # User expertise one-hot encoding (3 dimensions)
        expertise_vector = [0.0, 0.0, 0.0]
        expertise_map = {'novice': 0, 'intermediate': 1, 'expert': 2}
        if context.user_expertise in expertise_map:
            expertise_vector[expertise_map[context.user_expertise]] = 1.0
        
        # Domain one-hot encoding (5 dimensions)
        domain_vector = [0.0, 0.0, 0.0, 0.0, 0.0]
        domain_map = {
            'general': 0,
            'engineering': 1,
            'data_science': 2,
            'web_dev': 3,
            'devops': 4
        }
        if context.domain in domain_map:
            domain_vector[domain_map[context.domain]] = 1.0
        
        return expertise_vector + domain_vector