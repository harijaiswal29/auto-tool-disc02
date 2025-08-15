"""
Tool Discovery Agent for Autonomous Tool Discovery System.

This agent specializes in discovering and recommending tools based on
user intents and requirements using semantic search and graph exploration.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from src.agents.intent_recognition_agent import IntentResult
from src.core.tool_registry import ToolRegistry
from src.utils.logger import get_logger


@dataclass
class ToolCandidate:
    """Represents a tool candidate with scoring information."""
    tool_id: str
    tool_name: str
    tool_type: str
    capabilities: List[str]
    semantic_score: float = 0.0
    capability_score: float = 0.0
    performance_score: float = 0.0
    relationship_score: float = 0.0
    overall_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolDiscoveryAgent:
    """
    Agent responsible for discovering tools that match user intents.
    
    Uses multiple strategies:
    1. Semantic search using embeddings
    2. Capability-based matching
    3. Graph-based relationship exploration
    4. Performance-based ranking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Tool Discovery Agent."""
        self.logger = get_logger(__name__)
        
        # Load configuration
        if config is None:
            config = self._load_default_config()
        self.config = config
        
        # Initialize components
        self.logger.info("Initializing Tool Discovery Agent...")
        
        # Sentence transformer for semantic search
        model_name = config.get('model', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(model_name, device='cpu')
        
        # Tool Registry
        registry_path = config.get('database', {}).get('tool_registry', 'data/registry/tools.db')
        self.tool_registry = ToolRegistry(registry_path)
        
        # Tool relationship graph
        self.tool_graph = nx.DiGraph()
        
        # Cache for tool embeddings
        self.tool_embeddings = {}
        self.embedding_cache_size = config.get('cache_size', 1000)
        
        # Intent to capability mapping (comprehensive)
        self.intent_capability_map = {
            'query.search': [
                'search', 'find', 'query', 'list', 'discover', 'locate',
                'browse', 'explore', 'lookup', 'scan'
            ],
            'query.retrieve': [
                'read', 'get', 'fetch', 'retrieve', 'load', 'access',
                'view', 'show', 'display', 'extract'
            ],
            'query.analyze': [
                'view', 'show', 'display', 'report', 'insight',
                'metric', 'statistic', 'result', 'finding', 'summary'
            ],
            'action.create': [
                'create', 'write', 'generate', 'make', 'build', 'produce',
                'compose', 'construct', 'initialize', 'add'
            ],
            'action.modify': [
                'update', 'edit', 'modify', 'change', 'alter', 'revise',
                'adjust', 'transform', 'configure', 'patch'
            ],
            'action.delete': [
                'delete', 'remove', 'clear', 'drop', 'erase', 'destroy',
                'purge', 'clean', 'eliminate', 'discard'
            ],
            'action.analyze': [
                'analyze', 'examine', 'inspect', 'evaluate', 'assess',
                'investigate', 'study', 'review', 'diagnose', 'profile',
                'process', 'compute', 'interpret', 'measure'
            ],
            'action.export': [
                'export', 'save', 'output', 'download', 'write', 'dump',
                'serialize', 'backup', 'convert', 'extract'
            ],
            'system.configure': [
                'configure', 'setup', 'initialize', 'install', 'deploy',
                'provision', 'prepare', 'customize', 'tune', 'optimize'
            ],
            'system.monitor': [
                'monitor', 'track', 'watch', 'observe', 'supervise',
                'audit', 'log', 'trace', 'measure', 'report'
            ]
        }
        
        # Capability synonyms for better matching - ENHANCED
        self.capability_synonyms = {
            # Query operations
            'search': ['find', 'query', 'lookup', 'locate', 'discover', 'seek', 'explore', 'scan', 'filter'],
            'read': ['get', 'fetch', 'retrieve', 'load', 'access', 'view', 'show', 'extract', 'pull', 'obtain'],
            'list': ['enumerate', 'show', 'display', 'index', 'catalog', 'directory', 'browse', 'inventory'],
            
            # Modification operations
            'write': ['create', 'save', 'store', 'add', 'insert', 'put', 'generate', 'produce', 'compose'],
            'update': ['modify', 'edit', 'change', 'alter', 'revise', 'patch', 'amend', 'adjust', 'refresh'],
            'delete': ['remove', 'drop', 'clear', 'erase', 'destroy', 'purge', 'eliminate', 'discard', 'clean'],
            
            # Analysis operations
            'analyze': ['examine', 'inspect', 'evaluate', 'assess', 'investigate', 'study', 'diagnose', 'review'],
            'process': ['handle', 'execute', 'run', 'perform', 'compute', 'calculate', 'transform', 'convert'],
            'export': ['output', 'save', 'download', 'extract', 'transfer', 'dump', 'serialize', 'backup'],
            
            # System operations
            'configure': ['setup', 'initialize', 'install', 'customize', 'tune', 'prepare', 'provision'],
            'monitor': ['track', 'watch', 'observe', 'supervise', 'check', 'audit', 'measure', 'log'],
            'connect': ['link', 'join', 'attach', 'bind', 'couple', 'interface', 'integrate', 'sync'],
            
            # Data operations
            'import': ['load', 'upload', 'ingest', 'read', 'fetch', 'pull', 'acquire', 'collect'],
            'filter': ['select', 'choose', 'pick', 'screen', 'sift', 'refine', 'narrow', 'constrain'],
            'sort': ['order', 'arrange', 'organize', 'rank', 'classify', 'group', 'categorize'],
            
            # File operations
            'copy': ['duplicate', 'clone', 'replicate', 'backup', 'mirror', 'reproduce'],
            'move': ['transfer', 'relocate', 'migrate', 'shift', 'transport', 'displace'],
            'compress': ['zip', 'archive', 'pack', 'compact', 'shrink', 'reduce']
        }
        
        # Build reverse mapping for bidirectional lookup
        self.reverse_synonyms = {}
        for primary, synonyms in self.capability_synonyms.items():
            for synonym in synonyms:
                if synonym not in self.reverse_synonyms:
                    self.reverse_synonyms[synonym] = []
                self.reverse_synonyms[synonym].append(primary)
        
        self.logger.info("Tool Discovery Agent initialized successfully")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'model': 'all-MiniLM-L6-v2',
            'cache_size': 1000,
            'discovery': {
                'semantic_weight': 0.3,
                'capability_weight': 0.3,
                'performance_weight': 0.2,
                'relationship_weight': 0.2
            },
            'min_score_threshold': 0.1
        }
    
    async def discover_tools(self, intent_result: IntentResult, 
                           context: Optional[Dict[str, Any]] = None) -> List[ToolCandidate]:
        """
        Discover tools that match the given intent.
        
        Args:
            intent_result: Result from intent recognition
            context: Optional context information
            
        Returns:
            List of tool candidates ranked by relevance
        """
        if context is None:
            context = {}
        
        self.logger.info(f"Discovering tools for intent: {intent_result.primary_intent.type}")
        
        # Get all available tools
        all_tools = await self.tool_registry.get_all_tools()
        
        if not all_tools:
            self.logger.warning("No tools available in registry")
            return []
        
        # Build tool graph if not already built
        if not self.tool_graph.nodes():
            await self._build_tool_graph()
        
        # Score each tool
        candidates = []
        for tool in all_tools:
            candidate = await self._score_tool(tool, intent_result, context)
            if candidate.overall_score >= self.config.get('min_score_threshold', 0.1):
                candidates.append(candidate)
        
        # Sort by overall score
        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        self.logger.info(f"Discovered {len(candidates)} relevant tools")
        
        return candidates
    
    async def _score_tool(self, tool: Dict[str, Any], 
                         intent_result: IntentResult,
                         context: Dict[str, Any]) -> ToolCandidate:
        """Score a single tool based on multiple criteria."""
        tool_id = tool['id']
        tool_name = tool['name']
        tool_type = tool.get('type', 'unknown')
        
        # Parse capabilities
        capabilities = self._parse_capabilities(tool.get('capabilities', {}))
        
        # Calculate individual scores
        raw_semantic_score = await self._calculate_semantic_score(tool, intent_result)
        raw_capability_score = self._calculate_capability_score(capabilities, intent_result)
        raw_performance_score = tool.get('performance_score', 0.5)
        raw_relationship_score = await self._calculate_relationship_score(tool_id, context)
        
        # Normalize all scores to ensure they're in [0, 1] range
        # Use sigmoid for semantic scores to spread them out better
        semantic_score = self._normalize_score(raw_semantic_score, method='sigmoid')
        # Use minmax for capability scores as they're already well-distributed
        capability_score = self._normalize_score(raw_capability_score, method='minmax')
        # Use minmax for performance scores
        performance_score = self._normalize_score(raw_performance_score, method='minmax')
        # Use minmax for relationship scores
        relationship_score = self._normalize_score(raw_relationship_score, method='minmax')
        
        # Calculate weighted overall score
        weights = self.config.get('discovery', {})
        overall_score = (
            semantic_score * weights.get('semantic_weight', 0.3) +
            capability_score * weights.get('capability_weight', 0.3) +
            performance_score * weights.get('performance_weight', 0.2) +
            relationship_score * weights.get('relationship_weight', 0.2)
        )
        
        return ToolCandidate(
            tool_id=tool_id,
            tool_name=tool_name,
            tool_type=tool_type,
            capabilities=capabilities,
            semantic_score=semantic_score,
            capability_score=capability_score,
            performance_score=performance_score,
            relationship_score=relationship_score,
            overall_score=overall_score,
            metadata=tool
        )
    
    def _parse_capabilities(self, capabilities: Any) -> List[str]:
        """Parse capabilities from various formats."""
        if isinstance(capabilities, str):
            try:
                capabilities = json.loads(capabilities)
            except:
                return [capabilities]
        
        if isinstance(capabilities, dict):
            # Extract operation names AND categories
            operations = capabilities.get('operations', [])
            cap_list = []
            for op in operations:
                if isinstance(op, dict):
                    # Add operation name
                    name = op.get('name', '')
                    if name:
                        cap_list.append(name)
                    # Also add category if present
                    category = op.get('category', '')
                    if category and category not in cap_list:
                        cap_list.append(category)
                else:
                    cap_list.append(str(op))
            return cap_list
        
        if isinstance(capabilities, list):
            return [str(c) for c in capabilities]
        
        return []
    
    def _build_tool_description(self, tool: Dict[str, Any]) -> str:
        """Build a comprehensive tool description for embedding generation."""
        tool_name = tool.get('name', '')
        tool_description = tool.get('description', '')
        tool_type = tool.get('type', '')
        capabilities = self._parse_capabilities(tool.get('capabilities', {}))
        
        # Build rich description with proper weighting
        description_parts = []
        
        # Name is most important
        if tool_name:
            description_parts.append(f"Tool: {tool_name}")
        
        # Full description carries significant weight
        if tool_description:
            description_parts.append(f"Description: {tool_description}")
        
        # Tool type helps with categorization
        if tool_type:
            description_parts.append(f"Type: {tool_type}")
        
        # Capabilities provide functional context
        if capabilities:
            # Expand capability descriptions with synonyms
            expanded_caps = []
            for cap in capabilities:
                expanded_caps.append(cap)
                # Add primary term if it's a synonym
                cap_lower = cap.lower()
                if cap_lower in self.reverse_synonyms:
                    expanded_caps.extend(self.reverse_synonyms[cap_lower])
            
            description_parts.append(f"Capabilities: {', '.join(expanded_caps)}")
        
        # Add endpoint/server type for additional context
        if 'endpoint' in tool:
            endpoint_type = tool['endpoint'].split('://')[0] if '://' in tool['endpoint'] else 'unknown'
            description_parts.append(f"Protocol: {endpoint_type}")
        
        if 'server_type' in tool:
            description_parts.append(f"Server Type: {tool['server_type']}")
        
        # Combine all parts with proper spacing
        return ' | '.join(description_parts)
    
    def _normalize_score(self, score: float, method: str = 'sigmoid') -> float:
        """
        Normalize a score to [0, 1] range using various methods.
        
        Args:
            score: Raw score to normalize
            method: Normalization method ('sigmoid', 'minmax', 'tanh')
            
        Returns:
            Normalized score in [0, 1] range
        """
        if method == 'sigmoid':
            # Sigmoid normalization with adjustable steepness
            # Scores around 0.5 map to ~0.5, higher scores approach 1
            import math
            return 1 / (1 + math.exp(-10 * (score - 0.5)))
        
        elif method == 'tanh':
            # Tanh normalization scaled to [0, 1]
            import math
            return (math.tanh(2 * score - 1) + 1) / 2
        
        elif method == 'minmax':
            # Simple clipping to [0, 1]
            return max(0.0, min(1.0, score))
        
        else:
            # Default to simple clipping
            return max(0.0, min(1.0, score))
    
    async def _calculate_semantic_score(self, tool: Dict[str, Any], 
                                       intent_result: IntentResult) -> float:
        """Calculate semantic similarity between tool and intent."""
        # Create enhanced tool description using helper method
        tool_desc = self._build_tool_description(tool)
        
        # Get or create tool embedding
        tool_embedding = await self._get_tool_embedding(tool['id'], tool_desc)
        
        # Get query embedding
        # Check if embedding is in metadata.features or metadata directly
        if hasattr(intent_result, 'features') and intent_result.features:
            query_embedding = intent_result.features.get('embedding')
        elif hasattr(intent_result, 'metadata') and intent_result.metadata:
            features = intent_result.metadata.get('features', {})
            query_embedding = features.get('embedding') if features else None
        else:
            query_embedding = None
        
        if query_embedding is None:
            # Compute query embedding if not provided
            query_text = getattr(intent_result, 'raw_query', '') or getattr(intent_result, 'original_query', '')
            if query_text:
                query_embedding = self.model.encode(query_text)
            else:
                # If no query text and no embedding, use keywords as fallback
                if hasattr(intent_result, 'primary_intent') and intent_result.primary_intent:
                    keywords = getattr(intent_result.primary_intent, 'keywords', [])
                    if keywords:
                        query_text = ' '.join(keywords)
                        query_embedding = self.model.encode(query_text)
                    else:
                        return 0.0
                else:
                    return 0.0
        
        # Calculate cosine similarity
        similarity = cosine_similarity([tool_embedding], [query_embedding])[0][0]
        
        # Apply keyword-based boost for better differentiation
        query_lower = (getattr(intent_result, 'raw_query', '') or '').lower()
        tool_name = tool.get('name', '').lower()
        tool_id = tool.get('id', '').lower()
        
        # Give bonus points for exact keyword matches
        bonus = 0.0
        if 'news' in query_lower and 'news' in tool_name:
            bonus += 0.2  # Strong boost for news match
        elif 'latest' in query_lower and 'news' in tool_name:
            bonus += 0.15  # Boost for "latest" implying news
        elif 'image' in query_lower and 'image' in tool_name:
            bonus += 0.2  # Strong boost for image match
        elif 'video' in query_lower and 'video' in tool_name:
            bonus += 0.2  # Strong boost for video match
        elif 'github' in query_lower and 'github' in tool_id:
            bonus += 0.15  # Boost for GitHub-specific queries
        elif 'repository' in query_lower and 'github' in tool_id:
            bonus += 0.15  # Boost for repository queries
        elif 'code' in query_lower and 'github' in tool_id:
            bonus += 0.1  # Smaller boost for code queries
        
        # Apply penalty for mismatched types
        penalty = 0.0
        if 'news' in query_lower and 'github' in tool_id:
            penalty = 0.15  # News queries shouldn't prefer GitHub
        elif 'news' in query_lower and ('image' in tool_name or 'video' in tool_name):
            penalty = 0.1  # News queries shouldn't prefer image/video search
        elif 'github' in query_lower and 'search' in tool_id and 'github' not in tool_id:
            penalty = 0.1  # GitHub queries shouldn't prefer general search
        
        # Combine similarity with bonuses/penalties
        final_score = similarity + bonus - penalty
        
        return max(0.0, min(1.0, final_score))
    
    async def _get_tool_embedding(self, tool_id: str, tool_desc: str) -> np.ndarray:
        """Get or compute tool embedding with caching."""
        # Use tool description as part of cache key to handle description changes
        cache_key = f"{tool_id}:{hash(tool_desc)}"
        if cache_key in self.tool_embeddings:
            return self.tool_embeddings[cache_key]
        
        # Compute embedding
        embedding = self.model.encode(tool_desc)
        
        # Cache it
        if len(self.tool_embeddings) >= self.embedding_cache_size:
            # Remove oldest entry
            self.tool_embeddings.pop(next(iter(self.tool_embeddings)))
        
        self.tool_embeddings[cache_key] = embedding
        
        return embedding
    
    def _calculate_capability_score(self, tool_capabilities: List[str], 
                                   intent_result: IntentResult) -> float:
        """Calculate capability match score with enhanced synonym matching."""
        if not tool_capabilities:
            # Give a small default score rather than 0 to avoid penalizing tools without explicit capabilities
            return 0.3
        
        # Get required capabilities for the intent
        intent_type = intent_result.primary_intent.type
        required_capabilities = self.intent_capability_map.get(intent_type, [])
        
        # Also consider keywords from the query
        query_keywords = intent_result.primary_intent.keywords
        
        # Fallback: If no keywords, extract from raw query
        if not query_keywords and hasattr(intent_result, 'raw_query'):
            # Simple keyword extraction from raw query
            raw_query = intent_result.raw_query.lower()
            # Common action words to extract
            action_words = ['read', 'write', 'get', 'fetch', 'create', 'update', 'delete', 
                           'search', 'find', 'query', 'analyze', 'export', 'import',
                           'load', 'save', 'list', 'show', 'display', 'modify', 'edit']
            query_keywords = [word for word in action_words if word in raw_query]
        
        # Normalize and expand tool capabilities with synonyms
        tool_caps_expanded = set()
        for cap in tool_capabilities:
            cap_lower = cap.lower()
            tool_caps_expanded.add(cap_lower)
            
            # Add primary terms if this is a synonym
            if cap_lower in self.reverse_synonyms:
                tool_caps_expanded.update(self.reverse_synonyms[cap_lower])
            
            # Add synonyms if this is a primary term
            if cap_lower in self.capability_synonyms:
                tool_caps_expanded.update(self.capability_synonyms[cap_lower])
        
        # Normalize and expand required capabilities
        required_caps_expanded = set()
        for req_cap in required_capabilities:
            req_cap_lower = req_cap.lower()
            required_caps_expanded.add(req_cap_lower)
            
            # Add synonyms for required capabilities
            if req_cap_lower in self.capability_synonyms:
                required_caps_expanded.update(self.capability_synonyms[req_cap_lower])
        
        # Calculate overlap with enhanced matching
        matches = 0
        matched_capabilities = set()
        
        # Check required capabilities (full weight)
        for req_cap in required_caps_expanded:
            if any(req_cap in tool_cap or tool_cap in req_cap for tool_cap in tool_caps_expanded):
                if req_cap not in matched_capabilities:
                    matches += 1
                    matched_capabilities.add(req_cap)
        
        # Check query keywords (partial weight)
        for keyword in query_keywords:
            keyword_lower = keyword.lower()
            
            # Check direct match or synonym match
            if any(keyword_lower in tool_cap or tool_cap in keyword_lower for tool_cap in tool_caps_expanded):
                matches += 0.7  # Higher weight for keyword matches with synonym support
            elif keyword_lower in self.reverse_synonyms:
                # Check if any synonym matches
                for primary in self.reverse_synonyms[keyword_lower]:
                    if any(primary in tool_cap for tool_cap in tool_caps_expanded):
                        matches += 0.5
                        break
        
        # Calculate total expected matches
        total = len(required_capabilities) + len(query_keywords)
        if total == 0:
            return 0.5  # Neutral score
        
        # Normalize score with bonus for exact matches
        base_score = min(matches / total, 1.0)
        
        # Bonus for having multiple matching capabilities
        diversity_bonus = min(len(matched_capabilities) * 0.05, 0.2)
        
        # If we used fallback keywords and found matches, boost the score
        # This compensates for poor intent classification
        if matches > 0 and required_capabilities and not any(
            cap in tool_caps_expanded for cap in required_capabilities
        ):
            # Intent type doesn't match well, but keywords do - boost score
            base_score = max(base_score, 0.6)
        
        return min(base_score + diversity_bonus, 1.0)
    
    async def _calculate_relationship_score(self, tool_id: str, context: Dict[str, Any]) -> float:
        """Calculate score based on tool relationships and context."""
        if not self.tool_graph.has_node(tool_id):
            return 0.5  # Neutral score
        
        score = 0.5  # Base score
        
        # Check if tool complements recently used tools
        recent_tools = context.get('recent_tools', [])
        for recent_tool in recent_tools:
            if self.tool_graph.has_edge(recent_tool, tool_id):
                edge_data = self.tool_graph.get_edge_data(recent_tool, tool_id)
                if edge_data.get('type') == 'complements':
                    score += 0.2
                elif edge_data.get('type') == 'requires':
                    score += 0.1
        
        # Check tool popularity (degree centrality)
        centrality = nx.degree_centrality(self.tool_graph).get(tool_id, 0)
        score += centrality * 0.2
        
        return min(score, 1.0)
    
    async def _build_tool_graph(self):
        """Build the tool relationship graph."""
        self.logger.info("Building tool relationship graph...")
        
        # Get all tools
        tools = await self.tool_registry.get_all_tools()
        
        # Add nodes
        for tool in tools:
            self.tool_graph.add_node(tool['id'], **tool)
        
        # Get relationships from registry
        relationships = await self.tool_registry.get_tool_relationships()
        
        # Add edges
        for rel in relationships:
            self.tool_graph.add_edge(
                rel['tool1_id'],
                rel['tool2_id'],
                type=rel['relationship_type'],
                strength=rel.get('strength', 0.5)
            )
        
        self.logger.info(f"Tool graph built with {self.tool_graph.number_of_nodes()} nodes "
                        f"and {self.tool_graph.number_of_edges()} edges")
    
    async def find_complementary_tools(self, tool_ids: List[str], max_tools: int = 5) -> List[ToolCandidate]:
        """
        Find tools that complement the given tools.
        
        Args:
            tool_ids: List of tool IDs to find complements for
            max_tools: Maximum number of complementary tools to return
            
        Returns:
            List of complementary tool candidates
        """
        if not self.tool_graph.nodes():
            await self._build_tool_graph()
        
        complementary = {}
        
        for tool_id in tool_ids:
            if not self.tool_graph.has_node(tool_id):
                continue
            
            # Find neighbors with 'complements' relationship
            for neighbor in self.tool_graph.neighbors(tool_id):
                edge_data = self.tool_graph.get_edge_data(tool_id, neighbor)
                if edge_data.get('type') == 'complements':
                    strength = edge_data.get('strength', 0.5)
                    if neighbor in complementary:
                        complementary[neighbor] = max(complementary[neighbor], strength)
                    else:
                        complementary[neighbor] = strength
        
        # Convert to candidates
        candidates = []
        for tool_id, strength in complementary.items():
            if tool_id not in tool_ids:  # Don't include original tools
                tool_data = self.tool_graph.nodes[tool_id]
                candidate = ToolCandidate(
                    tool_id=tool_id,
                    tool_name=tool_data.get('name', tool_id),
                    tool_type=tool_data.get('type', 'unknown'),
                    capabilities=self._parse_capabilities(tool_data.get('capabilities', {})),
                    relationship_score=strength,
                    overall_score=strength,
                    metadata=tool_data
                )
                candidates.append(candidate)
        
        # Sort by relationship strength
        candidates.sort(key=lambda x: x.relationship_score, reverse=True)
        
        return candidates[:max_tools]
    
    async def discover_tools_by_pattern(self, pattern: str) -> List[ToolCandidate]:
        """
        Discover tools using a specific search pattern.
        
        Args:
            pattern: Search pattern (e.g., "database query tools")
            
        Returns:
            List of matching tool candidates
        """
        # Create a mock intent result for pattern-based search
        pattern_embedding = self.model.encode(pattern)
        
        all_tools = await self.tool_registry.get_all_tools()
        candidates = []
        
        for tool in all_tools:
            # Use the same enhanced description builder
            tool_desc = self._build_tool_description(tool)
            tool_embedding = await self._get_tool_embedding(tool['id'], tool_desc)
            
            similarity = cosine_similarity([pattern_embedding], [tool_embedding])[0][0]
            
            if similarity > 0.2:  # Threshold for pattern matching
                candidate = ToolCandidate(
                    tool_id=tool['id'],
                    tool_name=tool['name'],
                    tool_type=tool.get('type', 'unknown'),
                    capabilities=self._parse_capabilities(tool.get('capabilities', {})),
                    semantic_score=similarity,
                    overall_score=similarity,
                    metadata=tool
                )
                candidates.append(candidate)
        
        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        return candidates
    
    async def get_tool_recommendations(self, tool_id: str, max_recommendations: int = 5) -> List[ToolCandidate]:
        """
        Get tool recommendations based on a specific tool.
        
        Args:
            tool_id: ID of the reference tool
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of recommended tools
        """
        recommendations = []
        
        # Find complementary tools
        complementary = await self.find_complementary_tools([tool_id], max_recommendations)
        recommendations.extend(complementary)
        
        # Find similar tools
        if tool_id in self.tool_graph.nodes:
            tool_data = self.tool_graph.nodes[tool_id]
            tool_desc = f"{tool_data['name']} {tool_data.get('type', '')}"
            similar = await self.discover_tools_by_pattern(tool_desc)
            
            # Filter out the original tool
            similar = [t for t in similar if t.tool_id != tool_id]
            
            # Add to recommendations if not already there
            for tool in similar[:max_recommendations - len(recommendations)]:
                if not any(t.tool_id == tool.tool_id for t in recommendations):
                    recommendations.append(tool)
        
        return recommendations[:max_recommendations]
    
    async def initialize(self):
        """Initialize the Tool Discovery Agent."""
        await self.tool_registry.initialize()
        await self._build_tool_graph()
    
    async def close(self):
        """Close the Tool Discovery Agent."""
        await self.tool_registry.close()


# Example usage
if __name__ == "__main__":
    from src.agents.intent_recognition_agent import IntentRecognitionAgent, Intent
    
    async def test_tool_discovery():
        """Test the Tool Discovery Agent."""
        # Initialize agents
        intent_agent = IntentRecognitionAgent()
        discovery_agent = ToolDiscoveryAgent()
        
        await discovery_agent.initialize()
        
        # Test queries
        test_queries = [
            "Search for files in the project",
            "Query the database for user information",
            "Create a new configuration file"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            # Get intent
            intent_result = await intent_agent.process_query(query)
            print(f"Intent: {intent_result.primary_intent.type}")
            
            # Discover tools
            candidates = await discovery_agent.discover_tools(intent_result)
            
            print(f"\nDiscovered {len(candidates)} tools:")
            for candidate in candidates[:5]:  # Show top 5
                print(f"\n  Tool: {candidate.tool_name} ({candidate.tool_id})")
                print(f"  Type: {candidate.tool_type}")
                print(f"  Capabilities: {', '.join(candidate.capabilities[:3])}")
                print(f"  Scores:")
                print(f"    - Semantic: {candidate.semantic_score:.2f}")
                print(f"    - Capability: {candidate.capability_score:.2f}")
                print(f"    - Performance: {candidate.performance_score:.2f}")
                print(f"    - Overall: {candidate.overall_score:.2f}")
        
        await discovery_agent.close()
    
    # Run the test
    asyncio.run(test_tool_discovery())