"""
Intent Classification Pipeline Stage.

This module handles intent classification based on taxonomy and patterns
as part of the intent recognition pipeline.
"""

import re
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple

from src.pipeline.base import PipelineStage, PipelineData
from src.models.intent import Intent


class IntentClassifierStage(PipelineStage):
    """
    Pipeline stage for classifying intents based on taxonomy and patterns.
    
    This stage:
    - Performs keyword-based intent classification
    - Combines semantic scores with keyword matching
    - Generates ranked intent candidates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intent classifier stage."""
        super().__init__(name="IntentClassifier", config=config)
        
        # Intent taxonomy mapping keywords to intent types - ENHANCED
        self.intent_taxonomy = {
            'query': {
                'search': ['find', 'search', 'look for', 'where is', 'locate', 'discover', 
                          'browse', 'explore', 'scan', 'seek', 'lookup'],
                'retrieve': ['read', 'get', 'fetch', 'show', 'display', 'list', 'view',
                            'query', 'data', 'database', 'user', 'information', 'details',
                            'retrieve', 'access', 'load', 'pull', 'extract'],
                'analyze': ['what is the analysis', 'show analysis', 'get insights', 'view metrics',
                           'display statistics', 'retrieve report', 'fetch results', 'analysis report',
                           'analysis results', 'show me the analysis', 'what are the analysis',
                           'analysis of', 'the analysis', 'analyze']
            },
            'action': {
                'create': ['create', 'make', 'generate', 'build', 'add', 'new', 
                          'produce', 'construct', 'write', 'compose', 'establish'],
                'modify': ['update', 'change', 'edit', 'modify', 'alter', 'revise',
                          'adjust', 'transform', 'configure', 'set', 'patch'],
                'delete': ['remove', 'delete', 'drop', 'clear', 'erase', 'destroy',
                          'purge', 'eliminate', 'discard', 'clean', 'wipe'],
                'analyze': ['analyze', 'examine', 'investigate', 'inspect', 'evaluate', 'assess',
                           'process', 'compute', 'calculate', 'measure', 'study', 'review',
                           'interpret', 'diagnose', 'profile', 'benchmark'],
                'export': ['export', 'save', 'output', 'download', 'write to', 'dump',
                          'serialize', 'backup', 'csv', 'json', 'xml', 'pdf', 'excel']
            },
            'system': {
                'configure': ['setup', 'configure', 'settings', 'config', 'initialize',
                             'install', 'deploy', 'provision', 'prepare', 'customize'],
                'monitor': ['check', 'monitor', 'status', 'health', 'track', 'watch',
                           'observe', 'supervise', 'audit', 'log', 'report']
            }
        }
        
        # Allow custom taxonomy from config
        if config and 'intent_taxonomy' in config:
            self.intent_taxonomy.update(config['intent_taxonomy'])
        
        # Create reverse mapping for quick lookup
        self.keyword_to_intent = {}
        self._build_keyword_mapping()
        
        # Classification parameters
        self.keyword_weight = config.get('keyword_weight', 0.4) if config else 0.4
        self.semantic_weight = config.get('semantic_weight', 0.5) if config else 0.5
        self.context_weight = config.get('context_weight', 0.1) if config else 0.1
        self.history_boost = config.get('history_boost', 0.15) if config else 0.15
        self.domain_boost = config.get('domain_boost', 0.1) if config else 0.1
    
    async def _initialize(self):
        """No special initialization needed for classifier."""
        self.logger.debug("Intent classifier stage initialized")
    
    def _build_keyword_mapping(self):
        """Build reverse mapping from keywords to intent types."""
        self.keyword_to_intent = {}
        
        for category, subcategories in self.intent_taxonomy.items():
            for subcat, keywords in subcategories.items():
                intent_type = f"{category}.{subcat}"
                for keyword in keywords:
                    if keyword not in self.keyword_to_intent:
                        self.keyword_to_intent[keyword] = []
                    self.keyword_to_intent[keyword].append(intent_type)
    
    async def process(self, data: PipelineData) -> PipelineData:
        """
        Classify intents based on features and patterns.
        
        Args:
            data: Pipeline data containing tokens and features
            
        Returns:
            Pipeline data with classified intents
        """
        try:
            # Get tokens from tokenizer
            tokens = data.get_stage_result('Tokenizer', 'tokens', [])
            content_tokens = data.get_stage_result('Tokenizer', 'content_tokens', tokens)
            
            # Get semantic scores from feature extractor
            semantic_scores = data.get_stage_result('FeatureExtractor', 'semantic_scores', {})
            
            # Get normalized text
            normalized_text = data.get_stage_result('TextPreprocessor', 'normalized_text', '')
            
            # Get context information from pipeline data
            original_context = data.context or {}
            
            # Calculate context score based on available context
            context_score = self._calculate_context_score(original_context)
            
            # Prepare context info for scoring
            domain = original_context.get('domain', 'general')
            # Convert domain to dict if it's a string
            if isinstance(domain, str):
                domain = {'name': domain}
            
            context_info = {
                'history': original_context.get('history', []),
                'domain': domain,
                'user_profile': original_context.get('user_profile', {}),
                'session': original_context.get('session', {})
            }
            
            self.logger.debug(f"Classifying intent for tokens: {tokens}")
            self.logger.debug(f"Context score: {context_score}")
            
            # Extract keywords from tokens
            keywords = self.extract_keywords(content_tokens)
            
            # Get keyword-based classification scores
            keyword_scores = self.classify_by_keywords(content_tokens)
            
            # Apply context-based adjustments for specific scenarios
            try:
                domain = context_info.get('domain', {})
                domain_name = domain.get('name') if isinstance(domain, dict) else str(domain)
                
                if domain_name == 'engineering':
                    # In engineering context, "analyze the code" is typically a query
                    if 'analyze' in content_tokens and any(word in content_tokens for word in ['code', 'function', 'class']):
                        # Boost query.analyze in both keyword and semantic scores
                        if 'query.analyze' not in keyword_scores:
                            keyword_scores['query.analyze'] = 1.0
                        else:
                            keyword_scores['query.analyze'] *= 2.0
                            
                        # Also adjust semantic scores
                        if 'query.analyze' not in semantic_scores:
                            semantic_scores['query.analyze'] = 0.8
                        else:
                            semantic_scores['query.analyze'] *= 1.5
                            
                        # Reduce action.analyze scores
                        if 'action.analyze' in keyword_scores:
                            keyword_scores['action.analyze'] *= 0.3
                        if 'action.analyze' in semantic_scores:
                            semantic_scores['action.analyze'] *= 0.5
            except Exception as e:
                self.logger.debug(f"Error applying context adjustments: {e}")
            
            # Combine semantic, keyword, and context scores
            combined_scores = self.combine_scores(
                semantic_scores, 
                keyword_scores,
                context_score,
                context_info
            )
            
            # Create intent objects
            intents = self.create_intents(combined_scores, keywords)
            
            # Store classification results
            data.add_stage_result(self.name, 'keywords', keywords)
            data.add_stage_result(self.name, 'keyword_scores', keyword_scores)
            data.add_stage_result(self.name, 'combined_scores', combined_scores)
            data.add_stage_result(self.name, 'classified_intents', intents)
            
            # Add keywords to metadata
            data.add_metadata('keywords', keywords)
            
            self.logger.debug(f"Classified {len(intents)} intent candidates")
            
            return data
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in IntentClassifier process: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return data with empty results so pipeline can continue
            data.add_stage_result(self.name, 'keywords', [])
            data.add_stage_result(self.name, 'keyword_scores', {})
            data.add_stage_result(self.name, 'combined_scores', {})
            data.add_stage_result(self.name, 'classified_intents', [])
            
            return data
    
    def extract_keywords(self, tokens: List[str]) -> List[str]:
        """
        Extract relevant keywords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of identified keywords
        """
        keywords = []
        
        # Check each token against our keyword mapping
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.keyword_to_intent:
                keywords.append(token_lower)
            
            # Also check partial matches for compound words
            # e.g., "querying" should match "query"
            for keyword in self.keyword_to_intent:
                if len(keyword) >= 4 and keyword in token_lower and keyword not in keywords:
                    keywords.append(keyword)
        
        # Check multi-word keywords in the full text
        text = ' '.join(tokens).lower()
        for keyword in self.keyword_to_intent:
            if ' ' in keyword and keyword in text and keyword not in keywords:
                keywords.append(keyword)
        
        # Additional keyword extraction for common patterns
        # Extract words that are commonly used but might not be in taxonomy
        additional_keywords = []
        for token in tokens:
            token_lower = token.lower()
            # Check for database-related terms
            if any(term in token_lower for term in ['database', 'db', 'data', 'table', 'record']):
                if token_lower not in keywords:
                    additional_keywords.append(token_lower)
            # Check for file-related terms
            elif any(term in token_lower for term in ['file', 'document', 'folder', 'directory']):
                if token_lower not in keywords:
                    additional_keywords.append(token_lower)
            # Check for common action verbs that might be variations
            elif any(term in token_lower for term in ['querying', 'searching', 'fetching', 'getting']):
                # Extract the base form
                for base in ['query', 'search', 'fetch', 'get']:
                    if base in token_lower and base not in keywords:
                        keywords.append(base)
        
        # Combine all keywords
        all_keywords = list(set(keywords + additional_keywords))
        
        return all_keywords
    
    def _apply_context_rules(self, intent_scores: Dict[str, float], text: str, tokens: List[str]) -> Dict[str, float]:
        """
        Apply context-based rules to adjust intent scores.
        
        Args:
            intent_scores: Current intent scores
            text: Full text query
            tokens: Query tokens
            
        Returns:
            Adjusted intent scores
        """
        # Rule 1: If "analyze" appears with imperatives, boost action.analyze
        if any(word in ['analyze', 'examine', 'investigate', 'evaluate'] for word in tokens):
            # Check for imperative patterns
            imperative_patterns = [
                r'^analyze\s+',  # Starts with analyze
                r'^please\s+analyze',  # Polite imperative
                r'^can you analyze',  # Request form
                r'^i need to analyze',  # Need expression
                r'analyze this',  # Direct object
                r'analyze the',  # Direct object
            ]
            
            for pattern in imperative_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Boost action.analyze, reduce query.analyze
                    if 'action.analyze' in intent_scores:
                        intent_scores['action.analyze'] *= 2.0
                    if 'query.analyze' in intent_scores:
                        intent_scores['query.analyze'] *= 0.5
                    break
            
            # Check for query patterns
            query_patterns = [
                r'what is the analysis',
                r'show.*analysis',
                r'get.*analysis',
                r'view.*analysis',
                r'where.*analysis',
            ]
            
            for pattern in query_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Boost query.analyze, reduce action.analyze
                    if 'query.analyze' in intent_scores:
                        intent_scores['query.analyze'] *= 2.0
                    if 'action.analyze' in intent_scores:
                        intent_scores['action.analyze'] *= 0.5
                    break
        
        # Rule 2: Export patterns should strongly favor action.export
        if any(word in ['export', 'save', 'output', 'csv', 'json'] for word in tokens):
            if 'action.export' in intent_scores:
                intent_scores['action.export'] *= 2.5
        
        # Rule 3: Question words suggest query intents
        question_words = ['what', 'where', 'when', 'which', 'who', 'how']
        if any(word in tokens[:3] for word in question_words):  # Check first 3 tokens
            # Boost all query.* intents
            for intent_type in intent_scores:
                if intent_type.startswith('query.'):
                    intent_scores[intent_type] *= 1.5
                elif intent_type.startswith('action.'):
                    intent_scores[intent_type] *= 0.7
        
        # Rule 4: "analysis" with question context should be query.analyze
        if 'analysis' in text and any(word in text.split()[:5] for word in ['what', 'show', 'get', 'view', 'where']):
            if 'query.analyze' not in intent_scores:
                intent_scores['query.analyze'] = 1.0
            else:
                intent_scores['query.analyze'] *= 2.5
            # Reduce action.analyze if present
            if 'action.analyze' in intent_scores:
                intent_scores['action.analyze'] *= 0.3
        
        # Rule 5: Specific pattern "what is the analysis of X" should be query.analyze
        if re.search(r'what\s+(is|are)\s+the\s+analysis', text, re.IGNORECASE):
            if 'query.analyze' not in intent_scores:
                intent_scores['query.analyze'] = 2.0
            else:
                intent_scores['query.analyze'] *= 3.0
            # Remove other query types to ensure query.analyze wins
            for intent_type in list(intent_scores.keys()):
                if intent_type.startswith('query.') and intent_type != 'query.analyze':
                    intent_scores[intent_type] *= 0.2
        
        return intent_scores
    
    def classify_by_keywords(self, tokens: List[str]) -> Dict[str, float]:
        """
        Classify intents based on keyword matching.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary of intent types to scores
        """
        intent_scores = defaultdict(float)
        text = ' '.join(tokens).lower()
        
        # Score based on keyword matches
        for token in tokens:
            if token in self.keyword_to_intent:
                for intent_type in self.keyword_to_intent[token]:
                    intent_scores[intent_type] += 1.0
        
        # Check multi-word keywords
        for keyword, intent_types in self.keyword_to_intent.items():
            if ' ' in keyword and keyword in text:
                for intent_type in intent_types:
                    intent_scores[intent_type] += 1.5  # Higher weight for multi-word matches
        
        # Apply context-based adjustments
        intent_scores = self._apply_context_rules(intent_scores, text, tokens)
        
        # Normalize scores
        if intent_scores:
            max_score = max(intent_scores.values())
            for intent_type in intent_scores:
                intent_scores[intent_type] /= max_score
        
        return dict(intent_scores)
    
    def combine_scores(self, semantic_scores: Dict[str, float], 
                      keyword_scores: Dict[str, float],
                      context_score: float = 0.5,
                      context_info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Combine semantic, keyword, and context scores.
        
        Args:
            semantic_scores: Scores from semantic similarity
            keyword_scores: Scores from keyword matching
            context_score: Overall context relevance score
            context_info: Additional context information (history, domain, etc.)
            
        Returns:
            Combined scores for each intent type
        """
        combined_scores = {}
        
        # Get all possible intent types
        all_intent_types = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        for intent_type in all_intent_types:
            # Get individual scores (default to 0 if not present)
            semantic_score = semantic_scores.get(intent_type, 0.0)
            keyword_score = keyword_scores.get(intent_type, 0.0)
            
            # Ensure weights sum to 1.0 for proper normalization
            total_weight = self.semantic_weight + self.keyword_weight + self.context_weight
            if total_weight > 0:
                norm_semantic_weight = self.semantic_weight / total_weight
                norm_keyword_weight = self.keyword_weight / total_weight
                norm_context_weight = self.context_weight / total_weight
            else:
                norm_semantic_weight = 0.5
                norm_keyword_weight = 0.4
                norm_context_weight = 0.1
            
            # Weighted combination including context
            combined_score = (
                norm_semantic_weight * semantic_score +
                norm_keyword_weight * keyword_score +
                norm_context_weight * context_score
            )
            
            # Apply context-based boosts if context info is available
            if context_info:
                # History boost: check if this intent type appeared in recent history
                history = context_info.get('history', [])
                for hist_item in history[-3:]:  # Look at last 3 interactions
                    # Handle both string and dict history items
                    if isinstance(hist_item, dict):
                        if hist_item.get('intent_type') == intent_type:
                            combined_score += self.history_boost
                            break
                    elif isinstance(hist_item, str):
                        # For simple string history, check if intent type is mentioned
                        if intent_type in hist_item:
                            combined_score += self.history_boost * 0.5  # Reduced boost for string match
                            break
                
                # Domain boost: check if intent matches current domain
                domain = context_info.get('domain', {})
                domain_name = domain.get('name', '') if isinstance(domain, dict) else str(domain)
                if domain_name and domain_name != 'general':
                    # Check if domain name appears in intent type
                    if domain_name.lower() in intent_type.lower():
                        combined_score += self.domain_boost
            
            # Boost score if both methods agree
            if semantic_score > 0.5 and keyword_score > 0.5:
                combined_score *= 1.2  # 20% boost
            
            # Ensure score is between 0 and 1
            combined_scores[intent_type] = max(0.0, min(combined_score, 1.0))
        
        return combined_scores
    
    def create_intents(self, scores: Dict[str, float], keywords: List[str]) -> List[Intent]:
        """
        Create Intent objects from scores.
        
        Args:
            scores: Intent type scores
            keywords: Extracted keywords
            
        Returns:
            List of Intent objects
        """
        intents = []
        
        for intent_type, score in scores.items():
            intent = Intent(
                type=intent_type,
                confidence=score,
                keywords=keywords,
                entities=[],  # Entity extraction could be added later
                parameters={}
            )
            intents.append(intent)
        
        # Sort by confidence descending
        intents.sort(key=lambda x: x.confidence, reverse=True)
        
        return intents
    
    def _calculate_context_score(self, context: Dict[str, Any]) -> float:
        """
        Calculate context relevance score based on available context.
        
        Args:
            context: The context dictionary
            
        Returns:
            Context score between 0 and 1
        """
        score = 0.5  # Base score
        
        try:
            # Increase score based on available context elements
            if context.get('history'):
                score += 0.2
            
            domain = context.get('domain')
            if domain:
                # Handle both string and dict domains
                if isinstance(domain, str) and domain != 'general':
                    score += 0.15
                elif isinstance(domain, dict) and domain.get('name', 'general') != 'general':
                    score += 0.15
                    
            if context.get('user_profile'):
                score += 0.1
            if context.get('session'):
                score += 0.05
        except Exception as e:
            self.logger.error(f"Error calculating context score: {e}")
            
        return min(score, 1.0)
    
    async def validate_input(self, data: PipelineData) -> bool:
        """Validate that input contains required data."""
        # We need tokens from tokenizer
        if not data.get_stage_result('Tokenizer', 'tokens'):
            self.logger.error("No tokens found from tokenizer stage")
            return False
        
        return True