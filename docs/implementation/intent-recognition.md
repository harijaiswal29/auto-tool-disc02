# Intent Recognition Architecture

## Overview

The Intent Recognition layer uses NLP techniques to understand user queries and classify them into actionable intents. It uses sentence-transformers for semantic understanding with a similarity threshold of 0.7.

## NLP Pipeline Components

```python
class IntentRecognitionPipeline:
    def __init__(self):
        self.components = [
            TextPreprocessor(),
            TokenizerModule(),
            FeatureExtractor(),
            IntentClassifier(),
            ContextEnricher()
        ]
    
    async def process(self, user_query: str, context: Context):
        result = {
            'raw_query': user_query,
            'context': context
        }
        
        for component in self.components:
            result = await component.process(result)
        
        return result
```

## Text Preprocessing

```python
class TextPreprocessor:
    def __init__(self):
        self.normalizers = [
            self.lowercase,
            self.expand_contractions,
            self.remove_special_chars,
            self.normalize_whitespace
        ]
    
    def process(self, text):
        for normalizer in self.normalizers:
            text = normalizer(text)
        return text
    
    def expand_contractions(self, text):
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            # ... more contractions
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text
```

## Feature Extraction Process

```python
class FeatureExtractor:
    def __init__(self):
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.keyword_extractor = KeywordExtractor()
        self.entity_recognizer = EntityRecognizer()
    
    def extract_features(self, text, context):
        features = {
            'semantic_embedding': self.sentence_encoder.encode(text),
            'keywords': self.keyword_extractor.extract(text),
            'entities': self.entity_recognizer.extract(text),
            'linguistic_features': self.extract_linguistic_features(text),
            'context_features': self.extract_context_features(context)
        }
        return features
    
    def extract_linguistic_features(self, text):
        return {
            'word_count': len(text.split()),
            'question_words': self.count_question_words(text),
            'imperative_indicators': self.detect_imperative(text),
            'sentiment': self.analyze_sentiment(text)
        }
```

## Intent Classification Taxonomy

```python
class IntentTaxonomy:
    INTENTS = {
        'query': {
            'search': ['find', 'search', 'look for', 'where is'],
            'retrieve': ['get', 'fetch', 'show', 'display'],
            'analyze': ['analyze', 'examine', 'investigate']
        },
        'action': {
            'create': ['create', 'make', 'generate', 'build'],
            'modify': ['update', 'change', 'edit', 'modify'],
            'delete': ['remove', 'delete', 'drop', 'clear']
        },
        'system': {
            'configure': ['setup', 'configure', 'settings'],
            'monitor': ['check', 'monitor', 'status', 'health']
        }
    }
    
    def classify(self, features):
        scores = {}
        for category, subcategories in self.INTENTS.items():
            for subcat, keywords in subcategories.items():
                score = self.calculate_match_score(features, keywords)
                scores[f"{category}.{subcat}"] = score
        
        return max(scores.items(), key=lambda x: x[1])
```

## Context Management System

```python
class ContextManager:
    def __init__(self, max_history=10):
        self.conversation_history = deque(maxlen=max_history)
        self.user_profile = {}
        self.session_state = {}
        self.domain_context = None
    
    def update_context(self, query, intent, result):
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'intent': intent,
            'result': result
        })
        
        # Update session state
        self.session_state['last_intent'] = intent
        self.session_state['last_tools'] = result.get('tools_used', [])
        
        # Update user profile
        self.update_user_profile(intent, result)
    
    def get_relevant_context(self, current_query):
        # Extract relevant history
        relevant_history = self.find_relevant_history(current_query)
        
        return {
            'history': relevant_history,
            'session': self.session_state,
            'user': self.user_profile,
            'domain': self.domain_context
        }
```

## Conversation State Machine

```python
class ConversationStateMachine:
    STATES = {
        'IDLE': ['QUERY_RECEIVED'],
        'QUERY_RECEIVED': ['INTENT_RECOGNIZED', 'CLARIFICATION_NEEDED'],
        'INTENT_RECOGNIZED': ['TOOLS_DISCOVERED', 'NO_TOOLS_FOUND'],
        'CLARIFICATION_NEEDED': ['CLARIFICATION_RECEIVED', 'TIMEOUT'],
        'TOOLS_DISCOVERED': ['EXECUTION_STARTED', 'USER_CANCELLED'],
        'EXECUTION_STARTED': ['EXECUTION_COMPLETE', 'EXECUTION_FAILED'],
        'EXECUTION_COMPLETE': ['FEEDBACK_RECEIVED', 'IDLE'],
        'EXECUTION_FAILED': ['RETRY_REQUESTED', 'IDLE']
    }
    
    def __init__(self):
        self.current_state = 'IDLE'
        self.state_history = []
        self.transition_handlers = {}
    
    def transition(self, new_state):
        if new_state not in self.STATES.get(self.current_state, []):
            raise ValueError(f"Invalid transition: {self.current_state} -> {new_state}")
        
        # Execute transition handler
        handler = self.transition_handlers.get((self.current_state, new_state))
        if handler:
            handler()
        
        # Update state
        self.state_history.append({
            'from': self.current_state,
            'to': new_state,
            'timestamp': datetime.now()
        })
        self.current_state = new_state
```

## Intent Confidence Scoring

```python
class IntentConfidenceScorer:
    def __init__(self):
        self.feature_weights = {
            'semantic_similarity': 0.4,
            'keyword_match': 0.3,
            'context_relevance': 0.2,
            'historical_accuracy': 0.1
        }
    
    def calculate_confidence(self, intent, features, context):
        scores = {
            'semantic_similarity': self.semantic_score(intent, features),
            'keyword_match': self.keyword_score(intent, features),
            'context_relevance': self.context_score(intent, context),
            'historical_accuracy': self.history_score(intent, context)
        }
        
        # Weighted average
        confidence = sum(
            self.feature_weights[key] * value 
            for key, value in scores.items()
        )
        
        return {
            'confidence': confidence,
            'scores': scores,
            'threshold_met': confidence > 0.7
        }
```

## Multi-Intent Handling

```python
class MultiIntentHandler:
    def __init__(self):
        self.intent_separator = IntentSeparator()
        self.dependency_resolver = DependencyResolver()
    
    def handle_compound_query(self, query, features):
        # Detect multiple intents
        intents = self.intent_separator.separate(query, features)
        
        if len(intents) == 1:
            return intents[0]
        
        # Resolve dependencies
        intent_graph = self.dependency_resolver.build_graph(intents)
        execution_order = self.dependency_resolver.topological_sort(intent_graph)
        
        return {
            'type': 'compound',
            'intents': intents,
            'execution_order': execution_order,
            'dependencies': intent_graph
        }
```

## Key Configuration

- **Model**: all-MiniLM-L6-v2 (sentence-transformers)
- **Similarity Threshold**: 0.7
- **Max History**: 10 conversations
- **Confidence Threshold**: 0.7

## Intent Categories

### Query Intents
- **search**: Find, search, look for, where is
- **retrieve**: Get, fetch, show, display
- **analyze**: Analyze, examine, investigate

### Action Intents
- **create**: Create, make, generate, build
- **modify**: Update, change, edit, modify
- **delete**: Remove, delete, drop, clear

### System Intents
- **configure**: Setup, configure, settings
- **monitor**: Check, monitor, status, health

## Best Practices

1. **Preprocessing**: Always normalize text before processing
2. **Context**: Maintain conversation history for better understanding
3. **Confidence**: Only proceed with high-confidence intents (>0.7)
4. **Multi-Intent**: Handle compound queries gracefully
5. **Fallback**: Have clear fallback strategies for unclear intents