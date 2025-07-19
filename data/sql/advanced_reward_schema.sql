-- Advanced Reward Strategies Database Schema
-- This schema extends the existing database with tables for tracking
-- advanced reward calculation strategies and their performance.

-- Strategy performance tracking table
CREATE TABLE IF NOT EXISTS reward_strategy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    reward_contribution REAL NOT NULL,
    computation_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional metadata
    components JSON,  -- Breakdown of reward components
    metadata JSON,    -- Strategy-specific metadata
    
    -- Indexes for performance
    INDEX idx_strategy_name ON reward_strategy_metrics(strategy_name),
    INDEX idx_execution_id ON reward_strategy_metrics(execution_id),
    INDEX idx_created_at ON reward_strategy_metrics(created_at DESC)
);

-- Goal hierarchy for hierarchical rewards
CREATE TABLE IF NOT EXISTS goal_hierarchy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL UNIQUE,
    goal_name TEXT NOT NULL,
    parent_goal_id INTEGER,
    goal_type TEXT NOT NULL CHECK (goal_type IN ('primary', 'secondary', 'tertiary', 'milestone', 'subtask')),
    weight REAL NOT NULL DEFAULT 1.0,
    required_tools JSON,  -- List of required tool IDs
    success_criteria JSON,  -- Criteria for goal achievement
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (parent_goal_id) REFERENCES goal_hierarchy(id) ON DELETE CASCADE,
    INDEX idx_goal_type ON goal_hierarchy(goal_type),
    INDEX idx_parent_goal ON goal_hierarchy(parent_goal_id)
);

-- Goal progress tracking
CREATE TABLE IF NOT EXISTS goal_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    progress REAL NOT NULL CHECK (progress >= 0 AND progress <= 1),
    achieved BOOLEAN DEFAULT FALSE,
    achieved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (goal_id) REFERENCES goal_hierarchy(goal_id),
    INDEX idx_execution_goal ON goal_progress(execution_id, goal_id),
    INDEX idx_achieved ON goal_progress(achieved)
);

-- Curiosity metrics for information-theoretic rewards
CREATE TABLE IF NOT EXISTS novelty_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_hash TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    novelty_score REAL NOT NULL,
    visit_count INTEGER DEFAULT 1,
    information_gain REAL,
    last_visited TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(state_hash, action_hash),
    INDEX idx_state_action ON novelty_metrics(state_hash, action_hash),
    INDEX idx_visit_count ON novelty_metrics(visit_count),
    INDEX idx_last_visited ON novelty_metrics(last_visited DESC)
);

-- Adaptive reward shaping metrics
CREATE TABLE IF NOT EXISTS adaptive_shaping_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    curriculum_stage INTEGER NOT NULL,
    component_weights JSON NOT NULL,  -- Current weights for each component
    performance_score REAL,
    adaptation_rate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_execution_shaping ON adaptive_shaping_metrics(execution_id),
    INDEX idx_stage ON adaptive_shaping_metrics(curriculum_stage)
);

-- Temporal difference tracking
CREATE TABLE IF NOT EXISTS temporal_reward_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    td_error REAL,
    n_step_return REAL,
    eligibility_trace REAL,
    lambda_param REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_execution_temporal ON temporal_reward_states(execution_id),
    INDEX idx_state_temporal ON temporal_reward_states(state_hash)
);

-- A/B testing results
CREATE TABLE IF NOT EXISTS reward_ab_test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_group TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    strategy_used TEXT NOT NULL,
    reward REAL NOT NULL,
    performance_score REAL,
    success_rate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_test_group ON reward_ab_test_results(test_group),
    INDEX idx_ab_execution ON reward_ab_test_results(execution_id)
);

-- Strategy ensemble configuration
CREATE TABLE IF NOT EXISTS strategy_ensemble_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_name TEXT NOT NULL UNIQUE,
    strategy_weights JSON NOT NULL,
    combination_method TEXT NOT NULL,
    active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_active_config ON strategy_ensemble_config(active)
);

-- Strategy performance summary
CREATE VIEW IF NOT EXISTS strategy_performance_summary AS
SELECT 
    strategy_name,
    COUNT(*) as execution_count,
    AVG(reward_contribution) as avg_reward,
    STD(reward_contribution) as reward_std,
    AVG(computation_time_ms) as avg_computation_time,
    MAX(created_at) as last_used
FROM reward_strategy_metrics
GROUP BY strategy_name;

-- Goal achievement summary
CREATE VIEW IF NOT EXISTS goal_achievement_summary AS
SELECT 
    gh.goal_id,
    gh.goal_name,
    gh.goal_type,
    COUNT(DISTINCT gp.execution_id) as attempt_count,
    SUM(CASE WHEN gp.achieved THEN 1 ELSE 0 END) as achievement_count,
    AVG(gp.progress) as avg_progress,
    CAST(SUM(CASE WHEN gp.achieved THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as achievement_rate
FROM goal_hierarchy gh
LEFT JOIN goal_progress gp ON gh.goal_id = gp.goal_id
GROUP BY gh.goal_id, gh.goal_name, gh.goal_type;

-- Novelty exploration summary
CREATE VIEW IF NOT EXISTS novelty_exploration_summary AS
SELECT 
    COUNT(DISTINCT state_hash) as unique_states,
    COUNT(DISTINCT action_hash) as unique_actions,
    COUNT(*) as total_state_actions,
    AVG(novelty_score) as avg_novelty,
    AVG(visit_count) as avg_visits,
    AVG(information_gain) as avg_info_gain
FROM novelty_metrics;

-- Triggers for updating timestamps
CREATE TRIGGER IF NOT EXISTS update_goal_hierarchy_timestamp
AFTER UPDATE ON goal_hierarchy
BEGIN
    UPDATE goal_hierarchy SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_strategy_config_timestamp
AFTER UPDATE ON strategy_ensemble_config
BEGIN
    UPDATE strategy_ensemble_config SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Sample data for default goals
INSERT OR IGNORE INTO goal_hierarchy (goal_id, goal_name, goal_type, weight) VALUES
    ('complete_task', 'Complete User Task', 'primary', 1.0),
    ('find_tools', 'Find Appropriate Tools', 'secondary', 0.5),
    ('execute_efficiently', 'Execute Efficiently', 'secondary', 0.5),
    ('minimize_errors', 'Minimize Execution Errors', 'tertiary', 0.25),
    ('optimize_resources', 'Optimize Resource Usage', 'tertiary', 0.25);

-- Update parent relationships
UPDATE goal_hierarchy SET parent_goal_id = (SELECT id FROM goal_hierarchy WHERE goal_id = 'complete_task')
WHERE goal_id IN ('find_tools', 'execute_efficiently');

UPDATE goal_hierarchy SET parent_goal_id = (SELECT id FROM goal_hierarchy WHERE goal_id = 'execute_efficiently')
WHERE goal_id IN ('minimize_errors', 'optimize_resources');