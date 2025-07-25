-- PostgreSQL initialization script for MCP testing
-- This script creates sample tables and data for testing

-- Create sample tables
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(200) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT
);

-- Insert sample data
INSERT INTO users (username, email) VALUES 
    ('alice', 'alice@example.com'),
    ('bob', 'bob@example.com'),
    ('charlie', 'charlie@example.com')
ON CONFLICT DO NOTHING;

INSERT INTO categories (name, description) VALUES
    ('Technology', 'Tech-related posts'),
    ('Science', 'Scientific discoveries and research'),
    ('Arts', 'Creative and artistic content')
ON CONFLICT DO NOTHING;

INSERT INTO posts (user_id, title, content) VALUES
    (1, 'Getting Started with PostgreSQL', 'PostgreSQL is a powerful open-source database...'),
    (2, 'MCP Integration Guide', 'Model Context Protocol allows seamless integration...'),
    (3, 'Testing Best Practices', 'When testing database connections...')
ON CONFLICT DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at DESC);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO auto_tool_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO auto_tool_user;