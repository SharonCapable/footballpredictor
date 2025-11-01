-- Football Prediction Database Schema
-- PostgreSQL 16

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Leagues
CREATE TABLE IF NOT EXISTS leagues (
    league_id INTEGER PRIMARY KEY,
    league_name VARCHAR(200) NOT NULL,
    country VARCHAR(100) NOT NULL,
    logo_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Teams
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_name VARCHAR(200) NOT NULL,
    country VARCHAR(100),
    founded INTEGER,
    stadium_name VARCHAR(200),
    stadium_capacity INTEGER,
    stadium_lat DECIMAL(10, 8),
    stadium_lon DECIMAL(11, 8),
    logo_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    player_name VARCHAR(200) NOT NULL,
    age INTEGER,
    nationality VARCHAR(100),
    height_cm INTEGER,
    weight_kg INTEGER,
    position VARCHAR(50),
    photo_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- MATCH DATA
-- =====================================================

-- Matches (Main table)
CREATE TABLE IF NOT EXISTS matches (
    match_id BIGINT PRIMARY KEY,
    fixture_date TIMESTAMP NOT NULL,
    league_id INTEGER REFERENCES leagues(league_id),
    season INTEGER NOT NULL,
    round VARCHAR(100),
    venue_name VARCHAR(200),
    venue_city VARCHAR(200),
    
    -- Teams
    home_team_id INTEGER REFERENCES teams(team_id),
    away_team_id INTEGER REFERENCES teams(team_id),
    
    -- Match Result
    home_goals INTEGER,
    away_goals INTEGER,
    home_halftime_goals INTEGER,
    away_halftime_goals INTEGER,
    match_status VARCHAR(20),
    
    -- Outcome Labels
    outcome CHAR(1), -- H/D/A
    outcome_numeric INTEGER, -- 1/0/2
    
    -- Formations
    home_formation VARCHAR(10),
    away_formation VARCHAR(10),
    
    -- Metadata
    data_collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_match_date ON matches(fixture_date);
CREATE INDEX IF NOT EXISTS idx_league_season ON matches(league_id, season);
CREATE INDEX IF NOT EXISTS idx_home_team ON matches(home_team_id);
CREATE INDEX IF NOT EXISTS idx_away_team ON matches(away_team_id);

-- =====================================================
-- LINEUPS & PLAYER PARTICIPATION
-- =====================================================

-- Match Lineups
CREATE TABLE IF NOT EXISTS match_lineups (
    lineup_id SERIAL PRIMARY KEY,
    match_id BIGINT REFERENCES matches(match_id),
    player_id INTEGER REFERENCES players(player_id),
    team_id INTEGER REFERENCES teams(team_id),
    
    -- Position Info
    position VARCHAR(50),
    shirt_number INTEGER,
    is_starter BOOLEAN DEFAULT TRUE,
    grid_position VARCHAR(10),
    
    -- Performance
    minutes_played INTEGER,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    rating DECIMAL(3, 1),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (match_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_match_lineup ON match_lineups(match_id);
CREATE INDEX IF NOT EXISTS idx_player_lineup ON match_lineups(player_id);

-- =====================================================
-- INJURIES & SUSPENSIONS
-- =====================================================

-- Injuries
CREATE TABLE IF NOT EXISTS injuries (
    injury_id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(player_id),
    team_id INTEGER REFERENCES teams(team_id),
    
    injury_type VARCHAR(100),
    injury_reason TEXT,
    severity VARCHAR(50),
    
    -- Timing
    injury_date DATE,
    expected_return_date DATE,
    actual_return_date DATE,
    
    -- Context
    related_match_id BIGINT REFERENCES matches(match_id),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_player_injury ON injuries(player_id);
CREATE INDEX IF NOT EXISTS idx_injury_date ON injuries(injury_date);

-- =====================================================
-- PLAYER STATISTICS (Season-level)
-- =====================================================

-- Player Season Stats
CREATE TABLE IF NOT EXISTS player_season_stats (
    stat_id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(player_id),
    team_id INTEGER REFERENCES teams(team_id),
    league_id INTEGER REFERENCES leagues(league_id),
    season INTEGER NOT NULL,
    
    -- Appearance
    appearances INTEGER DEFAULT 0,
    minutes_played INTEGER DEFAULT 0,
    starting_appearances INTEGER DEFAULT 0,
    
    -- Offensive
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    shots_total INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    
    -- Passing
    passes_total INTEGER DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    pass_accuracy DECIMAL(5, 2),
    key_passes INTEGER DEFAULT 0,
    
    -- Defensive
    tackles INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    blocks INTEGER DEFAULT 0,
    clearances INTEGER DEFAULT 0,
    
    -- Discipline
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    fouls_committed INTEGER DEFAULT 0,
    fouls_drawn INTEGER DEFAULT 0,
    
    -- Performance
    average_rating DECIMAL(3, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (player_id, team_id, league_id, season)
);

CREATE INDEX IF NOT EXISTS idx_player_season ON player_season_stats(player_id, season);

-- =====================================================
-- WEATHER DATA
-- =====================================================

-- Match Weather
CREATE TABLE IF NOT EXISTS match_weather (
    weather_id SERIAL PRIMARY KEY,
    match_id BIGINT REFERENCES matches(match_id) UNIQUE,
    
    temperature DECIMAL(5, 2),
    feels_like DECIMAL(5, 2),
    humidity INTEGER,
    
    weather_condition VARCHAR(50),
    weather_description TEXT,
    
    wind_speed DECIMAL(5, 2),
    wind_direction INTEGER,
    
    clouds INTEGER,
    precipitation DECIMAL(5, 2),
    
    pressure INTEGER,
    visibility INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_weather_match ON match_weather(match_id);

-- =====================================================
-- NEWS & SENTIMENT DATA
-- =====================================================

-- Player News
CREATE TABLE IF NOT EXISTS player_news (
    news_id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(player_id),
    team_id INTEGER REFERENCES teams(team_id),
    
    headline TEXT NOT NULL,
    article_text TEXT,
    source VARCHAR(200),
    source_url VARCHAR(500),
    
    published_date TIMESTAMP,
    scraped_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Sentiment Analysis
    sentiment_score DECIMAL(5, 4),
    sentiment_label VARCHAR(20),
    
    -- News Type
    news_category VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_player_news ON player_news(player_id);
CREATE INDEX IF NOT EXISTS idx_news_date ON player_news(published_date);

-- Team News
CREATE TABLE IF NOT EXISTS team_news (
    news_id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(team_id),
    
    headline TEXT NOT NULL,
    article_text TEXT,
    source VARCHAR(200),
    source_url VARCHAR(500),
    
    published_date TIMESTAMP,
    scraped_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    sentiment_score DECIMAL(5, 4),
    sentiment_label VARCHAR(20),
    news_category VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_team_news ON team_news(team_id);
CREATE INDEX IF NOT EXISTS idx_team_news_date ON team_news(published_date);

-- =====================================================
-- ENGINEERED FEATURES (For ML)
-- =====================================================

-- Pre-computed Team Form Features
CREATE TABLE IF NOT EXISTS team_form_features (
    feature_id SERIAL PRIMARY KEY,
    match_id BIGINT REFERENCES matches(match_id),
    team_id INTEGER REFERENCES teams(team_id),
    is_home_team BOOLEAN,
    
    -- Form (last N matches)
    form_last_5_points INTEGER,
    form_last_10_points INTEGER,
    
    -- Goals
    avg_goals_scored_last_5 DECIMAL(4, 2),
    avg_goals_conceded_last_5 DECIMAL(4, 2),
    avg_goals_scored_last_10 DECIMAL(4, 2),
    avg_goals_conceded_last_10 DECIMAL(4, 2),
    
    -- Win rates
    win_rate_last_5 DECIMAL(4, 3),
    win_rate_last_10 DECIMAL(4, 3),
    win_rate_home DECIMAL(4, 3),
    win_rate_away DECIMAL(4, 3),
    
    -- Streaks
    current_win_streak INTEGER,
    current_unbeaten_streak INTEGER,
    
    -- League position
    league_position INTEGER,
    league_points INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (match_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_form_match ON team_form_features(match_id);

-- Head-to-Head Features
CREATE TABLE IF NOT EXISTS h2h_features (
    h2h_id SERIAL PRIMARY KEY,
    match_id BIGINT REFERENCES matches(match_id) UNIQUE,
    home_team_id INTEGER REFERENCES teams(team_id),
    away_team_id INTEGER REFERENCES teams(team_id),
    
    -- Last N meetings
    h2h_matches_last_5 INTEGER,
    h2h_home_wins_last_5 INTEGER,
    h2h_draws_last_5 INTEGER,
    h2h_away_wins_last_5 INTEGER,
    
    h2h_home_goals_avg DECIMAL(4, 2),
    h2h_away_goals_avg DECIMAL(4, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_h2h_match ON h2h_features(match_id);

-- Player Impact Features
CREATE TABLE IF NOT EXISTS player_impact_features (
    impact_id SERIAL PRIMARY KEY,
    match_id BIGINT REFERENCES matches(match_id),
    team_id INTEGER REFERENCES teams(team_id),
    
    missing_key_players INTEGER,
    missing_defenders INTEGER,
    missing_midfielders INTEGER,
    missing_forwards INTEGER,
    
    avg_rating_of_missing DECIMAL(3, 2),
    avg_rating_of_lineup DECIMAL(3, 2),
    
    lineup_strength_score DECIMAL(5, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_impact_match ON player_impact_features(match_id);

-- =====================================================
-- MODEL TRACKING
-- =====================================================

-- Training Logs
CREATE TABLE IF NOT EXISTS model_training_logs (
    log_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    
    training_date TIMESTAMP NOT NULL,
    training_duration_seconds INTEGER,
    
    -- Dataset Info
    total_samples INTEGER,
    train_samples INTEGER,
    validation_samples INTEGER,
    test_samples INTEGER,
    
    -- Performance Metrics
    accuracy DECIMAL(6, 4),
    precision_macro DECIMAL(6, 4),
    recall_macro DECIMAL(6, 4),
    f1_score_macro DECIMAL(6, 4),
    
    -- Per-class metrics
    home_win_f1 DECIMAL(6, 4),
    draw_f1 DECIMAL(6, 4),
    away_win_f1 DECIMAL(6, 4),
    
    -- Hyperparameters
    hyperparameters JSONB,
    
    -- Feature Importance
    feature_importance JSONB,
    
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_version ON model_training_logs(model_version);
CREATE INDEX IF NOT EXISTS idx_training_date ON model_training_logs(training_date);

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    match_id BIGINT REFERENCES matches(match_id),
    model_version VARCHAR(50),
    
    prediction_date TIMESTAMP NOT NULL,
    
    -- Predictions
    predicted_outcome CHAR(1),
    predicted_outcome_numeric INTEGER,
    
    -- Probabilities
    prob_home_win DECIMAL(5, 4),
    prob_draw DECIMAL(5, 4),
    prob_away_win DECIMAL(5, 4),
    
    confidence_score DECIMAL(5, 4),
    
    -- Actual Outcome
    actual_outcome CHAR(1),
    is_correct BOOLEAN,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_model_pred ON predictions(model_version);

-- =====================================================
-- VIEWS FOR CONVENIENCE
-- =====================================================

-- Complete Match Data View
CREATE OR REPLACE VIEW vw_complete_matches AS
SELECT 
    m.*,
    l.league_name,
    l.country,
    ht.team_name as home_team_name,
    at.team_name as away_team_name,
    w.temperature,
    w.weather_condition,
    w.precipitation
FROM matches m
LEFT JOIN leagues l ON m.league_id = l.league_id
LEFT JOIN teams ht ON m.home_team_id = ht.team_id
LEFT JOIN teams at ON m.away_team_id = at.team_id
LEFT JOIN match_weather w ON m.match_id = w.match_id;

-- Model Performance Summary View
CREATE OR REPLACE VIEW vw_model_performance AS
SELECT 
    model_version,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_predictions,
    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100 as accuracy_percentage,
    AVG(confidence_score) as avg_confidence
FROM predictions
WHERE actual_outcome IS NOT NULL
GROUP BY model_version
ORDER BY accuracy_percentage DESC;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Database schema created successfully!';
    RAISE NOTICE 'Total tables created: 15';
    RAISE NOTICE 'Total views created: 2';
END $$;