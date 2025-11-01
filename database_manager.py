"""
PostgreSQL Database Manager for Football Prediction System
Handles all database operations with proper connection pooling
"""

import psycopg2
from psycopg2 import pool, extras
import pandas as pd
from datetime import datetime
import json
from contextlib import contextmanager

class FootballDatabaseManager:
    def __init__(self, host='localhost', port=5432, database='football_predictions', 
                 user='postgres', password='your_password'):
        """
        Initialize database connection pool
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            
            if self.connection_pool:
                print(f"✓ Connected to PostgreSQL database: {database}")
            
        except (Exception, psycopg2.Error) as error:
            print(f"Error connecting to PostgreSQL: {error}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, commit=False):
        """Context manager for database cursors"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()
    
    # ==================== INITIALIZATION ====================
    
    def execute_schema_file(self, schema_file_path):
        """Execute SQL schema file to create tables"""
        with open(schema_file_path, 'r') as f:
            schema_sql = f.read()
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(schema_sql)
        
        print("✓ Database schema created successfully")
    
    # ==================== LEAGUES ====================
    
    def insert_league(self, league_id, league_name, country, logo_url=None):
        """Insert or update league"""
        query = """
            INSERT INTO leagues (league_id, league_name, country, logo_url)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (league_id) 
            DO UPDATE SET league_name = EXCLUDED.league_name,
                         country = EXCLUDED.country,
                         logo_url = EXCLUDED.logo_url
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (league_id, league_name, country, logo_url))
    
    # ==================== TEAMS ====================
    
    def insert_team(self, team_data):
        """Insert or update team"""
        query = """
            INSERT INTO teams (team_id, team_name, country, founded, 
                             stadium_name, stadium_capacity, stadium_lat, stadium_lon, logo_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id)
            DO UPDATE SET team_name = EXCLUDED.team_name,
                         stadium_name = EXCLUDED.stadium_name,
                         stadium_lat = EXCLUDED.stadium_lat,
                         stadium_lon = EXCLUDED.stadium_lon
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                team_data['team_id'],
                team_data['team_name'],
                team_data.get('country'),
                team_data.get('founded'),
                team_data.get('stadium_name'),
                team_data.get('stadium_capacity'),
                team_data.get('stadium_lat'),
                team_data.get('stadium_lon'),
                team_data.get('logo_url')
            ))
    
    def get_stadium_coordinates(self, team_id):
        """Get stadium coordinates for weather lookup"""
        query = "SELECT stadium_lat, stadium_lon FROM teams WHERE team_id = %s"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (team_id,))
            result = cursor.fetchone()
            
            if result and result['stadium_lat'] and result['stadium_lon']:
                return {
                    'lat': float(result['stadium_lat']),
                    'lon': float(result['stadium_lon'])
                }
        
        return None
    
    # ==================== PLAYERS ====================
    
    def insert_player(self, player_data):
        """Insert or update player"""
        query = """
            INSERT INTO players (player_id, player_name, age, nationality, 
                               height_cm, weight_kg, position, photo_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id)
            DO UPDATE SET age = EXCLUDED.age,
                         position = EXCLUDED.position
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                player_data['player_id'],
                player_data['player_name'],
                player_data.get('age'),
                player_data.get('nationality'),
                player_data.get('height_cm'),
                player_data.get('weight_kg'),
                player_data.get('position'),
                player_data.get('photo_url')
            ))
    
    # ==================== MATCHES ====================
    
    def insert_match(self, match_data):
        """Insert or update match"""
        query = """
            INSERT INTO matches (
                match_id, fixture_date, league_id, season, round,
                venue_name, venue_city, home_team_id, away_team_id,
                home_goals, away_goals, home_halftime_goals, away_halftime_goals,
                match_status, outcome, outcome_numeric,
                home_formation, away_formation, data_collected_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (match_id)
            DO UPDATE SET 
                home_goals = EXCLUDED.home_goals,
                away_goals = EXCLUDED.away_goals,
                match_status = EXCLUDED.match_status,
                outcome = EXCLUDED.outcome,
                outcome_numeric = EXCLUDED.outcome_numeric,
                home_formation = EXCLUDED.home_formation,
                away_formation = EXCLUDED.away_formation,
                updated_at = CURRENT_TIMESTAMP
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                match_data['match_id'],
                match_data['fixture_date'],
                match_data['league_id'],
                match_data['season'],
                match_data.get('round'),
                match_data.get('venue_name'),
                match_data.get('venue_city'),
                match_data['home_team_id'],
                match_data['away_team_id'],
                match_data.get('home_goals'),
                match_data.get('away_goals'),
                match_data.get('home_halftime_goals'),
                match_data.get('away_halftime_goals'),
                match_data.get('match_status'),
                match_data.get('outcome'),
                match_data.get('outcome_numeric'),
                match_data.get('home_formation'),
                match_data.get('away_formation'),
                datetime.now()
            ))
    
    def get_matches_by_team(self, team_id, limit=None):
        """Get matches for a specific team"""
        query = """
            SELECT * FROM vw_complete_matches
            WHERE home_team_id = %s OR away_team_id = %s
            ORDER BY fixture_date DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (team_id, team_id))
            return pd.DataFrame(cursor.fetchall())
    
    def get_matches_by_season(self, league_id, season):
        """Get all matches for a season"""
        query = """
            SELECT * FROM vw_complete_matches
            WHERE league_id = %s AND season = %s
            ORDER BY fixture_date
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (league_id, season))
            return pd.DataFrame(cursor.fetchall())
    
    # ==================== LINEUPS ====================
    
    def insert_lineup(self, lineup_data):
        """Insert player lineup for a match"""
        query = """
            INSERT INTO match_lineups (
                match_id, player_id, team_id, position, shirt_number,
                is_starter, grid_position, minutes_played, goals, assists,
                yellow_cards, red_cards, rating
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (match_id, player_id) DO NOTHING
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                lineup_data['match_id'],
                lineup_data['player_id'],
                lineup_data['team_id'],
                lineup_data.get('position'),
                lineup_data.get('shirt_number'),
                lineup_data.get('is_starter', True),
                lineup_data.get('grid_position'),
                lineup_data.get('minutes_played'),
                lineup_data.get('goals', 0),
                lineup_data.get('assists', 0),
                lineup_data.get('yellow_cards', 0),
                lineup_data.get('red_cards', 0),
                lineup_data.get('rating')
            ))
    
    # ==================== INJURIES ====================
    
    def insert_injury(self, injury_data):
        """Insert injury/suspension record"""
        query = """
            INSERT INTO injuries (
                player_id, team_id, injury_type, injury_reason, severity,
                injury_date, expected_return_date, related_match_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING injury_id
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                injury_data['player_id'],
                injury_data['team_id'],
                injury_data.get('injury_type'),
                injury_data.get('injury_reason'),
                injury_data.get('severity'),
                injury_data.get('injury_date'),
                injury_data.get('expected_return_date'),
                injury_data.get('related_match_id')
            ))
            
            return cursor.fetchone()['injury_id']
    
    def get_active_injuries_for_match(self, team_id, match_date):
        """Get players injured/suspended at time of match"""
        query = """
            SELECT i.*, p.player_name, p.position
            FROM injuries i
            JOIN players p ON i.player_id = p.player_id
            WHERE i.team_id = %s
            AND i.injury_date <= %s
            AND (i.expected_return_date IS NULL OR i.expected_return_date >= %s)
            ORDER BY i.severity DESC
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (team_id, match_date, match_date))
            return pd.DataFrame(cursor.fetchall())
    
    # ==================== PLAYER STATS ====================
    
    def insert_player_season_stats(self, stats_data):
        """Insert player season statistics"""
        query = """
            INSERT INTO player_season_stats (
                player_id, team_id, league_id, season,
                appearances, minutes_played, starting_appearances,
                goals, assists, shots_total, shots_on_target,
                passes_total, passes_completed, pass_accuracy,
                tackles, interceptions, yellow_cards, red_cards,
                average_rating
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (player_id, team_id, league_id, season)
            DO UPDATE SET
                appearances = EXCLUDED.appearances,
                minutes_played = EXCLUDED.minutes_played,
                goals = EXCLUDED.goals,
                assists = EXCLUDED.assists,
                average_rating = EXCLUDED.average_rating,
                updated_at = CURRENT_TIMESTAMP
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                stats_data['player_id'],
                stats_data['team_id'],
                stats_data['league_id'],
                stats_data['season'],
                stats_data.get('appearances', 0),
                stats_data.get('minutes_played', 0),
                stats_data.get('starting_appearances', 0),
                stats_data.get('goals', 0),
                stats_data.get('assists', 0),
                stats_data.get('shots_total', 0),
                stats_data.get('shots_on_target', 0),
                stats_data.get('passes_total', 0),
                stats_data.get('passes_completed', 0),
                stats_data.get('pass_accuracy'),
                stats_data.get('tackles', 0),
                stats_data.get('interceptions', 0),
                stats_data.get('yellow_cards', 0),
                stats_data.get('red_cards', 0),
                stats_data.get('average_rating')
            ))
    
    # ==================== WEATHER ====================
    
    def insert_weather(self, weather_data):
        """Insert weather data for a match"""
        query = """
            INSERT INTO match_weather (
                match_id, temperature, feels_like, humidity,
                weather_condition, weather_description,
                wind_speed, wind_direction, clouds, precipitation,
                pressure, visibility
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (match_id)
            DO UPDATE SET
                temperature = EXCLUDED.temperature,
                weather_condition = EXCLUDED.weather_condition,
                precipitation = EXCLUDED.precipitation
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                weather_data['match_id'],
                weather_data.get('temperature'),
                weather_data.get('feels_like'),
                weather_data.get('humidity'),
                weather_data.get('weather_condition'),
                weather_data.get('weather_description'),
                weather_data.get('wind_speed'),
                weather_data.get('wind_direction'),
                weather_data.get('clouds'),
                weather_data.get('precipitation'),
                weather_data.get('pressure'),
                weather_data.get('visibility')
            ))
    
    # ==================== NEWS ====================
    
    def insert_player_news(self, news_data):
        """Insert player news article"""
        query = """
            INSERT INTO player_news (
                player_id, team_id, headline, article_text, source, source_url,
                published_date, sentiment_score, sentiment_label, news_category
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING news_id
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                news_data['player_id'],
                news_data.get('team_id'),
                news_data['headline'],
                news_data.get('article_text'),
                news_data.get('source'),
                news_data.get('source_url'),
                news_data.get('published_date'),
                news_data.get('sentiment_score'),
                news_data.get('sentiment_label'),
                news_data.get('news_category')
            ))
            
            return cursor.fetchone()['news_id']
    
    def get_recent_player_news(self, player_id, days_back=7):
        """Get recent news about a player"""
        query = """
            SELECT * FROM player_news
            WHERE player_id = %s
            AND published_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY published_date DESC
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (player_id, days_back))
            return pd.DataFrame(cursor.fetchall())
    
    # ==================== FEATURE ENGINEERING ====================
    
    def insert_team_form_features(self, features_data):
        """Insert pre-computed team form features"""
        query = """
            INSERT INTO team_form_features (
                match_id, team_id, is_home_team,
                form_last_5_points, form_last_10_points,
                avg_goals_scored_last_5, avg_goals_conceded_last_5,
                avg_goals_scored_last_10, avg_goals_conceded_last_10,
                win_rate_last_5, win_rate_last_10,
                win_rate_home, win_rate_away,
                current_win_streak, current_unbeaten_streak,
                league_position, league_points
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (match_id, team_id) DO NOTHING
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                features_data['match_id'],
                features_data['team_id'],
                features_data['is_home_team'],
                features_data.get('form_last_5_points'),
                features_data.get('form_last_10_points'),
                features_data.get('avg_goals_scored_last_5'),
                features_data.get('avg_goals_conceded_last_5'),
                features_data.get('avg_goals_scored_last_10'),
                features_data.get('avg_goals_conceded_last_10'),
                features_data.get('win_rate_last_5'),
                features_data.get('win_rate_last_10'),
                features_data.get('win_rate_home'),
                features_data.get('win_rate_away'),
                features_data.get('current_win_streak'),
                features_data.get('current_unbeaten_streak'),
                features_data.get('league_position'),
                features_data.get('league_points')
            ))
    
    def insert_h2h_features(self, h2h_data):
        """Insert head-to-head features"""
        query = """
            INSERT INTO h2h_features (
                match_id, home_team_id, away_team_id,
                h2h_matches_last_5, h2h_home_wins_last_5,
                h2h_draws_last_5, h2h_away_wins_last_5,
                h2h_home_goals_avg, h2h_away_goals_avg
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (match_id) DO NOTHING
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                h2h_data['match_id'],
                h2h_data['home_team_id'],
                h2h_data['away_team_id'],
                h2h_data.get('h2h_matches_last_5'),
                h2h_data.get('h2h_home_wins_last_5'),
                h2h_data.get('h2h_draws_last_5'),
                h2h_data.get('h2h_away_wins_last_5'),
                h2h_data.get('h2h_home_goals_avg'),
                h2h_data.get('h2h_away_goals_avg')
            ))
    
    # ==================== MODEL TRAINING ====================
    
    def get_training_dataset(self, start_date=None, end_date=None, leagues=None):
        """
        Get complete training dataset with all features
        
        Args:
            start_date: Filter matches from this date
            end_date: Filter matches until this date
            leagues: List of league IDs to include
        """
        query = """
            SELECT 
                m.match_id,
                m.fixture_date,
                m.league_id,
                m.season,
                m.outcome,
                m.outcome_numeric,
                
                -- Home team form
                htf.form_last_5_points as home_form_last_5,
                htf.avg_goals_scored_last_5 as home_goals_scored_avg_5,
                htf.avg_goals_conceded_last_5 as home_goals_conceded_avg_5,
                htf.win_rate_last_5 as home_win_rate_5,
                htf.win_rate_home as home_win_rate_home,
                htf.league_position as home_league_position,
                
                -- Away team form
                atf.form_last_5_points as away_form_last_5,
                atf.avg_goals_scored_last_5 as away_goals_scored_avg_5,
                atf.avg_goals_conceded_last_5 as away_goals_conceded_avg_5,
                atf.win_rate_last_5 as away_win_rate_5,
                atf.win_rate_away as away_win_rate_away,
                atf.league_position as away_league_position,
                
                -- Head-to-head
                h2h.h2h_home_wins_last_5,
                h2h.h2h_draws_last_5,
                h2h.h2h_away_wins_last_5,
                
                -- Weather
                w.temperature,
                w.precipitation,
                w.wind_speed,
                
                -- Player impact
                hpi.missing_key_players as home_missing_players,
                hpi.avg_rating_of_lineup as home_lineup_rating,
                api.missing_key_players as away_missing_players,
                api.avg_rating_of_lineup as away_lineup_rating
                
            FROM matches m
            LEFT JOIN team_form_features htf ON m.match_id = htf.match_id AND htf.is_home_team = true
            LEFT JOIN team_form_features atf ON m.match_id = atf.match_id AND atf.is_home_team = false
            LEFT JOIN h2h_features h2h ON m.match_id = h2h.match_id
            LEFT JOIN match_weather w ON m.match_id = w.match_id
            LEFT JOIN player_impact_features hpi ON m.match_id = hpi.match_id AND hpi.team_id = m.home_team_id
            LEFT JOIN player_impact_features api ON m.match_id = api.match_id AND api.team_id = m.away_team_id
            
            WHERE m.match_status = 'FT'
            AND m.outcome IS NOT NULL
        """
        
        conditions = []
        params = []
        
        if start_date:
            conditions.append("AND m.fixture_date >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("AND m.fixture_date <= %s")
            params.append(end_date)
        
        if leagues:
            placeholders = ','.join(['%s'] * len(leagues))
            conditions.append(f"AND m.league_id IN ({placeholders})")
            params.extend(leagues)
        
        query += ' '.join(conditions)
        query += " ORDER BY m.fixture_date"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return pd.DataFrame(cursor.fetchall())
    
    # ==================== MODEL TRACKING ====================
    
    def log_training(self, log_data):
        """Log model training session"""
        query = """
            INSERT INTO model_training_logs (
                model_name, model_version, training_date, training_duration_seconds,
                total_samples, train_samples, validation_samples, test_samples,
                accuracy, precision_macro, recall_macro, f1_score_macro,
                home_win_f1, draw_f1, away_win_f1,
                hyperparameters, feature_importance, notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING log_id
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                log_data['model_name'],
                log_data['model_version'],
                log_data['training_date'],
                log_data.get('training_duration_seconds'),
                log_data['total_samples'],
                log_data['train_samples'],
                log_data['validation_samples'],
                log_data['test_samples'],
                log_data['accuracy'],
                log_data.get('precision_macro'),
                log_data.get('recall_macro'),
                log_data['f1_score_macro'],
                log_data.get('home_win_f1'),
                log_data.get('draw_f1'),
                log_data.get('away_win_f1'),
                json.dumps(log_data.get('hyperparameters')),
                json.dumps(log_data.get('feature_importance')),
                log_data.get('notes')
            ))
            
            return cursor.fetchone()['log_id']
    
    def save_prediction(self, pred_data):
        """Save model prediction"""
        query = """
            INSERT INTO predictions (
                match_id, model_version, prediction_date,
                predicted_outcome, predicted_outcome_numeric,
                prob_home_win, prob_draw, prob_away_win,
                confidence_score, actual_outcome, is_correct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING prediction_id
        """
        
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, (
                pred_data['match_id'],
                pred_data['model_version'],
                pred_data['prediction_date'],
                pred_data['predicted_outcome'],
                pred_data['predicted_outcome_numeric'],
                pred_data['prob_home_win'],
                pred_data['prob_draw'],
                pred_data['prob_away_win'],
                pred_data['confidence_score'],
                pred_data.get('actual_outcome'),
                pred_data.get('is_correct')
            ))
            
            return cursor.fetchone()['prediction_id']
    
    def get_model_performance(self, model_version=None):
        """Get model performance metrics"""
        if model_version:
            query = "SELECT * FROM vw_model_performance WHERE model_version = %s"
            params = (model_version,)
        else:
            query = "SELECT * FROM vw_model_performance"
            params = None
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return pd.DataFrame(cursor.fetchall())
    
    # ==================== UTILITIES ====================
    
    def get_database_stats(self):
        """Get database statistics"""
        stats = {}
        
        queries = {
            'total_matches': "SELECT COUNT(*) as count FROM matches",
            'total_players': "SELECT COUNT(*) as count FROM players",
            'total_teams': "SELECT COUNT(*) as count FROM teams",
            'total_leagues': "SELECT COUNT(*) as count FROM leagues",
            'finished_matches': "SELECT COUNT(*) as count FROM matches WHERE match_status = 'FT'",
            'total_predictions': "SELECT COUNT(*) as count FROM predictions",
            'training_runs': "SELECT COUNT(*) as count FROM model_training_logs"
        }
        
        with self.get_cursor() as cursor:
            for key, query in queries.items():
                cursor.execute(query)
                stats[key] = cursor.fetchone()['count']
        
        return stats
    
    def close(self):
        """Close all connections in pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            print("✓ Database connections closed")


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = FootballDatabaseManager(
        host='localhost',
        database='football_predictions',
        user='postgres',
        password='your_password'
    )
    
    # Get database stats
    stats = db.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Close connections
    db.close()