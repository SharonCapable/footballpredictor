"""
Complete Data Pipeline Orchestrator
Coordinates data collection, cleaning, and storage
"""

import time
from datetime import datetime
import json
import pandas as pd
from data_collector import ComprehensiveFootballCollector
from news_scraper import FootballNewsScraper
from database_manager import FootballDatabaseManager

class DataPipeline:
    def __init__(self, api_football_key, weather_api_key, db_config):
        """
        Initialize complete pipeline
        
        Args:
            api_football_key: RapidAPI key for API-Football
            weather_api_key: OpenWeatherMap API key
            db_config: Dict with database connection info
        """
        # Initialize components
        self.collector = ComprehensiveFootballCollector(api_football_key, weather_api_key)
        self.news_scraper = FootballNewsScraper(rate_limit_delay=2)
        self.db = FootballDatabaseManager(**db_config)
        
        # League configurations
        self.leagues = {
            'Premier League': {'id': 39, 'country': 'England'},
            'Super Liga': {'id': 286, 'country': 'Serbia'},
            'Süper Lig': {'id': 203, 'country': 'Turkey'},
            'Pro League': {'id': 144, 'country': 'Belgium'},
            'Superliga': {'id': 119, 'country': 'Denmark'},
            'Primeira Liga': {'id': 94, 'country': 'Portugal'},
            'Premiership': {'id': 179, 'country': 'Scotland'},
            'Eredivisie': {'id': 88, 'country': 'Netherlands'}
        }
        
        # Stadium coordinates database (you'll need to expand this)
        self.stadium_coords = self._load_stadium_coordinates()
    
    def _load_stadium_coordinates(self):
        """Load/create stadium coordinates mapping"""
        # This is a starter - you'd expand this with all teams
        return {
            # Premier League
            33: {"lat": 53.4631, "lon": -2.2913},  # Man United - Old Trafford
            40: {"lat": 53.4308, "lon": -2.9608},  # Liverpool - Anfield
            42: {"lat": 51.5549, "lon": -0.1084},  # Arsenal - Emirates
            47: {"lat": 51.6042, "lon": -0.0662},  # Spurs - Tottenham Stadium
            49: {"lat": 51.4817, "lon": -0.1910},  # Chelsea - Stamford Bridge
            50: {"lat": 53.4831, "lon": -2.2004},  # Man City - Etihad
            # Add more teams as needed...
        }
    
    # ==================== STEP 1: INITIAL DATA COLLECTION ====================
    
    def collect_historical_data(self, seasons=[2021, 2022, 2023, 2024], 
                                save_progress=True):
        """
        Collect historical match data for all leagues and seasons
        
        Args:
            seasons: List of seasons to collect
            save_progress: Save progress after each league
        """
        print("=" * 60)
        print("STARTING HISTORICAL DATA COLLECTION")
        print("=" * 60)
        
        total_matches_collected = 0
        
        for league_name, league_info in self.leagues.items():
            league_id = league_info['id']
            country = league_info['country']
            
            print(f"\n{'='*60}")
            print(f"League: {league_name} ({country})")
            print(f"{'='*60}")
            
            # Insert league into database
            self.db.insert_league(
                league_id=league_id,
                league_name=league_name,
                country=country
            )
            
            for season in seasons:
                print(f"\n→ Collecting season {season}...")
                
                try:
                    # Get fixtures for this season
                    fixtures = self.collector.get_fixtures_detailed(league_id, season)
                    
                    if not fixtures:
                        print(f"  ✗ No fixtures found for {season}")
                        continue
                    
                    # Filter only finished matches
                    finished_matches = [
                        f for f in fixtures 
                        if f['fixture']['status']['short'] == 'FT'
                    ]
                    
                    print(f"  Found {len(finished_matches)} finished matches")
                    
                    # Process each match
                    for idx, fixture in enumerate(finished_matches, 1):
                        if idx % 10 == 0:
                            print(f"  Processing match {idx}/{len(finished_matches)}...")
                        
                        try:
                            self._process_and_store_match(fixture, league_id, season)
                            total_matches_collected += 1
                            
                        except Exception as e:
                            print(f"  ✗ Error processing match {fixture['fixture']['id']}: {e}")
                            continue
                        
                        # Rate limiting (100 requests/day = ~4 per hour if running 24hrs)
                        # Be conservative
                        time.sleep(1)  # 1 second between matches
                    
                    print(f"  ✓ Season {season} complete")
                    
                except Exception as e:
                    print(f"  ✗ Error collecting season {season}: {e}")
                    continue
                
                # Longer delay between seasons
                time.sleep(5)
            
            print(f"\n✓ {league_name} complete")
            
            if save_progress:
                print(f"Progress saved. Total matches: {total_matches_collected}")
            
            # Delay between leagues (respect API limits)
            time.sleep(10)
        
        print(f"\n{'='*60}")
        print(f"DATA COLLECTION COMPLETE")
        print(f"Total matches collected: {total_matches_collected}")
        print(f"{'='*60}")
        
        return total_matches_collected
    
    def _process_and_store_match(self, fixture, league_id, season):
        """Process a single match and store all related data"""
        match_id = fixture['fixture']['id']
        
        # 1. Extract and store basic match data
        match_data = self._extract_match_data(fixture, league_id, season)
        self.db.insert_match(match_data)
        
        # 2. Store team data
        self._store_team_data(fixture)
        
        # 3. Get and store lineups (if available)
        time.sleep(1)
        self._store_lineup_data(match_id, fixture)
        
        # 4. Get and store weather data
        self._store_weather_data(match_id, fixture)
    
    def _extract_match_data(self, fixture, league_id, season):
        """Extract match data from API response"""
        goals_home = fixture['goals']['home']
        goals_away = fixture['goals']['away']
        
        # Determine outcome
        if goals_home is not None and goals_away is not None:
            if goals_home > goals_away:
                outcome = 'H'
                outcome_numeric = 1
            elif goals_home < goals_away:
                outcome = 'A'
                outcome_numeric = 2
            else:
                outcome = 'D'
                outcome_numeric = 0
        else:
            outcome = None
            outcome_numeric = None
        
        return {
            'match_id': fixture['fixture']['id'],
            'fixture_date': fixture['fixture']['date'],
            'league_id': league_id,
            'season': season,
            'round': fixture['league']['round'],
            'venue_name': fixture['fixture']['venue']['name'],
            'venue_city': fixture['fixture']['venue']['city'],
            'home_team_id': fixture['teams']['home']['id'],
            'away_team_id': fixture['teams']['away']['id'],
            'home_goals': goals_home,
            'away_goals': goals_away,
            'home_halftime_goals': fixture['score']['halftime']['home'],
            'away_halftime_goals': fixture['score']['halftime']['away'],
            'match_status': fixture['fixture']['status']['short'],
            'outcome': outcome,
            'outcome_numeric': outcome_numeric
        }
    
    def _store_team_data(self, fixture):
        """Store team information"""
        for team_type in ['home', 'away']:
            team = fixture['teams'][team_type]
            
            team_data = {
                'team_id': team['id'],
                'team_name': team['name'],
                'logo_url': team['logo']
            }
            
            self.db.insert_team(team_data)
    
    def _store_lineup_data(self, match_id, fixture):
        """Get and store lineup data"""
        try:
            lineups = self.collector.get_fixture_lineups(match_id)
            
            if not lineups:
                return
            
            for team_name, lineup_info in lineups.items():
                team_id = lineup_info['team_id']
                formation = lineup_info['formation']
                
                # Update match with formation
                # (You'd need to add an update method or handle this in insert_match)
                
                # Store each player in lineup
                for player in lineup_info['starters']:
                    # Store player info
                    player_data = {
                        'player_id': player['player_id'],
                        'player_name': player['player_name'],
                        'position': player['position']
                    }
                    self.db.insert_player(player_data)
                    
                    # Store lineup participation
                    lineup_data = {
                        'match_id': match_id,
                        'player_id': player['player_id'],
                        'team_id': team_id,
                        'position': player['position'],
                        'shirt_number': player['number'],
                        'is_starter': True,
                        'grid_position': player.get('grid')
                    }
                    self.db.insert_lineup(lineup_data)
                
        except Exception as e:
            print(f"    Warning: Could not get lineups for match {match_id}: {e}")
    
    def _store_weather_data(self, match_id, fixture):
        """Get and store weather data"""
        try:
            home_team_id = fixture['teams']['home']['id']
            
            # Get stadium coordinates
            coords = self.stadium_coords.get(home_team_id)
            
            if coords:
                match_datetime = datetime.fromisoformat(
                    fixture['fixture']['date'].replace('Z', '+00:00')
                )
                
                weather = self.collector.get_historical_weather(
                    coords['lat'], 
                    coords['lon'],
                    match_datetime
                )
                
                if weather:
                    weather['match_id'] = match_id
                    self.db.insert_weather(weather)
            
        except Exception as e:
            print(f"    Warning: Could not get weather for match {match_id}: {e}")
    
    # ==================== STEP 2: COLLECT NEWS DATA ====================
    
    def collect_team_news(self, days_back=7):
        """
        Collect recent news for all teams in database
        
        Args:
            days_back: Collect news from last N days
        """
        print("\n" + "="*60)
        print("COLLECTING TEAM NEWS")
        print("="*60)
        
        # Get all teams from database
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT team_id, team_name FROM teams")
            teams = cursor.fetchall()
        
        print(f"Found {len(teams)} teams to scrape news for")
        
        total_articles = 0
        
        for team in teams[:5]:  # Limit to first 5 for testing
            team_id = team['team_id']
            team_name = team['team_name']
            
            print(f"\n→ Scraping news for {team_name}...")
            
            try:
                articles = self.news_scraper.scrape_comprehensive_team_news(
                    team_name, 
                    days_back=days_back
                )
                
                # Store in database
                for article in articles:
                    news_data = {
                        'team_id': team_id,
                        'headline': article['headline'],
                        'source': article['source'],
                        'source_url': article.get('source_url'),
                        'published_date': article.get('published_date'),
                        'sentiment_score': article['sentiment_score'],
                        'sentiment_label': article['sentiment_label'],
                        'news_category': article['news_category']
                    }
                    
                    # Note: You'd need to add insert_team_news method to database_manager
                    # self.db.insert_team_news(news_data)
                
                total_articles += len(articles)
                print(f"  ✓ Collected {len(articles)} articles")
                
                # Rate limiting
                time.sleep(5)
                
            except Exception as e:
                print(f"  ✗ Error scraping news for {team_name}: {e}")
                continue
        
        print(f"\n✓ News collection complete. Total articles: {total_articles}")
        
        return total_articles
    
    # ==================== STEP 3: FEATURE ENGINEERING ====================
    
    def compute_all_features(self):
        """
        Compute all engineered features for matches in database
        This should be run after initial data collection
        """
        print("\n" + "="*60)
        print("COMPUTING FEATURES")
        print("="*60)
        
        # Get all matches from database
        matches_df = self.db.get_training_dataset()
        
        print(f"Computing features for {len(matches_df)} matches...")
        
        # You would implement feature computation here
        # For now, placeholder
        print("Feature computation would happen here...")
        print("(Implement in next phase)")
    
    # ==================== PIPELINE EXECUTION ====================
    
    def run_full_pipeline(self, seasons=[2021, 2022, 2023, 2024]):
        """
        Run the complete data collection and processing pipeline
        
        Args:
            seasons: List of seasons to collect
        """
        print("\n" + "="*60)
        print("FOOTBALL PREDICTION PIPELINE")
        print("Starting full data collection process...")
        print("="*60)
        
        start_time = datetime.now()
        
        # Step 1: Collect historical match data
        print("\n[STEP 1/4] Collecting historical match data...")
        matches_collected = self.collect_historical_data(seasons=seasons)
        
        # Step 2: Collect news data
        print("\n[STEP 2/4] Collecting news data...")
        # Uncomment when ready:
        # articles_collected = self.collect_team_news(days_back=7)
        
        # Step 3: Compute features
        print("\n[STEP 3/4] Computing features...")
        # Uncomment when feature engineering is implemented:
        # self.compute_all_features()
        
        # Step 4: Generate summary report
        print("\n[STEP 4/4] Generating summary report...")
        self.generate_report()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print(f"Duration: {duration}")
        print("="*60)
    
    def generate_report(self):
        """Generate summary report of collected data"""
        stats = self.db.get_database_stats()
        
        print("\n" + "-"*60)
        print("DATABASE SUMMARY")
        print("-"*60)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("-"*60)
    
    def close(self):
        """Close all connections"""
        self.db.close()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Configuration
    API_FOOTBALL_KEY = "YOUR_RAPIDAPI_KEY_HERE"
    WEATHER_API_KEY = "YOUR_OPENWEATHER_KEY_HERE"
    
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'football_predictions',
        'user': 'postgres',
        'password': 'your_password'
    }
    
    # Initialize pipeline
    pipeline = DataPipeline(
        api_football_key=API_FOOTBALL_KEY,
        weather_api_key=WEATHER_API_KEY,
        db_config=DB_CONFIG
    )
    
    # Run full pipeline for last 4 seasons
    pipeline.run_full_pipeline(seasons=[2021, 2022, 2023, 2024])
    
    # Close connections
    pipeline.close()