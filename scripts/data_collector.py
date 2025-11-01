"""
Comprehensive Football Data Collection Module
Collects match data, lineups, player stats, injuries, weather, and news
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from bs4 import BeautifulSoup

class ComprehensiveFootballCollector:
    def __init__(self, api_football_key, weather_api_key):
        """
        Initialize with API keys
        
        Args:
            api_football_key: RapidAPI key for API-Football
            weather_api_key: OpenWeatherMap API key
        """
        self.api_football_key = api_football_key
        self.weather_api_key = weather_api_key
        
        self.api_football_base = "https://v3.football.api-sports.io"
        self.api_football_headers = {
            "x-apisports-key": api_football_key
        }  
        self.weather_base = "http://api.openweathermap.org/data/2.5"
    
    # ==================== MATCH DATA ====================
    
    def get_fixtures_detailed(self, league_id, season):
        """
        Get detailed fixture data including results
        
        Args:
            league_id (int): League ID (39 = Premier League)
            season (int): Season year (e.g., 2023)
        """
        endpoint = f"{self.api_football_base}/fixtures"
        params = {"league": league_id, "season": season}
        
        response = requests.get(endpoint, headers=self.api_football_headers, params=params)
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            print(f"Error fetching fixtures: {response.status_code}")
            return None
    
    # ==================== LINEUP DATA ====================
    
    def get_fixture_lineups(self, fixture_id):
        """
        Get lineups for a specific fixture
        Returns starting XI and substitutes for both teams
        """
        endpoint = f"{self.api_football_base}/fixtures/lineups"
        params = {"fixture": fixture_id}
        
        response = requests.get(endpoint, headers=self.api_football_headers, params=params)
        
        if response.status_code == 200:
            data = response.json()['response']
            if data:
                return self._parse_lineups(data)
        
        return None
    
    def _parse_lineups(self, lineup_data):
        """Parse lineup data into structured format"""
        lineups = {}
        
        for team_data in lineup_data:
            team_name = team_data['team']['name']
            team_id = team_data['team']['id']
            formation = team_data['formation']
            
            starters = []
            for player in team_data['startXI']:
                starters.append({
                    'player_id': player['player']['id'],
                    'player_name': player['player']['name'],
                    'number': player['player']['number'],
                    'position': player['player']['pos'],
                    'grid': player['player']['grid']
                })
            
            substitutes = []
            for player in team_data['substitutes']:
                substitutes.append({
                    'player_id': player['player']['id'],
                    'player_name': player['player']['name'],
                    'number': player['player']['number'],
                    'position': player['player']['pos']
                })
            
            lineups[team_name] = {
                'team_id': team_id,
                'formation': formation,
                'starters': starters,
                'substitutes': substitutes
            }
        
        return lineups
    
    # ==================== PLAYER STATISTICS ====================
    
    def get_player_statistics(self, player_id, season):
        """
        Get detailed statistics for a specific player
        """
        endpoint = f"{self.api_football_base}/players"
        params = {"id": player_id, "season": season}
        
        response = requests.get(endpoint, headers=self.api_football_headers, params=params)
        
        if response.status_code == 200:
            data = response.json()['response']
            if data:
                return self._parse_player_stats(data[0])
        
        return None
    
    def _parse_player_stats(self, player_data):
        """Parse player statistics"""
        player_info = player_data['player']
        stats = player_data['statistics'][0] if player_data['statistics'] else {}
        
        return {
            'player_id': player_info['id'],
            'name': player_info['name'],
            'age': player_info['age'],
            'position': stats.get('games', {}).get('position'),
            'rating': stats.get('games', {}).get('rating'),
            'appearances': stats.get('games', {}).get('appearences', 0),
            'minutes_played': stats.get('games', {}).get('minutes', 0),
            'goals': stats.get('goals', {}).get('total', 0),
            'assists': stats.get('goals', {}).get('assists', 0),
            'shots_total': stats.get('shots', {}).get('total', 0),
            'shots_on_target': stats.get('shots', {}).get('on', 0),
            'passes_total': stats.get('passes', {}).get('total', 0),
            'passes_accuracy': stats.get('passes', {}).get('accuracy', 0),
            'tackles': stats.get('tackles', {}).get('total', 0),
            'duels_won': stats.get('duels', {}).get('won', 0),
            'yellow_cards': stats.get('cards', {}).get('yellow', 0),
            'red_cards': stats.get('cards', {}).get('red', 0)
        }
    
    # ==================== INJURIES & SUSPENSIONS ====================
    
    def get_team_injuries(self, team_id):
        """
        Get current injuries and suspensions for a team
        """
        endpoint = f"{self.api_football_base}/injuries"
        params = {"team": team_id}
        
        response = requests.get(endpoint, headers=self.api_football_headers, params=params)
        
        if response.status_code == 200:
            data = response.json()['response']
            return self._parse_injuries(data)
        
        return []
    
    def _parse_injuries(self, injury_data):
        """Parse injury data"""
        injuries = []
        
        for injury in injury_data:
            injuries.append({
                'player_id': injury['player']['id'],
                'player_name': injury['player']['name'],
                'player_photo': injury['player']['photo'],
                'injury_type': injury['player']['type'],
                'injury_reason': injury['player']['reason'],
                'fixture_id': injury['fixture']['id'],
                'league': injury['league']['name'],
                'season': injury['league']['season']
            })
        
        return injuries
    
    # ==================== WEATHER DATA ====================
    
    def get_historical_weather(self, stadium_lat, stadium_lon, match_datetime):
        """
        Get historical weather for a specific match
        Note: Requires historical weather API (paid) or approximate from current
        
        Args:
            stadium_lat (float): Stadium latitude
            stadium_lon (float): Stadium longitude
            match_datetime (datetime): Match date and time
        """
        # For historical data, you'd need OpenWeatherMap's paid tier
        # For now, we'll use a simplified approach
        
        endpoint = f"{self.weather_base}/weather"
        params = {
            "lat": stadium_lat,
            "lon": stadium_lon,
            "appid": self.weather_api_key,
            "units": "metric"
        }
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'weather_condition': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'clouds': data['clouds']['all'],
                'rain': data.get('rain', {}).get('1h', 0)
            }
        
        return None
    
    def get_forecast_weather(self, stadium_lat, stadium_lon):
        """Get weather forecast for upcoming match"""
        endpoint = f"{self.weather_base}/forecast"
        params = {
            "lat": stadium_lat,
            "lon": stadium_lon,
            "appid": self.weather_api_key,
            "units": "metric"
        }
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            return response.json()
        
        return None
    
    # ==================== STADIUM COORDINATES ====================
    
    def get_stadium_coordinates(self, team_id):
        """
        Get stadium coordinates for weather data
        Note: You might need to maintain a manual mapping
        """
        # This is a simplified version - you'd maintain a database
        stadium_coords = {
            # Premier League examples
            33: {"name": "Old Trafford", "lat": 53.4631, "lon": -2.2913},  # Man United
            34: {"name": "St. James' Park", "lat": 54.9756, "lon": -1.6217},  # Newcastle
            40: {"name": "Anfield", "lat": 53.4308, "lon": -2.9608},  # Liverpool
            42: {"name": "Emirates Stadium", "lat": 51.5549, "lon": -0.1084},  # Arsenal
            47: {"name": "Tottenham Hotspur Stadium", "lat": 51.6042, "lon": -0.0662},  # Spurs
            49: {"name": "Stamford Bridge", "lat": 51.4817, "lon": -0.1910},  # Chelsea
            50: {"name": "Etihad Stadium", "lat": 53.4831, "lon": -2.2004},  # Man City
            # Add more as needed
        }
        
        return stadium_coords.get(team_id)
    
    # ==================== NEWS SCRAPING ====================
    
    def scrape_team_news(self, team_name, days_back=7):
        """
        Scrape recent news about a team
        Note: This is a basic example - you'd need to respect robots.txt
        """
        # This is a placeholder - real implementation would scrape BBC Sport, etc.
        # For production, consider using News API instead
        
        news_articles = []
        
        # Example structure for what you'd scrape
        # You'd need to implement actual scraping logic
        
        return news_articles
    
    def get_news_sentiment(self, text):
        """
        Analyze sentiment of news text
        You'd use a library like TextBlob or transformers
        """
        # Placeholder - implement with TextBlob or VADER
        # from textblob import TextBlob
        # blob = TextBlob(text)
        # return blob.sentiment.polarity
        
        return 0  # Neutral
    
    # ==================== COMPREHENSIVE DATA COLLECTION ====================
    
    def collect_complete_match_data(self, fixture_id, league_id, season):
        """
        Collect ALL data for a specific match
        Returns a comprehensive dictionary with all match context
        """
        print(f"Collecting complete data for fixture {fixture_id}...")
        
        # 1. Get match basic info
        fixtures = self.get_fixtures_detailed(league_id, season)
        match_data = next((f for f in fixtures if f['fixture']['id'] == fixture_id), None)
        
        if not match_data:
            print("Match not found")
            return None
        
        complete_data = {
            'match_info': {
                'fixture_id': fixture_id,
                'date': match_data['fixture']['date'],
                'venue': match_data['fixture']['venue']['name'],
                'city': match_data['fixture']['venue']['city'],
                'home_team_id': match_data['teams']['home']['id'],
                'home_team_name': match_data['teams']['home']['name'],
                'away_team_id': match_data['teams']['away']['id'],
                'away_team_name': match_data['teams']['away']['name'],
                'home_goals': match_data['goals']['home'],
                'away_goals': match_data['goals']['away'],
                'status': match_data['fixture']['status']['short']
            }
        }
        
        # 2. Get lineups
        time.sleep(1)  # Rate limiting
        lineups = self.get_fixture_lineups(fixture_id)
        complete_data['lineups'] = lineups
        
        # 3. Get injuries for both teams
        time.sleep(1)
        home_injuries = self.get_team_injuries(complete_data['match_info']['home_team_id'])
        away_injuries = self.get_team_injuries(complete_data['match_info']['away_team_id'])
        complete_data['injuries'] = {
            'home': home_injuries,
            'away': away_injuries
        }
        
        # 4. Get weather data
        home_stadium = self.get_stadium_coordinates(complete_data['match_info']['home_team_id'])
        if home_stadium:
            weather = self.get_historical_weather(
                home_stadium['lat'],
                home_stadium['lon'],
                datetime.fromisoformat(match_data['fixture']['date'].replace('Z', '+00:00'))
            )
            complete_data['weather'] = weather
        
        # 5. Get player stats for lineup players (if lineups available)
        if lineups:
            complete_data['player_stats'] = {}
            for team, lineup in lineups.items():
                complete_data['player_stats'][team] = []
                for player in lineup['starters'][:3]:  # Limit to first 3 for API quota
                    time.sleep(1)
                    stats = self.get_player_statistics(player['player_id'], season)
                    if stats:
                        complete_data['player_stats'][team].append(stats)
        
        print("✓ Complete data collected")
        return complete_data
    
    def save_to_json(self, data, filename):
        """Save collected data to JSON"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved to {filename}")


# Example usage
if __name__ == "__main__":
    API_FOOTBALL_KEY = "YOUR_RAPIDAPI_KEY"
    WEATHER_API_KEY = "YOUR_OPENWEATHER_KEY"
    
    collector = ComprehensiveFootballCollector(API_FOOTBALL_KEY, WEATHER_API_KEY)
    
    # Example: Collect data for a specific match
    fixture_id = 1035116  # Example fixture ID
    league_id = 39  # Premier League
    season = 2023
    
    complete_data = collector.collect_complete_match_data(fixture_id, league_id, season)
    
    if complete_data:
        collector.save_to_json(complete_data, 'data/raw/match_complete_data.json')