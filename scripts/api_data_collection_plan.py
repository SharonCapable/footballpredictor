"""
Strategic API Data Collection Plan
Maximize impact with limited API calls (100/day)
"""

import os
import sys
sys.path.append('..')
from dotenv import load_dotenv
from data_collector import ComprehensiveFootballCollector
from database_manager import FootballDatabaseManager
import time
from datetime import datetime

load_dotenv()

class StrategicAPICollector:
    def __init__(self):
        self.collector = ComprehensiveFootballCollector(
            api_football_key=os.getenv('API_FOOTBALL_KEY'),
            weather_api_key=os.getenv('WEATHER_API_KEY')
        )
        
        self.db = FootballDatabaseManager(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT')),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
    
    # ==================== PHASE 1: LINEUPS (HIGHEST PRIORITY) ====================
    
    def collect_lineups_batch(self, num_matches=80):
        """
        Day 1-5: Collect lineups for most recent matches
        This is the MOST important data
        
        Uses: 1 API call per match
        """
        print("="*60)
        print("PHASE 1: COLLECTING LINEUPS")
        print("="*60)
        
        # Get most recent finished matches without lineups
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT m.match_id, m.fixture_date, m.league_id, m.season
                FROM matches m
                LEFT JOIN match_lineups ml ON m.match_id = ml.match_id
                WHERE m.match_status = 'FT'
                AND ml.lineup_id IS NULL
                ORDER BY m.fixture_date DESC
                LIMIT %s
            """, (num_matches,))
            
            matches_to_process = cursor.fetchall()
        
        print(f"\nFound {len(matches_to_process)} matches without lineup data")
        print(f"This will use ~{len(matches_to_process)} API calls")
        
        proceed = input("\nProceed? (yes/no): ")
        if proceed.lower() != 'yes':
            return
        
        collected = 0
        failed = 0
        
        for idx, match in enumerate(matches_to_process, 1):
            match_id = match['match_id']
            
            print(f"\n[{idx}/{len(matches_to_process)}] Processing match {match_id}...")
            
            try:
                # Get lineups
                lineups = self.collector.get_fixture_lineups(match_id)
                
                if lineups:
                    # Store in database
                    for team_name, lineup_info in lineups.items():
                        team_id = lineup_info['team_id']
                        
                        # Store each player in starting XI
                        for player in lineup_info['starters']:
                            # Insert player first
                            self.db.insert_player({
                                'player_id': player['player_id'],
                                'player_name': player['player_name'],
                                'position': player['position']
                            })
                            
                            # Insert lineup entry
                            self.db.insert_lineup({
                                'match_id': match_id,
                                'player_id': player['player_id'],
                                'team_id': team_id,
                                'position': player['position'],
                                'shirt_number': player['number'],
                                'is_starter': True,
                                'grid_position': player.get('grid')
                            })
                    
                    collected += 1
                    print(f"  ✓ Saved lineup data")
                else:
                    failed += 1
                    print(f"  ✗ No lineup data available")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                failed += 1
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"PHASE 1 COMPLETE")
        print(f"  Collected: {collected}")
        print(f"  Failed: {failed}")
        print(f"  API calls used: ~{collected + failed}")
        print(f"{'='*60}")
    
    # ==================== PHASE 2: INJURIES (HIGH PRIORITY) ====================
    
    def collect_current_injuries(self):
        """
        Day 6: Collect current injuries for all teams
        
        Uses: ~20 API calls (one per team, batched smartly)
        """
        print("\n" + "="*60)
        print("PHASE 2: COLLECTING INJURY DATA")
        print("="*60)
        
        # Get all unique teams
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT team_id, team_name 
                FROM teams 
                ORDER BY team_name
            """)
            
            teams = cursor.fetchall()
        
        print(f"\nFound {len(teams)} teams")
        print(f"This will use ~{len(teams)} API calls")
        
        proceed = input("\nProceed? (yes/no): ")
        if proceed.lower() != 'yes':
            return
        
        collected = 0
        
        for idx, team in enumerate(teams, 1):
            team_id = team['team_id']
            team_name = team['team_name']
            
            print(f"\n[{idx}/{len(teams)}] {team_name}...")
            
            try:
                injuries = self.collector.get_team_injuries(team_id)
                
                if injuries:
                    for injury in injuries:
                        # Insert player if not exists
                        self.db.insert_player({
                            'player_id': injury['player_id'],
                            'player_name': injury['player_name']
                        })
                        
                        # Insert injury record
                        self.db.insert_injury({
                            'player_id': injury['player_id'],
                            'team_id': team_id,
                            'injury_type': injury['injury_type'],
                            'injury_reason': injury['injury_reason'],
                            'injury_date': datetime.now().date()
                        })
                    
                    print(f"  ✓ Found {len(injuries)} injuries/suspensions")
                    collected += len(injuries)
                else:
                    print(f"  ✓ No injuries")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"PHASE 2 COMPLETE")
        print(f"  Total injuries collected: {collected}")
        print(f"{'='*60}")
    
    # ==================== PHASE 3: PLAYER STATS (MEDIUM PRIORITY) ====================
    
    def collect_key_player_stats(self, top_n_per_team=3):
        """
        Day 7-10: Collect stats for key players
        Focus on top scorers/assist leaders per team
        
        Uses: ~60 API calls (3 players × 20 teams)
        """
        print("\n" + "="*60)
        print("PHASE 3: COLLECTING KEY PLAYER STATS")
        print("="*60)
        
        # Get players who appear most in lineups (key players)
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    p.player_id,
                    p.player_name,
                    ml.team_id,
                    t.team_name,
                    COUNT(*) as appearances
                FROM match_lineups ml
                JOIN players p ON ml.player_id = p.player_id
                JOIN teams t ON ml.team_id = t.team_id
                WHERE ml.is_starter = true
                GROUP BY p.player_id, p.player_name, ml.team_id, t.team_name
                ORDER BY ml.team_id, appearances DESC
            """)
            
            all_players = cursor.fetchall()
        
        # Get top N players per team
        from collections import defaultdict
        players_by_team = defaultdict(list)
        
        for player in all_players:
            if len(players_by_team[player['team_id']]) < top_n_per_team:
                players_by_team[player['team_id']].append(player)
        
        # Flatten
        key_players = [p for team_players in players_by_team.values() for p in team_players]
        
        print(f"\nIdentified {len(key_players)} key players")
        print(f"This will use ~{len(key_players)} API calls")
        
        proceed = input("\nProceed? (yes/no): ")
        if proceed.lower() != 'yes':
            return
        
        collected = 0
        current_season = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
        
        for idx, player in enumerate(key_players, 1):
            print(f"\n[{idx}/{len(key_players)}] {player['player_name']} ({player['team_name']})...")
            
            try:
                stats = self.collector.get_player_statistics(player['player_id'], current_season)
                
                if stats:
                    # Get league_id for this player's team
                    with self.db.get_cursor() as cursor:
                        cursor.execute("""
                            SELECT DISTINCT league_id 
                            FROM matches 
                            WHERE (home_team_id = %s OR away_team_id = %s) 
                            AND season = %s 
                            LIMIT 1
                        """, (player['team_id'], player['team_id'], current_season))
                        
                        result = cursor.fetchone()
                        league_id = result['league_id'] if result else 39
                    
                    # Insert stats
                    self.db.insert_player_season_stats({
                        'player_id': player['player_id'],
                        'team_id': player['team_id'],
                        'league_id': league_id,
                        'season': current_season,
                        'appearances': stats.get('appearances', 0),
                        'minutes_played': stats.get('minutes_played', 0),
                        'goals': stats.get('goals', 0),
                        'assists': stats.get('assists', 0),
                        'shots_total': stats.get('shots_total', 0),
                        'shots_on_target': stats.get('shots_on_target', 0),
                        'passes_total': stats.get('passes_total', 0),
                        'passes_completed': stats.get('passes_total', 0),  # Approximate
                        'tackles': stats.get('tackles', 0),
                        'yellow_cards': stats.get('yellow_cards', 0),
                        'red_cards': stats.get('red_cards', 0),
                        'average_rating': stats.get('rating')
                    })
                    
                    print(f"  ✓ Saved stats")
                    collected += 1
                else:
                    print(f"  ✗ No stats available")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"PHASE 3 COMPLETE")
        print(f"  Stats collected: {collected}")
        print(f"{'='*60}")
    
    # ==================== PHASE 4: WEATHER (LOW PRIORITY BUT EASY) ====================
    
    def collect_weather_for_matches(self, num_matches=50):
        """
        Day 11-12: Add weather data for recent matches
        
        Uses OpenWeather API (1000 calls/day - separate limit!)
        """
        print("\n" + "="*60)
        print("PHASE 4: COLLECTING WEATHER DATA")
        print("="*60)
        
        # Get recent matches without weather
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT m.match_id, m.fixture_date, m.home_team_id,
                       t.stadium_lat, t.stadium_lon
                FROM matches m
                JOIN teams t ON m.home_team_id = t.team_id
                LEFT JOIN match_weather w ON m.match_id = w.match_id
                WHERE m.match_status = 'FT'
                AND w.weather_id IS NULL
                AND t.stadium_lat IS NOT NULL
                ORDER BY m.fixture_date DESC
                LIMIT %s
            """, (num_matches,))
            
            matches = cursor.fetchall()
        
        print(f"\nFound {len(matches)} matches needing weather data")
        
        collected = 0
        
        for idx, match in enumerate(matches, 1):
            print(f"\n[{idx}/{len(matches)}] Match {match['match_id']}...")
            
            try:
                weather = self.collector.get_historical_weather(
                    match['stadium_lat'],
                    match['stadium_lon'],
                    match['fixture_date']
                )
                
                if weather:
                    weather['match_id'] = match['match_id']
                    self.db.insert_weather(weather)
                    print(f"  ✓ {weather.get('weather_condition')}, {weather.get('temperature')}°C")
                    collected += 1
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"PHASE 4 COMPLETE")
        print(f"  Weather data collected: {collected}")
        print(f"{'='*60}")
    
    # ==================== MASTER PLAN ====================
    
    def run_master_plan(self):
        """
        Execute complete data collection over ~2 weeks
        """
        print("\n" + "="*60)
        print("API DATA COLLECTION - MASTER PLAN")
        print("="*60)
        
        print("\nThis plan will collect:")
        print("  1. Lineups (80 matches) - Days 1-5")
        print("  2. Injuries (all teams) - Day 6")
        print("  3. Player stats (top 3 per team) - Days 7-10")
        print("  4. Weather data (50 matches) - Days 11-12")
        print("\nTotal API calls: ~200 (spread over 12 days)")
        print("Daily limit: 100 calls")
        
        choice = input("\nWhat would you like to do?\n1. Run Phase 1 (Lineups)\n2. Run Phase 2 (Injuries)\n3. Run Phase 3 (Player Stats)\n4. Run Phase 4 (Weather)\n5. Check current progress\n\nChoice: ")
        
        if choice == '1':
            self.collect_lineups_batch(80)
        elif choice == '2':
            self.collect_current_injuries()
        elif choice == '3':
            self.collect_key_player_stats(3)
        elif choice == '4':
            self.collect_weather_for_matches(50)
        elif choice == '5':
            self.show_progress()
    
    def show_progress(self):
        """Show current data collection status"""
        stats = self.db.get_database_stats()
        
        print("\n" + "="*60)
        print("CURRENT DATA STATUS")
        print("="*60)
        
        # Check lineup coverage
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT m.match_id) as total_matches,
                    COUNT(DISTINCT ml.match_id) as matches_with_lineups
                FROM matches m
                LEFT JOIN match_lineups ml ON m.match_id = ml.match_id
                WHERE m.match_status = 'FT'
            """)
            
            lineup_stats = cursor.fetchone()
        
        lineup_coverage = (lineup_stats['matches_with_lineups'] / lineup_stats['total_matches'] * 100) if lineup_stats['total_matches'] > 0 else 0
        
        print(f"\nLineups: {lineup_stats['matches_with_lineups']}/{lineup_stats['total_matches']} ({lineup_coverage:.1f}%)")
        print(f"Players: {stats['total_players']}")
        print(f"Injuries: (check database)")
        print(f"Weather: (check database)")
    
    def close(self):
        self.db.close()


if __name__ == "__main__":
    collector = StrategicAPICollector()
    
    try:
        collector.run_master_plan()
    finally:
        collector.close()