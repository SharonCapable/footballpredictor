"""
Hybrid Data Pipeline
Uses Kaggle datasets for historical data + API-Football for recent/upcoming matches
"""

import pandas as pd
import os
from datetime import datetime, timedelta
from data_collector import ComprehensiveFootballCollector
from database_manager import FootballDatabaseManager
import time

class HybridDataPipeline:
    def __init__(self, api_football_key, weather_api_key, db_config):
        """Initialize hybrid pipeline"""
        self.collector = ComprehensiveFootballCollector(api_football_key, weather_api_key)
        self.db = FootballDatabaseManager(**db_config)
        
        self.leagues = {
            'Premier League': {'id': 39, 'country': 'England'},
            'Primeira Liga': {'id': 94, 'country': 'Portugal'}
        }
    
    def load_kaggle_data(self, kaggle_csv_path):
        """Load historical data from Kaggle CSV"""
        print(f"\n{'='*60}")
        print(f"Loading Kaggle dataset: {kaggle_csv_path}")
        print(f"{'='*60}")
        
        df = pd.read_csv(kaggle_csv_path)
        print(f"✓ Loaded {len(df)} matches from CSV")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def transform_kaggle_to_database_format(self, df, league_id, league_name):
        """Transform Kaggle CSV format to our database format"""
        print(f"\nTransforming data for {league_name}...")
        
        # Common column mappings
        column_mapping = {
            'Date': 'date',
            'HomeTeam': 'home_team_name',
            'AwayTeam': 'away_team_name',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'FTR': 'result',
            'HTHG': 'home_halftime_goals',
            'HTAG': 'away_halftime_goals'
        }
        
        df_clean = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_clean.columns:
                df_clean = df_clean.rename(columns={old_col: new_col})
        
        # Parse date
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        
        # Extract season
        def get_season(date):
            if pd.isna(date):
                return None
            year = date.year
            month = date.month
            return year if month >= 8 else year - 1
        
        df_clean['season'] = df_clean['date'].apply(get_season)
        
        # Outcome
        if 'result' in df_clean.columns:
            df_clean['outcome'] = df_clean['result']
        else:
            def get_outcome(row):
                if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
                    return None
                if row['home_goals'] > row['away_goals']:
                    return 'H'
                elif row['home_goals'] < row['away_goals']:
                    return 'A'
                else:
                    return 'D'
            df_clean['outcome'] = df_clean.apply(get_outcome, axis=1)
        
        outcome_numeric_map = {'H': 1, 'D': 0, 'A': 2}
        df_clean['outcome_numeric'] = df_clean['outcome'].map(outcome_numeric_map)
        
        df_clean['league_id'] = league_id
        df_clean['league_name'] = league_name
        df_clean['match_status'] = 'FT'
        
        df_clean = df_clean.dropna(subset=['date', 'home_team_name', 'away_team_name', 
                                           'home_goals', 'away_goals'])
        
        print(f"✓ Transformed {len(df_clean)} valid matches")
        print(f"Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
        
        return df_clean
    
    def create_team_mapping(self, df):
        """Create team name to ID mapping"""
        all_teams = set(df['home_team_name'].unique()) | set(df['away_team_name'].unique())
        
        team_mapping = {}
        for idx, team_name in enumerate(sorted(all_teams), start=1000):
            team_mapping[team_name] = {
                'team_id': idx,
                'team_name': team_name
            }
        
        print(f"✓ Created mapping for {len(team_mapping)} teams")
        return team_mapping
    
    def insert_kaggle_data_to_db(self, df, team_mapping, league_id):
        """Insert Kaggle historical data into database"""
        print(f"\n{'='*60}")
        print("Inserting Kaggle data into database...")
        print(f"{'='*60}")
        
        # Insert teams
        print("\n1. Inserting teams...")
        for team_name, team_info in team_mapping.items():
            self.db.insert_team({
                'team_id': team_info['team_id'],
                'team_name': team_name
            })
        print(f"✓ Inserted {len(team_mapping)} teams")
        
        # Insert matches
        print("\n2. Inserting matches...")
        inserted = 0
        
        for idx, row in df.iterrows():
            try:
                home_team_id = team_mapping[row['home_team_name']]['team_id']
                away_team_id = team_mapping[row['away_team_name']]['team_id']
                
                match_id = int(f"{league_id}{row['date'].strftime('%Y%m%d')}{home_team_id % 100}")
                
                match_data = {
                    'match_id': match_id,
                    'fixture_date': row['date'],
                    'league_id': league_id,
                    'season': int(row['season']),
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_goals': int(row['home_goals']),
                    'away_goals': int(row['away_goals']),
                    'home_halftime_goals': int(row.get('home_halftime_goals', 0)) if pd.notna(row.get('home_halftime_goals')) else None,
                    'away_halftime_goals': int(row.get('away_halftime_goals', 0)) if pd.notna(row.get('away_halftime_goals')) else None,
                    'match_status': 'FT',
                    'outcome': row['outcome'],
                    'outcome_numeric': int(row['outcome_numeric'])
                }
                
                self.db.insert_match(match_data)
                inserted += 1
                
                if inserted % 100 == 0:
                    print(f"  Inserted {inserted}/{len(df)} matches...")
                
            except Exception as e:
                print(f"  ✗ Error inserting match {idx}: {e}")
                continue
        
        print(f"✓ Inserted {inserted} matches")
        return inserted
    
    def run_hybrid_pipeline(self, kaggle_files):
        """Run complete hybrid pipeline"""
        print("\n" + "="*60)
        print("HYBRID DATA PIPELINE - STARTING")
        print("="*60)
        
        total_matches = 0
        
        for league_name, csv_path in kaggle_files.items():
            if league_name not in self.leagues:
                print(f"✗ Unknown league: {league_name}")
                continue
            
            league_info = self.leagues[league_name]
            league_id = league_info['id']
            
            print(f"\n{'='*60}")
            print(f"Processing {league_name}")
            print(f"{'='*60}")
            
            # Insert league
            self.db.insert_league(league_id, league_name, league_info['country'])
            
            # Load and transform data
            df = self.load_kaggle_data(csv_path)
            df_clean = self.transform_kaggle_to_database_format(df, league_id, league_name)
            
            # Create team mapping
            team_mapping = self.create_team_mapping(df_clean)
            
            # Insert into database
            matches_inserted = self.insert_kaggle_data_to_db(df_clean, team_mapping, league_id)
            total_matches += matches_inserted
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print(f"Total matches: {total_matches}")
        print("="*60)
        
        self.generate_report()
    
    def generate_report(self):
        """Generate summary report"""
        stats = self.db.get_database_stats()
        
        print("\n" + "-"*60)
        print("DATABASE SUMMARY")
        print("-"*60)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title():<30}: {value:>10}")
        print("-"*60)
    
    def close(self):
        """Close database connections"""
        self.db.close()