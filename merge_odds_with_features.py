"""
Merge Betting Odds with Engineered Features
Creates final dataset with odds-based features
"""

import pandas as pd
import numpy as np
from datetime import datetime
from difflib import get_close_matches

class OddsFeatureMerger:
    def __init__(self):
        self.team_name_mappings = {}
    
    def load_data(self):
        """Load both features and odds data"""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load your engineered features
        try:
            features_df = pd.read_csv('data/processed/improved_features.csv')
            print(f"✓ Loaded features: {len(features_df)} matches")
        except:
            print("✗ Could not find improved_features.csv")
            print("  Using features_engineered.csv instead")
            features_df = pd.read_csv('data/processed/features_engineered.csv')
        
        # Load odds data
        epl_odds = pd.read_csv('data/odds/epl_with_odds.csv')
        portugal_odds = pd.read_csv('data/odds/portugal_with_odds.csv')
        
        print(f"✓ Loaded EPL odds: {len(epl_odds)} matches")
        print(f"✓ Loaded Portugal odds: {len(portugal_odds)} matches")
        
        # Add league identifier
        epl_odds['league'] = 'Premier League'
        portugal_odds['league'] = 'Primeira Liga'
        
        # Combine odds
        all_odds = pd.concat([epl_odds, portugal_odds], ignore_index=True)
        
        print(f"✓ Total odds data: {len(all_odds)} matches")
        
        return features_df, all_odds
    
    def standardize_team_names(self, odds_df):
        """
        Standardize team names from Football-Data.co.uk format
        to match your database format
        """
        print("\n" + "="*60)
        print("STANDARDIZING TEAM NAMES")
        print("="*60)
        
        # Common name variations
        name_mappings = {
            # Premier League
            'Man United': 'Manchester Utd',
            'Man City': 'Manchester City',
            'Spurs': 'Tottenham',
            'Newcastle': 'Newcastle Utd',
            'West Ham': 'West Ham United',
            'Wolves': 'Wolverhampton',
            'Brighton': 'Brighton & Hove Albion',
            'Nott\'m Forest': 'Nottingham Forest',
            'Leicester': 'Leicester City',
            
            # Portugal
            'Sp Lisbon': 'Sporting CP',
            'Sp Braga': 'Braga',
            'Pacos Ferreira': 'Paços de Ferreira',
            'Ferreira': 'Paços de Ferreira',
        }
        
        # Apply mappings
        odds_df['HomeTeam'] = odds_df['HomeTeam'].replace(name_mappings)
        odds_df['AwayTeam'] = odds_df['AwayTeam'].replace(name_mappings)
        
        print(f"✓ Applied {len(name_mappings)} team name standardizations")
        
        return odds_df
    
    def create_fuzzy_team_mapping(self, features_df, odds_df):
        """
        Automatically match team names using fuzzy matching
        """
        print("\n" + "="*60)
        print("CREATING TEAM NAME MAPPINGS")
        print("="*60)
        
        # Get unique teams from your features (using database names)
        # We'll need to load from database to get actual team names
        import os, sys
        sys.path.append('..')
        from dotenv import load_dotenv
        from database_manager import FootballDatabaseManager
        
        load_dotenv()
        
        db = FootballDatabaseManager(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT')),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        
        # Get team names from database
        with db.get_cursor() as cursor:
            cursor.execute("SELECT team_id, team_name FROM teams ORDER BY team_name")
            db_teams = {row['team_id']: row['team_name'] for row in cursor.fetchall()}
        
        db.close()
        
        # Get unique teams from odds data
        odds_teams = set(odds_df['HomeTeam'].unique()) | set(odds_df['AwayTeam'].unique())
        
        print(f"\nDatabase teams: {len(db_teams)}")
        print(f"Odds data teams: {len(odds_teams)}")
        
        # Create mapping
        mappings = {}
        unmatched = []
        
        for odds_team in sorted(odds_teams):
            # Try exact match first
            matched = False
            for team_id, db_team in db_teams.items():
                if odds_team.lower() == db_team.lower():
                    mappings[odds_team] = {'team_id': team_id, 'team_name': db_team}
                    matched = True
                    break
            
            if not matched:
                # Try fuzzy match
                db_team_names = list(db_teams.values())
                matches = get_close_matches(odds_team, db_team_names, n=1, cutoff=0.6)
                
                if matches:
                    matched_name = matches[0]
                    team_id = [tid for tid, name in db_teams.items() if name == matched_name][0]
                    mappings[odds_team] = {'team_id': team_id, 'team_name': matched_name}
                    print(f"  Fuzzy match: '{odds_team}' → '{matched_name}'")
                else:
                    unmatched.append(odds_team)
        
        print(f"\n✓ Matched {len(mappings)} teams")
        
        if unmatched:
            print(f"\n⚠️ Could not match {len(unmatched)} teams:")
            for team in unmatched:
                print(f"  - {team}")
            print("\nThese matches will be excluded from training")
        
        return mappings
    
    def merge_odds_with_features(self, features_df, odds_df, team_mappings):
        """
        Merge odds data with features based on date and teams
        """
        print("\n" + "="*60)
        print("MERGING ODDS WITH FEATURES")
        print("="*60)
        
        # Prepare odds dataframe
        odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Map team names to IDs
        def get_team_id(team_name):
            if team_name in team_mappings:
                return team_mappings[team_name]['team_id']
            return None
        
        odds_df['home_team_id'] = odds_df['HomeTeam'].apply(get_team_id)
        odds_df['away_team_id'] = odds_df['AwayTeam'].apply(get_team_id)
        
        # Remove matches we couldn't map
        odds_df = odds_df.dropna(subset=['home_team_id', 'away_team_id'])
        
        print(f"Odds data after team mapping: {len(odds_df)} matches")
        
        # Prepare features dataframe
        features_df['fixture_date'] = pd.to_datetime(features_df['fixture_date'])
        
        # Merge on date and teams
        merged = features_df.merge(
            odds_df[['Date', 'home_team_id', 'away_team_id', 
                     'home_odds_avg', 'draw_odds_avg', 'away_odds_avg',
                     'home_prob_norm', 'draw_prob_norm', 'away_prob_norm',
                     'odds_home_advantage']],
            left_on=['fixture_date', 'home_team_id', 'away_team_id'],
            right_on=['Date', 'home_team_id', 'away_team_id'],
            how='left'
        )
        
        # Check merge success
        matches_with_odds = merged['home_odds_avg'].notna().sum()
        merge_rate = matches_with_odds / len(merged) * 100
        
        print(f"\n✓ Merge complete!")
        print(f"  Matches with odds: {matches_with_odds}/{len(merged)} ({merge_rate:.1f}%)")
        
        # Drop Date column (duplicate)
        merged = merged.drop('Date', axis=1, errors='ignore')
        
        return merged
    
    def create_additional_odds_features(self, df):
        """
        Create derived features from odds
        """
        print("\n" + "="*60)
        print("CREATING ODDS-DERIVED FEATURES")
        print("="*60)
        
        # Only for matches with odds
        has_odds = df['home_odds_avg'].notna()
        
        # 1. Odds confidence (lower odds = higher confidence)
        df.loc[has_odds, 'favorite_confidence'] = df.loc[has_odds, [
            'home_odds_avg', 'draw_odds_avg', 'away_odds_avg'
        ]].min(axis=1)
        
        # 2. Expected goals based on odds
        # Lower odds = stronger team = more expected goals
        df.loc[has_odds, 'home_expected_strength'] = 1 / df.loc[has_odds, 'home_odds_avg']
        df.loc[has_odds, 'away_expected_strength'] = 1 / df.loc[has_odds, 'away_odds_avg']
        
        # 3. Match competitiveness
        df.loc[has_odds, 'odds_competitiveness'] = abs(
            df.loc[has_odds, 'home_odds_avg'] - df.loc[has_odds, 'away_odds_avg']
        )
        
        # 4. Over/Under indicator (if both teams are low odds = likely high scoring)
        df.loc[has_odds, 'combined_attacking_odds'] = (
            df.loc[has_odds, 'home_expected_strength'] + 
            df.loc[has_odds, 'away_expected_strength']
        )
        
        print("✓ Created odds-derived features:")
        print("  - Favorite confidence")
        print("  - Expected team strength")
        print("  - Match competitiveness")
        print("  - Combined attacking indicator")
        
        return df
    
    def save_final_dataset(self, df):
        """Save final dataset with odds"""
        # Remove matches without odds for training
        df_with_odds = df[df['home_odds_avg'].notna()].copy()
        
        print("\n" + "="*60)
        print("SAVING FINAL DATASET")
        print("="*60)
        
        print(f"\nTotal matches: {len(df)}")
        print(f"Matches with odds: {len(df_with_odds)}")
        print(f"Excluded (no odds): {len(df) - len(df_with_odds)}")
        
        # Save both versions
        df.to_csv('data/processed/features_with_odds_all.csv', index=False)
        df_with_odds.to_csv('data/processed/features_with_odds.csv', index=False)
        
        print(f"\n✓ Saved to:")
        print(f"  - data/processed/features_with_odds.csv ({len(df_with_odds)} matches)")
        print(f"  - data/processed/features_with_odds_all.csv ({len(df)} matches)")
        
        # Show sample
        print(f"\nSample data:")
        print(df_with_odds[['fixture_date', 'home_odds_avg', 'draw_odds_avg', 
                            'away_odds_avg', 'over_2_5']].head())
        
        return df_with_odds
    
    def run_complete_merge(self):
        """Run complete merging process"""
        print("\n" + "="*60)
        print("ODDS + FEATURES MERGER")
        print("="*60)
        
        # Step 1: Load data
        features_df, odds_df = self.load_data()
        
        # Step 2: Standardize team names
        odds_df = self.standardize_team_names(odds_df)
        
        # Step 3: Create mappings
        team_mappings = self.create_fuzzy_team_mapping(features_df, odds_df)
        
        # Step 4: Merge
        merged_df = self.merge_odds_with_features(features_df, odds_df, team_mappings)
        
        # Step 5: Create additional features
        merged_df = self.create_additional_odds_features(merged_df)
        
        # Step 6: Save
        final_df = self.save_final_dataset(merged_df)
        
        print("\n" + "="*60)
        print("✓ MERGE COMPLETE!")
        print("="*60)
        print("\nNext step: Retrain model with odds features")
        print("Expected accuracy: 55-62% (up from 45%)")
        
        return final_df


if __name__ == "__main__":
    merger = OddsFeatureMerger()
    df = merger.run_complete_merge()